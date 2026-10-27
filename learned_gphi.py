"""
LBYL W3 — the LEARNED consistency head g_phi (model).

A small head on top of FROZEN features: it takes a control's PIXEL-channel feature
(vision-encoded rendered crop) and STRUCTURAL-channel feature (text-encoded aria/role)
and predicts whether the two channels DISAGREE about the control's action (a perceptual
dark pattern). Only this head is trained — the VLM stays frozen (the paper's "controller
on a frozen backbone").

Inductive bias for a *consistency* head: after projecting both channels to a common
space, we feed not just [pix, struct] but also their DIFFERENCE and element-wise PRODUCT,
giving the head a direct cross-modal "do these two disagree?" basis (this is what makes
it a screenshot-vs-DOM comparator, not two independent classifiers — the delta vs Visual
Confused Deputy, which scores each channel against a knowledge base).

Pure torch; tiny (~few MB). Trains fine on CPU (cardless) once features are extracted.
"""
import torch
import torch.nn as nn

# tag one-hot for the structural numeric features
ROLES = ["button", "a", "input", "div", "span"]
NUM_DIM = 1 + 4 + (len(ROLES) + 1)   # opacity + geom(x,y,w,h) + role one-hot(+other)


def numeric_feats(row, vw=1280, vh=720):
    """Structural scalar features from a gphi_data.jsonl row -> list[float] (len NUM_DIM)."""
    try:
        op = float(row.get("opacity", "1"))
    except Exception:
        op = 1.0
    x, y, w, h = row["bbox"]
    geom = [x / vw, y / vh, w / vw, h / vh]
    role = (row.get("role") or "").lower()
    onehot = [1.0 if role == r else 0.0 for r in ROLES]
    onehot.append(0.0 if role in ROLES else 1.0)   # "other"
    return [op] + geom + onehot


class GphiHead(nn.Module):
    def __init__(self, vis_dim, txt_dim, num_dim=NUM_DIM, proj=256, hidden=256, p=0.2):
        super().__init__()
        self.vis = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, proj), nn.GELU())
        self.txt = nn.Sequential(nn.LayerNorm(txt_dim), nn.Linear(txt_dim, proj), nn.GELU())
        self.num = nn.Sequential(nn.Linear(num_dim, 32), nn.GELU())
        # [pix, struct, pix-struct, pix*struct, numeric] -> logit
        self.head = nn.Sequential(
            nn.Linear(proj * 4 + 32, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, 1))

    def forward(self, vis, txt, num):
        pv, pt = self.vis(vis), self.txt(txt)
        cross = torch.cat([pv, pt, pv - pt, pv * pt, self.num(num)], dim=-1)
        return self.head(cross).squeeze(-1)


def predict_proba(model, vis, txt, num):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(vis, txt, num))
