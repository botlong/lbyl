"""
LBYL ③ — pi_theta re-observation POLICY head (W3-RL).

For ONE g_phi-flagged element (the control the frozen agent is about to click), pi_theta
outputs a distribution over corrective interventions. The chosen intervention is applied to
the page (env.apply_recheck) and the FROZEN agent re-grounds on the corrected view; the
episode outcome is the reward (train_pi.py / GRPO).

Design (locked 2026-06-28, see plans/2026-06-28-pi-theta-rl-design.md):
  * Action space (D1, "加料版" B): {remove, relabel, highlight, noop}. `badge` is EXCLUDED
    from the policy — it is kept only as a baseline (§5.4 showed a frozen agent ignores it).
  * Runs on the SAME frozen (vis, txt) features g_phi uses (extract_feats.py), so only this
    tiny head trains; the VLM stays frozen (keeps the clean "not the base getting stronger"
    ablation). Mirrors learned_gphi.GphiHead's cross-modal basis [pv, pt, pv-pt, pv*pt].
  * Plus a few context scalars so the policy can condition on the situation (e.g. don't pick
    `highlight` when there is no safe candidate to steer toward).
"""
import torch
import torch.nn as nn

# policy action space — order is the index contract used everywhere downstream
ACTIONS = ["remove", "relabel", "highlight", "noop"]
N_ACTIONS = len(ACTIONS)

# context scalars fed alongside the frozen features (keep in sync with ctx_features())
CTX_KEYS = ["gphi_p", "opacity", "n_safe_norm", "n_flagged_norm"]
CTX_DIM = len(CTX_KEYS)


def ctx_features(gphi_p, opacity, n_safe, n_flagged):
    """Build the context vector (plain floats, no grad). n_* are counts -> squashed to [0,1]."""
    try:
        op = float(opacity)
    except Exception:
        op = 1.0
    return [float(gphi_p), op, min(int(n_safe), 5) / 5.0, min(int(n_flagged), 5) / 5.0]


class PiTheta(nn.Module):
    """Tiny policy head on frozen VLM features. forward -> action logits [B, N_ACTIONS]."""

    def __init__(self, vis_dim, txt_dim, hidden=256, ctx_dim=CTX_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.pv = nn.Linear(vis_dim, hidden)
        self.pt = nn.Linear(txt_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(4 * hidden + ctx_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, vis, txt, ctx):
        pv = torch.relu(self.pv(vis))
        pt = torch.relu(self.pt(txt))
        basis = torch.cat([pv, pt, pv - pt, pv * pt, ctx], dim=-1)
        return self.net(basis)

    def dist(self, vis, txt, ctx):
        return torch.distributions.Categorical(logits=self.forward(vis, txt, ctx))
