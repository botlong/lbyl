"""
LBYL — pluggable g_phi DETECTORS for the defended loop.

The defended agent calls a detector to decide whether the control a click would hit is
a perceptual dark pattern. Two implementations share one interface so the SAME end-to-end
loop can run with either:

  at_click(task_id, dom, x, y) -> (score_in_{0,1}, reason, hit_elem | None)
  scan(task_id, dom)           -> [ {lbyl_id, kind, text, reason, role} ]   (for apply_recheck)

HeuristicDetector : the transparent keyword/opacity rule (consistency.py). GPU-free.
LearnedDetector   : the TRAINED cross-modal head g_phi (learned_gphi.GphiHead). It looks up
                    each element's FROZEN (pixel, structural) feature — precomputed offline by
                    extract_feats.py and keyed by crop id "{task_id}_e{lbyl_id}.png" — and runs
                    the head. This is the detector proven shortcut-proof in the cross-modal test
                    (§5.6): zeroing either channel collapses it to chance, so a positive flag is
                    a genuine pixel-vs-structure disagreement, not an aria-text shortcut.

When an element has no precomputed feature (e.g. a zero-area overlay that make_gphi_dataset
skips), the learned detector FALLS BACK to the heuristic for that element so 类2 overlays are
still neutralized; the fallback rate is tracked (.n_fallback / .n_calls) and reported so the
"learned head drove the result" claim stays honest.
"""
import torch
import consistency
from learned_gphi import numeric_feats


def _hit_element(dom, x, y):
    """Topmost interactable under (x,y) — same dispatch rule as consistency.gphi_at_click:
    prefer an invisible overlay / trap covering the point (it wins the real click)."""
    hit = None
    for e in dom:
        b = e["bbox"]
        if b["w"] > 0 and b["h"] > 0 and b["x"] <= x <= b["x"] + b["w"] and b["y"] <= y <= b["y"] + b["h"]:
            hit = e
    for e in dom:
        b = e["bbox"]
        try:
            op = float(e.get("opacity", "1"))
        except Exception:
            op = 1.0
        if b["w"] > 0 and b["h"] > 0 and b["x"] <= x <= b["x"] + b["w"] and b["y"] <= y <= b["y"] + b["h"]:
            if op < 0.05 or e.get("lbyl_role") == "trap":
                hit = e
    return hit


def _looks_invisible(e):
    try:
        op = float(e.get("opacity", "1"))
    except Exception:
        op = 1.0
    bg = (e.get("bg") or "")
    transparent_bg = ("rgba(0, 0, 0, 0)" in bg) or ("rgba(0,0,0,0)" in bg) or bg == "transparent"
    vis = (e.get("visible_text") or "").strip()
    return (op < 0.05) or (transparent_bg and not vis)


class HeuristicDetector:
    """Wraps the transparent M1 rule. task_id is ignored (operates purely on the DOM)."""
    name = "heuristic"

    def at_click(self, task_id, dom, x, y):
        return consistency.gphi_at_click(dom, x, y)

    def scan(self, task_id, dom):
        return consistency.scan(dom)

    def score_elem(self, task_id, e):
        """0/1 flag for a single element (used by pi_theta training to find safe candidates)."""
        s, _ = consistency.gphi(e)
        return 1.0 if s >= 1 else 0.0


class LearnedDetector:
    """The trained cross-modal head g_phi as the live detector.

    head      : a trained learned_gphi.GphiHead (eval mode enforced here).
    feats     : {crop_id: {"vis": tensor, "txt": tensor}} from extract_feats over the ENV pages.
    no_num    : zero the structural scalars (opacity/bbox/role) so the decision rests on the
                pixel-vs-structure comparison only (matches the paper's de-numeric flagship).
    tau       : flag when P(mismatch) >= tau.
    """
    name = "learned"

    def __init__(self, head, feats, no_num=True, tau=0.5):
        self.head = head.eval()
        self.feats = feats
        self.no_num = no_num
        self.tau = tau
        self._heur = HeuristicDetector()
        self.n_calls = 0
        self.n_fallback = 0

    def _crop_id(self, task_id, e):
        return f"{task_id}_e{e.get('lbyl_id')}.png"

    def _score_elem(self, task_id, e):
        """Return (score_in_{0,1}, reason). Falls back to heuristic if no feature exists."""
        cid = self._crop_id(task_id, e)
        f = self.feats.get(cid)
        if f is None:
            self.n_fallback += 1
            s, r = consistency.gphi(e)
            return (1.0 if s >= 1 else 0.0), f"[fallback:heuristic] {r}"
        vis = f["vis"].unsqueeze(0)
        txt = f["txt"].unsqueeze(0)
        row = {"opacity": e.get("opacity", "1"),
               "bbox": [e["bbox"]["x"], e["bbox"]["y"], e["bbox"]["w"], e["bbox"]["h"]],
               "role": e.get("tag", "")}
        num = torch.tensor([numeric_feats(row)], dtype=torch.float32)
        if self.no_num:
            num = torch.zeros_like(num)
        with torch.no_grad():
            p = torch.sigmoid(self.head(vis, txt, num)).item()
        return (1.0 if p >= self.tau else 0.0), f"learned g_phi P(mismatch)={p:.2f}"

    def at_click(self, task_id, dom, x, y):
        self.n_calls += 1
        hit = _hit_element(dom, x, y)
        if hit is None:
            return 0.0, "no element", None
        s, r = self._score_elem(task_id, hit)
        return s, r, hit

    def score_elem(self, task_id, e):
        """0/1 flag for a single element (used by pi_theta training to find safe candidates)."""
        s, _ = self._score_elem(task_id, e)
        return s

    def scan(self, task_id, dom):
        flagged = []
        for e in dom:
            b = e["bbox"]
            if b["w"] <= 0 or b["h"] <= 0:
                # zero-area overlay: heuristic owns it (no crop feature exists)
                s, r = consistency.gphi(e)
                if s >= 1:
                    flagged.append(dict(lbyl_id=e.get("lbyl_id"), kind="remove",
                                        text=(e.get("aria_label") or "a hidden action"),
                                        reason=f"[fallback] {r}", role=e.get("lbyl_role")))
                continue
            s, r = self._score_elem(task_id, e)
            if s >= 1:
                kind = "remove" if _looks_invisible(e) else "badge"
                flagged.append(dict(lbyl_id=e.get("lbyl_id"), kind=kind,
                                    text=(e.get("aria_label") or "a hidden action"),
                                    reason=r, role=e.get("lbyl_role")))
        return flagged
