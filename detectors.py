"""
LBYL — pluggable g_phi DETECTORS for the defended loop.

The defended agent calls a detector to decide whether the control a click would hit is a perceptual
dark pattern. Two implementations share one interface so the SAME end-to-end loop can run with either:

  at_click(env, task_id, dom, x, y) -> (score_in_{0,1}, reason, hit_elem | None)
  scan(env, task_id, dom)           -> [ {lbyl_id, kind, text, reason} ]   (for apply_recheck)
  score_elem(env, task_id, e)       -> 0/1 flag for one element

`dom` must be env.public_dom(): detectors never see the generator's ground truth (lbyl_role,
topmost_role). The element a click reaches is the browser's own hit test (env.hit_test, i.e.
elementFromPoint), not a bounding-box rule (Codex review P1-4).

HeuristicDetector : the transparent keyword/opacity rule (consistency.py) over the computed
                    accessible name. GPU-free.
LearnedDetector   : the TRAINED cross-modal head g_phi (learned_gphi.GphiHead), computed LIVE: at
                    every call it screenshots the CURRENT page, crops the element, takes its current
                    computed accessible name and role, and encodes both with the frozen VLM
                    (encoder.FrozenEncoder, cached by content hash). After a correction edits the page
                    the head therefore sees the edited control (Codex review P1-3). Every score is
                    appended to .log with the input hash so before/after-correction inputs are auditable.

Elements too small to crop (< 4 px, e.g. a zero-area overlay) fall back to the heuristic; the
fallback rate is tracked (.n_fallback / .n_calls) and reported.
"""
import torch
import consistency
from ax import struct_text
from learned_gphi import numeric_feats


def hit_element(env, dom, x, y):
    """The enumerated interactable that a click at (x, y) actually reaches, or None."""
    hid = env.hit_test(x, y)
    if hid is None:
        return None
    return next((e for e in dom if e.get("lbyl_id") == hid), None)


def _looks_invisible(e):
    try:
        op = float(e.get("opacity", "1"))
    except Exception:
        op = 1.0
    bg = (e.get("bg") or "")
    transparent_bg = ("rgba(0, 0, 0, 0)" in bg) or ("rgba(0,0,0,0)" in bg) or bg == "transparent"
    vis = (e.get("visible_text") or "").strip()
    return (op < 0.05) or (transparent_bg and not vis)


def _croppable(e):
    return e["bbox"]["w"] >= 4 and e["bbox"]["h"] >= 4


class HeuristicDetector:
    """Wraps the transparent M1 rule. task_id is ignored (operates purely on the public DOM)."""
    name = "heuristic"

    def at_click(self, env, task_id, dom, x, y):
        hit = hit_element(env, dom, x, y)
        if hit is None:
            return 0.0, "no element", None
        s, r = consistency.gphi(hit)
        return s, r, hit

    def scan(self, env, task_id, dom):
        return consistency.scan(dom)

    def score_elem(self, env, task_id, e):
        s, _ = consistency.gphi(e)
        return 1.0 if s >= 1 else 0.0


class LearnedDetector:
    """The trained cross-modal head g_phi as the live detector.

    head    : a trained learned_gphi.GphiHead (eval mode enforced here).
    encoder : encoder.FrozenEncoder (the same frozen VLM that produced the training features).
    no_num  : zero the structural scalars (opacity/bbox/role) so the decision rests on the
              pixel-vs-structure comparison only (the paper's de-numeric deployed configuration).
    tau     : flag when P(mismatch) >= tau.
    """
    name = "learned"

    def __init__(self, head, encoder, no_num=True, tau=0.5):
        self.head = head.eval()
        self.enc = encoder
        self.no_num = no_num
        self.tau = tau
        self.n_calls = 0
        self.n_fallback = 0
        self.log = []
        self._png = None          # screenshot shared by the elements of one scan()

    def features(self, env, e, png=None):
        """Live (vis, txt) features of element e on the current page, plus the input hash."""
        f, key = self.enc.encode(env.crop(e, png or self._png or env.screenshot()), struct_text(e))
        return f, key

    def prob(self, env, task_id, e, png=None):
        f, key = self.features(env, e, png)
        row = {"opacity": e.get("opacity", "1"),
               "bbox": [e["bbox"]["x"], e["bbox"]["y"], e["bbox"]["w"], e["bbox"]["h"]],
               "role": e.get("tag", "")}
        num = torch.tensor([numeric_feats(row)], dtype=torch.float32)
        if self.no_num:
            num = torch.zeros_like(num)
        with torch.no_grad():
            p = torch.sigmoid(self.head(f["vis"].unsqueeze(0), f["txt"].unsqueeze(0), num)).item()
        self.log.append(dict(task=task_id, lbyl_id=e.get("lbyl_id"), key=key, p=round(p, 4),
                             name=e.get("ax_name", ""), vis=e.get("visible_text", "")))
        return p

    def _score_elem(self, env, task_id, e, png=None):
        """Return (score_in_{0,1}, reason). Falls back to the heuristic if e cannot be cropped."""
        if not _croppable(e):
            self.n_fallback += 1
            s, r = consistency.gphi(e)
            return (1.0 if s >= 1 else 0.0), f"[fallback:heuristic] {r}"
        p = self.prob(env, task_id, e, png)
        return (1.0 if p >= self.tau else 0.0), f"learned g_phi P(mismatch)={p:.2f}"

    def at_click(self, env, task_id, dom, x, y):
        self.n_calls += 1
        hit = hit_element(env, dom, x, y)
        if hit is None:
            return 0.0, "no element", None
        s, r = self._score_elem(env, task_id, hit)
        return s, r, hit

    def score_elem(self, env, task_id, e):
        s, _ = self._score_elem(env, task_id, e)
        return s

    def scan(self, env, task_id, dom):
        flagged = []
        self._png = env.screenshot()
        try:
            for e in dom:
                b = e["bbox"]
                if b["w"] <= 0 or b["h"] <= 0:
                    continue                       # not rendered: nothing a click could reach
                s, r = self._score_elem(env, task_id, e)
                if s >= 1:
                    kind = "remove" if _looks_invisible(e) else "badge"
                    flagged.append(dict(lbyl_id=e.get("lbyl_id"), kind=kind,
                                        text=(e.get("ax_name") or "a hidden action"), reason=r))
        finally:
            self._png = None
        return flagged
