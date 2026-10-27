"""
LBYL — consistency head g_phi (heuristic version, M1).

The paper's core mechanism: a learned head that scores the mismatch between the
PIXEL channel (what the screenshot shows) and the STRUCTURAL channel (DOM /
accessibility tree — what the element actually does). M1 first proves the SIGNAL
is separable with a transparent heuristic before we train the learned head (W3).

Heuristic rule (faithful to "screenshot vs a11y-tree" comparison):
  FLAG if the structural channel (aria_label / true action) reports a risky action
  that the visible label (pixels) does NOT disclose  — i.e. the control looks benign
  but structurally does something else. Plus a structural signal: an (near-)invisible
  interactive element receiving the click is a hijacking overlay (类2).

The structural channel is the browser-computed accessible name (ax_name, see ax.py); rows that
predate it fall back to the raw aria_label. This operates purely on env.public_dom() output → GPU-free.
"""

# risky action concepts the STRUCTURAL channel may reveal, with the keywords that
# would DISCLOSE that action if present in the visible (pixel) label.
RISKY = {
    "accept_tracking": dict(
        struct=["accept", "allow", "consent", "tracking", "all cookies", "personaliz", "enable all"],
        disclosed=["accept", "allow", "consent", "tracking", "all cookies", "personaliz", "enable all"]),
    "install_app": dict(
        struct=["install the app", "install our app", "download the app", "get the app", "download our app"],
        disclosed=["install", "download the app", "get the app", "download our app", "get app"]),
    "paid_subscribe": dict(
        struct=["subscrib", "/month", " per month", "premium", "$", "pay "],
        disclosed=["subscrib", "month", "premium", "pay", "$"]),
    "sponsored": dict(
        struct=["sponsor", "advertis", "third party", "third-party"],
        disclosed=["sponsor", "advertis", "third party", "third-party", " ad "]),
}


def _has(text, kws):
    t = " " + (text or "").lower() + " "
    return any(k in t for k in kws)


def gphi(elem):
    """Return (mismatch_score in {0,1}, reason). elem = an env.observe_dom() entry."""
    vis = (elem.get("visible_text") or "").strip()
    aria = (elem.get("ax_name", elem.get("aria_label")) or "").strip()
    opacity = elem.get("opacity", "1")
    bg = (elem.get("bg") or "")

    # signal 2 (structural / 类2): a near-invisible interactive element taking the click
    try:
        op = float(opacity)
    except Exception:
        op = 1.0
    transparent_bg = ("rgba(0, 0, 0, 0)" in bg) or ("rgba(0,0,0,0)" in bg) or bg == "transparent"
    looks_invisible = (op < 0.05) or (transparent_bg and not vis)
    if looks_invisible and aria:
        return 1.0, f"invisible interactive overlay with hidden action: '{aria}'"

    # signal 1 (label vs action / 类1): structural channel reveals a risky action the
    # visible label does not disclose
    for concept, kw in RISKY.items():
        if _has(aria, kw["struct"]) and not _has(vis, kw["disclosed"]):
            return 1.0, f"visible label '{vis or '∅'}' hides structural action '{aria}' ({concept})"

    return 0.0, "consistent"


def scan(dom):
    """Run g_phi over all interactables; return flagged ones with a recommended
    re-observation action: 'remove' for invisible overlays (类2, can't be safely
    interacted with), 'badge' for label-vs-action deception (类1, mark the truth)."""
    flagged = []
    for e in dom:
        s, r = gphi(e)
        if s >= 1:
            invisible = "invisible interactive overlay" in r
            flagged.append(dict(lbyl_id=e.get("lbyl_id"),
                                kind="remove" if invisible else "badge",
                                text=(e.get("ax_name", e.get("aria_label")) or "a hidden action"),
                                reason=r))
    return flagged
