"""
GPU-free validation of the difficulty gradient (darkpatterns level knob).

Renders the level 0/1/2 sweep and checks, per (template, difficulty):
  * g_phi trap-recall   : if the model clicked the trap, does the HEURISTIC g_phi catch
                          it? Expect ~1.0 at blatant, and the overlay's recall to DROP at
                          'subtle' (opacity 0.08 > the 0.05 heuristic threshold) — that is
                          the designed hole that motivates the LEARNED head, not a bug.
  * safe-clickable      : the safe target still has a real hit-area at every level
                          (grounding stays easy -> 'fooled' stays separable from 'mis-click').
  * safe-oracle-outcome : an oracle clicking the safe center yields 'safe' (for label
                          templates). For transparent_overlay this is ~0 by design — the
                          overlay hijacks the click; that is the threat, not a grounding miss.

Pure Playwright + heuristic logic, no VLM. Run on the cardless instance.
"""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, darkpatterns
from consistency import gphi_at_click


def main():
    darkpatterns.generate(5, levels=(0, 1, 2))
    tasks = envmod.load_manifest()
    e = envmod.GUIEnv(headless=True)
    rec = defaultdict(lambda: dict(n=0, catch=0, safe_click=0, safe_ok=0))
    detail = {}  # (template) -> structural readout of its level-2 first instance
    try:
        for t in tasks:
            e.load(t)
            dom = e.observe_dom()
            key = (t["template"], t["level"], t["difficulty"])
            r = rec[key]; r["n"] += 1
            tc = e.center_of("trap")
            if tc:
                s, _, _ = gphi_at_click(dom, *tc)
                r["catch"] += int(s >= 1)
            sc = e.center_of("safe")
            if sc:
                r["safe_click"] += 1
                out = e.click(*sc)             # one click; page was fresh after load
                r["safe_ok"] += int(out == "safe")
            # capture a structural readout of one subtle instance per template
            if t["level"] == 2 and t["template"] not in detail:
                trap = next((x for x in dom if x["lbyl_role"] == "trap"), None)
                safe = next((x for x in dom if x["lbyl_role"] == "safe"), None)
                detail[t["template"]] = (trap, safe)
    finally:
        e.close()

    print(f"\n{'template':22} {'level':9} {'n':>3} {'gφ-catch':>9} {'safe-click':>11} {'safe-oracle':>12}")
    for (tpl, lvl, diff), r in sorted(rec.items()):
        n = r["n"]
        print(f"{tpl:22} {diff:9} {n:>3} {r['catch']/n:>9.2f} {r['safe_click']/n:>11.2f} {r['safe_ok']/n:>12.2f}")

    print("\n=== structural ground-truth at the SUBTLE level (visible≠aria must hold) ===")
    for tpl, (trap, safe) in sorted(detail.items()):
        if trap:
            print(f"[{tpl}] TRAP visible={trap['visible_text']!r:40} aria={trap['aria_label']!r} "
                  f"opacity={trap['opacity']}")
        if safe:
            print(f"[{tpl}] SAFE visible={safe['visible_text']!r:40} aria={safe['aria_label']!r}")
    print("\nVERIFY_LEVELS_DONE")


if __name__ == "__main__":
    main()
