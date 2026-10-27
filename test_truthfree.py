"""
P1-4 acceptance check: detector decisions must not depend on the generator's ground truth.

For every task and for both scripted click targets (the trap a susceptible agent aims at, and the
safe control), we take the click point first (the scripted agents are oracles by design), then run
the detector twice on the same page: once as is, and once after deleting every data-lbyl attribute
from the live DOM (so neither public_dom() nor the browser hit test can see it). Decisions, hit
elements and, for the learned head, probabilities must be identical.

  python test_truthfree.py                  # heuristic + learned (live, seed 0)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod
from detectors import HeuristicDetector
from run_e2e_learned import build_learned_detector

STRIP = "() => document.querySelectorAll('[data-lbyl]').forEach(el => el.removeAttribute('data-lbyl'))"


def check(name, det, e, tasks):
    diffs, n = [], 0
    for t in tasks:
        for aim in ("trap", "safe"):
            e.load(t)
            xy = e.center_of(aim)
            if xy is None:
                continue
            a = det.at_click(e, t["id"], e.public_dom(), *xy)
            e.load(t)
            e.page.evaluate(STRIP)
            b = det.at_click(e, t["id"], e.public_dom(), *xy)
            n += 1
            same = (a[0] == b[0] and (a[2] or {}).get("lbyl_id") == (b[2] or {}).get("lbyl_id")
                    and a[1] == b[1])
            if not same:
                diffs.append((t["id"], aim, a[:2], b[:2]))
    print(f"[{name}] {n} click checks, {len(diffs)} differ after deleting ground truth", flush=True)
    for d in diffs[:10]:
        print("   ", d)
    return len(diffs)


def main():
    import darkpatterns
    darkpatterns.generate(20, clear=False)
    tasks = envmod.load_manifest()
    e = envmod.GUIEnv()
    try:
        bad = check("heuristic", HeuristicDetector(), e, tasks)
        bad += check("learned (live)", build_learned_detector(0), e, tasks)
    finally:
        e.close()
    print("TRUTHFREE_OK" if bad == 0 else "TRUTHFREE_FAIL", flush=True)


if __name__ == "__main__":
    main()
