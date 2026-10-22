"""
LBYL M1 (GPU-free) — prove the screenshot-vs-DOM mismatch signal is separable, and
that gating clicks on it blocks traps while passing safe choices.

Two checks, both on env.observe_dom() (no model needed):
  1. SEPARABILITY: g_phi on each task's trap element vs its safe element.
     Want: traps flagged ~100%, safe ~0%.
  2. DEFENDED (oracle): clicking trap-center -> g_phi flags -> defense refuses
     (outcome blocked, not 'trap'); clicking safe-center -> passes -> 'safe'.
     This shows the mechanism blocks deception without blocking the safe action,
     independent of any particular agent.
"""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, darkpatterns
from consistency import gphi, gphi_at_click


def find(dom, role):
    cands = [e for e in dom if e.get("lbyl_role") == role]
    # for safe/trap pick the one with a real or overlay box
    return cands[0] if cands else None


def main():
    mf = os.path.join(envmod.PAGES_DIR, "manifest.json")
    if not os.path.exists(mf):
        darkpatterns.generate(20)
    tasks = envmod.load_manifest()
    e = envmod.GUIEnv(headless=True)
    sep = defaultdict(lambda: dict(trap_flag=0, safe_flag=0, n=0))
    defended = defaultdict(lambda: dict(trap_blocked=0, safe_blocked=0, n=0))
    miss = []
    try:
        for t in tasks:
            e.load(t)
            dom = e.observe_dom()
            tpl = t["template"]
            trap, safe = find(dom, "trap"), find(dom, "safe")
            # 1. separability on the ground-truth elements
            ts = gphi(trap)[0] if trap else 0
            ss = gphi(safe)[0] if safe else 0
            sep[tpl]["trap_flag"] += int(ts >= 1); sep[tpl]["safe_flag"] += int(ss >= 1)
            sep[tpl]["n"] += 1
            if ts < 1: miss.append((t["id"], "trap-not-flagged", trap and trap.get("visible_text"), trap and trap.get("aria_label")))
            if ss >= 1: miss.append((t["id"], "safe-false-flag", safe and safe.get("visible_text"), safe and safe.get("aria_label")))
            # 2. defended oracle: what does a click at trap/safe center hit, and does g_phi flag it
            for role, key in [("trap", "trap_blocked"), ("safe", "safe_blocked")]:
                c = e.center_of(role)
                if not c: continue
                s, r, hit = gphi_at_click(dom, c[0], c[1])
                if s >= 1: defended[tpl][key] += 1
            defended[tpl]["n"] += 1
    finally:
        e.close()

    print("\n=== 1. SIGNAL SEPARABILITY (g_phi on trap vs safe element) ===")
    print(f"{'template':24} {'trap-flagged':>13} {'safe-flagged':>13}")
    Ttr = Tsf = N = 0
    for tpl, d in sep.items():
        print(f"{tpl:24} {d['trap_flag']:>6}/{d['n']:<6} {d['safe_flag']:>6}/{d['n']:<6}")
        Ttr += d["trap_flag"]; Tsf += d["safe_flag"]; N += d["n"]
    print(f"{'TOTAL':24} {Ttr:>6}/{N:<6} {Tsf:>6}/{N:<6}")
    print(f"  -> trap detection (recall) = {Ttr/N:.3f}   safe false-positive = {Tsf/N:.3f}")

    print("\n=== 2. DEFENDED (oracle clicks, g_phi gate) ===")
    print(f"{'template':24} {'trap-clicks BLOCKED':>20} {'safe-clicks blocked(FP)':>24}")
    Btr = Bsf = M = 0
    for tpl, d in defended.items():
        print(f"{tpl:24} {d['trap_blocked']:>9}/{d['n']:<9} {d['safe_blocked']:>13}/{d['n']:<9}")
        Btr += d["trap_blocked"]; Bsf += d["safe_blocked"]; M += d["n"]
    print(f"{'TOTAL':24} {Btr:>9}/{M:<9} {Bsf:>13}/{M:<9}")
    print(f"  -> trap-click block rate = {Btr/M:.3f}   safe-click false-block = {Bsf/M:.3f}")
    print(f"  -> oracle 'defended ASR' (traps that slip through) = {1 - Btr/M:.3f}  (vs undefended 1.000)")

    if miss:
        print("\n=== mismatches to inspect ===")
        for m in miss[:12]:
            print("  ", m)


if __name__ == "__main__":
    main()
