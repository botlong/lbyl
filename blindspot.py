"""
LBYL — honest limitation: the mismatch head's 类3 (consistent-deception) BLIND SPOT.

The mismatch head g_phi compares the pixel channel vs the structural channel. For
类3 attacks (DMAST-style: injected content renders consistently into BOTH channels —
visible label == structural action), there is NO perceptual discrepancy, so g_phi is
DESIGNED to miss them. This script quantifies that: g_phi recall on 类3 should be ~0,
contrasted with ~1.0 on 类1+2. This is the limitation the W6 "consistency-tax frontier"
+ a task-context-plausibility signal must address; we surface it early (plan §2/W6),
not hide it.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, darkpatterns
from consistency import gphi


def recall_on(manifest_name):
    tasks = envmod.load_manifest(manifest_name)
    e = envmod.GUIEnv(headless=True)
    flagged = n = 0
    sample = []
    try:
        for t in tasks:
            e.load(t)
            dom = e.observe_dom()
            trap = next((x for x in dom if x.get("lbyl_role") == "trap"), None)
            if not trap:
                continue
            s, r = gphi(trap)
            flagged += int(s >= 1); n += 1
            if len(sample) < 2:
                sample.append((t["id"], trap.get("visible_text"), trap.get("aria_label"), s, r))
    finally:
        e.close()
    return flagged, n, sample


def main():
    # main 类1+2 suite (regenerate to be safe)
    darkpatterns.generate(20)
    darkpatterns.generate_type3(20)

    f12, n12, _ = recall_on("manifest.json")
    f3, n3, samp3 = recall_on("manifest_t3.json")

    print("\n=== mismatch-head g_phi recall by threat class ===")
    print(f"  类1+2 (perceptual discrepancy)   : {f12}/{n12}  = {f12/n12:.3f}  (caught)")
    print(f"  类3   (consistent deception)     : {f3}/{n3}  = {f3/n3:.3f}  (BLIND SPOT — by design)")
    print("\n类3 trap elements (visible label == structural action -> no discrepancy to detect):")
    for tid, vis, aria, s, r in samp3:
        print(f"  {tid}: visible='{vis}'  aria='{aria}'  ->  g_phi={s}  ({r})")
    print("\nInterpretation: g_phi is a *perceptual-discrepancy* detector. 类3 is a different")
    print("threat (contextual/semantic deception, ~prompt-injection). LBYL scopes to 类1+2 and")
    print("addresses 类3 via the W6 consistency-tax frontier + a task-context-plausibility")
    print("signal (NOT the mismatch head) — see plan §2/§3 W6.")


if __name__ == "__main__":
    main()
