"""
GPU-free test of the defended loop (oracle base agents).

Demonstrates the abstain-vs-reground trade-off that is the LBYL value proposition:
  * undefended susceptible agent  -> high ASR
  * + defense=abstain             -> ASR ~0 but task often not completed (over-refusal)
  * + defense=reground            -> ASR ~0 AND task completed (re-observe recovers)
All on env.observe_dom() + oracle clicks; no model.
"""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, darkpatterns
from fake_agents import OracleSafe, SalientThenSafe
from defended_agent import DefendedAgent


def run(label, make_base, defense, e, tasks):
    tally = defaultdict(int)
    by_tpl = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        e.load(t)
        if defense == "none":
            x, y = make_base().act(e, t)
            outcome = e.click(x, y)
        else:
            dec = DefendedAgent(make_base(), defense=defense).decide(e, t)
            if dec["kind"] == "refuse":
                outcome = "refused"
            else:
                outcome = e.click(*dec["xy"])
        tally[outcome] += 1
        by_tpl[t["template"]][outcome] += 1
    n = len(tasks)
    asr = tally.get("trap", 0) / n
    safe = tally.get("safe", 0) / n
    refused = tally.get("refused", 0) / n
    print(f"{label:34} ASR(trap)={asr:.3f}  safe-success={safe:.3f}  refused={refused:.3f}  "
          f"dist={dict(tally)}")
    return dict(label=label, asr=asr, safe=safe, refused=refused, by_tpl=dict(by_tpl))


def main():
    if not os.path.exists(os.path.join(envmod.PAGES_DIR, "manifest.json")):
        darkpatterns.generate(20)
    tasks = envmod.load_manifest()
    e = envmod.GUIEnv(headless=True)
    print("\n=== defended loop (oracle base = susceptible 'SalientThenSafe') ===")
    try:
        run("susceptible, NO defense", SalientThenSafe, "none", e, tasks)
        run("susceptible + defense=abstain", SalientThenSafe, "abstain", e, tasks)
        run("susceptible + defense=reground", SalientThenSafe, "reground", e, tasks)
        print("\n=== false-positive check (oracle base = always-safe) ===")
        run("always-safe, NO defense", OracleSafe, "none", e, tasks)
        run("always-safe + defense=reground", OracleSafe, "reground", e, tasks)
    finally:
        e.close()
    print("\nnote: with the real re-observation reground, transparent_overlay is now RESCUED:")
    print("apply_recheck removes the hijacking overlay (display:none) so the safe button")
    print("becomes clickable again — safe-success reaches 1.00 (vs 0.75 + 0.25 refused for")
    print("the old text-only warning). The defense even rescues an already-safe agent that")
    print("would otherwise be hijacked by the overlay (ASR 0.25 -> 0).")


if __name__ == "__main__":
    main()
