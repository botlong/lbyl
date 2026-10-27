"""Second-round review (B-1/B-6): undefended baseline and all four constant correction rules through the
pi_theta evaluation path of eval_fixed_remove.py (find_state -> rollout, heuristic gate, frozen
Qwen2.5-VL-3B), on the in-distribution replay and on the held-out templates.

  RULES=nodef,remove,relabel,highlight,noop  SPLIT=indist|heldout  AGENT=qwen QWEN_PATH=... \
  DETECTOR=heuristic HOLDOUT=<eval templates> [LIMIT=60] python analysis/p2/p2_rl.py
Pages are NOT regenerated here (run darkpatterns.generate(20, clear=False) once beforehand), so
several processes can share them. Writes results/neucom_p2/rl_<split>_<rules>.json.
"""
import json, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
M0 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, M0)
os.chdir(M0)

import env as envmod
from train_pi import CFG, build_detector, build_agent
from eval_pi import report
from eval_fixed_remove import run_const
from fake_agents import OracleSafe

RULES = [r for r in os.environ.get("RULES", "nodef").split(",") if r]
SPLIT = os.environ["SPLIT"]


def run_nodef(label, agent, e, tasks):
    tally = defaultdict(int)
    for t in tasks:
        e.load(t)
        x, y = agent.act(e, t)
        tally[e.click(x, y)] += 1
    return report(label, tally, len(tasks))


def main():
    cfg = CFG
    tasks = envmod.load_manifest()
    held = [t for t in tasks if (not cfg["HOLDOUT"]) or t["template"] in cfg["HOLDOUT"]]
    if cfg["LIMIT"]:
        held = held[::max(1, len(held) // cfg["LIMIT"])][:cfg["LIMIT"]]
    print(f"[data] split={SPLIT} {len(tasks)} tasks total, eval on {len(held)} "
          f"templates={sorted({t['template'] for t in held})} rules={RULES}", flush=True)
    detector = build_detector(cfg)
    agent = build_agent(cfg)
    e = envmod.GUIEnv(headless=True)
    res = {}
    try:
        for rule in RULES:
            if rule == "nodef":
                res["nodef"] = run_nodef("no defense (susceptible)", agent, e, held)
                res["nodef_safe"] = run_nodef("no defense (always-safe)", OracleSafe(), e, held)
            else:
                res[rule] = run_const(f"CONST-{rule.upper()} (susceptible)", agent, detector, e, held,
                                      action=rule)
                res[rule + "_safe"] = run_const(f"CONST-{rule.upper()} (always-safe)", OracleSafe(),
                                                detector, e, held, action=rule)
    finally:
        e.close()
    os.makedirs("results/neucom_p2", exist_ok=True)
    path = f"results/neucom_p2/rl_{SPLIT}_{'-'.join(RULES)}.json"
    json.dump(dict(split=SPLIT, n=len(held), results=res), open(path, "w"), indent=1, default=str)
    print(f"P2_RL_DONE {path}", flush=True)


if __name__ == "__main__":
    main()
