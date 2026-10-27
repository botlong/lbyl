"""
LBYL §RQ4 — END-TO-END run with the learned g_phi as the LIVE detector in the defended loop.

The deployed head (numeric features removed) is trained on the channel-balanced realistic set
(CM_DIR, default gphi_data_realistic_v2) and runs inside DefendedAgent over the environment
dark-pattern pages. It is live (detectors.LearnedDetector): at each check it screenshots the current
page, crops the control the click would reach (browser hit test), reads its computed accessible name,
and encodes both with the frozen VLM, so after a correction the edited page is what it judges.

Agents are the deterministic scripted agents (fake_agents): the susceptible SalientThenSafe and the
always-safe OracleSafe; the head-training seed is the only source of variance (E2E_SEEDS).

  python run_e2e_learned.py                 # heuristic + learned seed 0
  E2E_SEEDS=0,1,2,3,4,5,6,7 python run_e2e_learned.py
"""
import os, sys, json
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import env as envmod, darkpatterns
import train_gphi
from train_gphi import train_head
from fake_agents import OracleSafe, SalientThenSafe
from defended_agent import DefendedAgent
from detectors import HeuristicDetector, LearnedDetector
from stats_util import mean_ci

CM_DIR = os.environ.get("CM_DIR", "gphi_data_realistic_v2")
CM_FEATS = os.environ.get("CM_FEATS", "feats.pt")
_ENC = None


def get_encoder():
    """One frozen VLM per process, shared by every live detector."""
    global _ENC
    if _ENC is None:
        from encoder import FrozenEncoder, DEFAULT_MODEL
        _ENC = FrozenEncoder(os.environ.get("GPHI_MODEL", DEFAULT_MODEL))
    return _ENC


def run(label, make_base, defense, e, tasks, detector):
    tally = defaultdict(int)
    by_tpl = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        e.load(t)
        if defense == "none":
            x, y = make_base().act(e, t)
            outcome = e.click(x, y)
        else:
            dec = DefendedAgent(make_base(), defense=defense, detector=detector).decide(e, t)
            outcome = "refused" if dec["kind"] == "refuse" else e.click(*dec["xy"])
        tally[outcome] += 1
        by_tpl[t["template"]][outcome] += 1
    n = len(tasks)
    asr = tally.get("trap", 0) / n
    safe = tally.get("safe", 0) / n
    refused = tally.get("refused", 0) / n
    print(f"  {label:42} ASR(trap)={asr:.3f}  safe-success={safe:.3f}  refused={refused:.3f}",
          flush=True)
    return dict(label=label, asr=asr, safe=safe, refused=refused, by_tpl=dict(by_tpl))


def train_deployed_head(seed=0, extra=None, exclude_template=None):
    """The deployed configuration: numeric features off, trained on CM_DIR (plus `extra` = (rows,
    feats) from another set, as in mixed training), optionally leaving one template out."""
    blob = torch.load(os.path.join(HERE, CM_DIR, CM_FEATS))
    feats = dict(blob["feats"])
    rows = [json.loads(l) for l in open(os.path.join(HERE, CM_DIR, "gphi_data.jsonl"))]
    rows = [r for r in rows if r["crop"] in feats]
    if extra is not None:
        xrows, xfeats = extra
        rows = rows + xrows
        feats.update(xfeats)
    if exclude_template is not None:
        rows = [r for r in rows if r["template"] != exclude_template]
    train_gphi.NO_NUM = True
    train_gphi.ZERO_VIS = train_gphi.ZERO_TXT = False
    head = train_head(rows, blob["vis_dim"], blob["txt_dim"], feats, seed=seed)
    train_gphi.NO_NUM = False
    return head.eval()


def build_learned_detector(seed=0):
    head = train_deployed_head(seed)
    print(f"deployed head: {CM_DIR}, seed {seed}, NO_NUM=1; live encoding with "
          f"{get_encoder().model_path.split('/')[-1]}", flush=True)
    return LearnedDetector(head, get_encoder(), no_num=True, tau=0.5)


def eval_detector(name, detector, e, tasks):
    print(f"\n===== detector = {name} =====")
    print("  [susceptible base 'SalientThenSafe']")
    res = {}
    res["none"] = run("no defense", SalientThenSafe, "none", e, tasks, detector)
    res["abstain"] = run("+ defense=abstain", SalientThenSafe, "abstain", e, tasks, detector)
    res["reground"] = run("+ defense=reground", SalientThenSafe, "reground", e, tasks, detector)
    print("  [false-positive check: always-safe base]")
    res["safe_none"] = run("always-safe, no defense", OracleSafe, "none", e, tasks, detector)
    res["safe_reground"] = run("always-safe + defense=reground", OracleSafe, "reground", e, tasks, detector)
    if isinstance(detector, LearnedDetector):
        fb = detector.n_fallback / max(1, detector.n_calls)
        print(f"  [learned detector: {detector.n_calls} click checks, heuristic fallback "
              f"{detector.n_fallback} ({fb:.2%}), {detector.enc.n_encoded} distinct inputs encoded live]")
    return res


def main():
    darkpatterns.generate(20, clear=False)          # the 280-task deployment suite (deterministic)
    tasks = envmod.load_manifest()
    # E2E_SEEDS="0,1,...": retrain the deployed head per seed (default "0" = the R3 run).
    # The scripted agents are deterministic, so head-training seed is the only source of variance.
    seeds = [int(s) for s in os.environ.get("E2E_SEEDS", "0").split(",")]
    e = envmod.GUIEnv(headless=True)
    per_seed = []
    try:
        if os.environ.get("SKIP_HEUR", "0") != "1":
            eval_detector("HEURISTIC (consistency.py, §5.4 baseline)", HeuristicDetector(), e, tasks)
        for s in seeds:
            learned = build_learned_detector(seed=s)
            res = eval_detector(f"LEARNED shortcut-proof g_phi (§5.6 -> deployed), seed {s}",
                                learned, e, tasks)
            per_seed.append((s, res))
    finally:
        e.close()
    if len(per_seed) > 1:
        print(f"\n=== learned g_phi over {len(per_seed)} head seeds (mean +- t 95% half-width) ===")
        for key, field in [("abstain", "asr"), ("reground", "asr"), ("reground", "safe"),
                           ("safe_reground", "safe"), ("safe_reground", "refused")]:
            xs = [r[key][field] for _, r in per_seed]
            m, h, _ = mean_ci(xs)
            print(f"  {key:14} {field:8} {m:.3f} +- {h:.3f}   per-seed {[round(x, 3) for x in xs]}")
    if os.environ.get("E2E_LOG"):
        json.dump(dict(seeds=[s for s, _ in per_seed], results=[r for _, r in per_seed]),
                  open(os.environ["E2E_LOG"], "w"), indent=1)
    print("\nE2E_LEARNED_DONE", flush=True)


if __name__ == "__main__":
    main()
