"""
LBYL ③ — evaluate the trained pi_theta vs baselines (W3-RL), LOTO held-out templates.

Headline metric (D2): SAFE-COMPLETION at ASR≈0. We compare, on the SAME frozen agent and the
SAME g_phi detector, over held-out dark-pattern templates (generalization, not memorization):

  * abstain          : refuse whenever g_phi flags (safe upper bound, completes nothing).
  * fixed-policy      : the hard-wired reground (detectors.scan: invisible->remove else->badge) —
    reground            this is the §5.4 / §5.7 baseline pi_theta must beat.
  * pi_theta (learned): greedy action from the trained policy, same safety-gated rollout.

Plus an over-intervention / FPR check: run the pi_theta path on an always-safe agent and confirm
it does not flag honest controls into broken completions.

Run:
  # CPU smoke (after train_pi.py smoke produced results/pi_theta.pt):
  OMP_NUM_THREADS=2 AGENT=stubborn DETECTOR=heuristic FEATS=synth THREADS=2 \
      HOLDOUT=disguised_decline,fake_close_ad /root/miniconda3/bin/python eval_pi.py
  # real: AGENT=qwen QWEN_PATH=... DETECTOR=learned FEATS=gphi_data/feats.pt HOLDOUT=... python eval_pi.py
"""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch

import env as envmod, darkpatterns
from policy import PiTheta, ACTIONS, ctx_features
from defended_agent import DefendedAgent
import train_pi
from train_pi import CFG, make_feat_provider, build_detector, build_agent, find_state, rollout


def load_policy(path, vd, td):
    blob = torch.load(path)
    assert blob["vis_dim"] == vd and blob["txt_dim"] == td, \
        f"policy feature dims {blob['vis_dim']}x{blob['txt_dim']} != eval feats {vd}x{td}"
    pi = PiTheta(vd, td)
    pi.load_state_dict(blob["state_dict"]); pi.eval()
    return pi


def run_baseline(label, agent, defense, detector, e, tasks):
    tally = defaultdict(int)
    for t in tasks:
        e.load(t)
        dec = DefendedAgent(agent, defense=defense, detector=detector).decide(e, t)
        outcome = "refused" if dec["kind"] == "refuse" else e.click(*dec["xy"])
        tally[outcome] += 1
    return report(label, tally, len(tasks))


def run_pi(label, agent, pi, detector, get_feats, e, tasks):
    tally = defaultdict(int)
    act_count = defaultdict(int)
    for t in tasks:
        e.load(t)
        st = find_state(e, t, agent, detector)
        if st is None:
            # g_phi did not flag the agent's click -> it clicked through; record that outcome
            x, y = agent.act(e, t)
            tally[e.click(x, y)] += 1
            continue
        hit, safe_ids, n_flagged = st
        tid = t.get("id")
        f = get_feats(f"{tid}_e{hit.get('lbyl_id')}.png")
        if f is None:
            tally["refused"] += 1            # no feature -> conservative refuse
            continue
        ctx = torch.tensor([ctx_features(1.0, hit.get("opacity", "1"),
                                         len(safe_ids), n_flagged)], dtype=torch.float32)
        with torch.no_grad():
            logits = pi(f[0].unsqueeze(0), f[1].unsqueeze(0), ctx).squeeze(0)
        action = ACTIONS[int(logits.argmax())]
        act_count[action] += 1
        out = rollout(e, t, agent, detector, tid, hit, safe_ids, action)
        tally["refused" if out == "refuse" else out] += 1
    r = report(label, tally, len(tasks))
    r["actions"] = dict(act_count)
    print(f"      pi_theta action picks: {dict(act_count)}", flush=True)
    return r


def report(label, tally, n):
    asr = tally.get("trap", 0) / n
    safe = tally.get("safe", 0) / n
    refused = tally.get("refused", 0) / n
    print(f"  {label:40} ASR(trap)={asr:.3f}  safe-completion={safe:.3f}  refused={refused:.3f}",
          flush=True)
    return dict(label=label, asr=asr, safe=safe, refused=refused, n=n)


def main():
    cfg = CFG
    torch.manual_seed(cfg["SEED"])
    get_feats, vd, td, finfo = make_feat_provider(cfg)
    if not os.path.exists(cfg["OUT"]):
        sys.exit(f"missing trained policy {cfg['OUT']} — run train_pi.py first")
    pi = load_policy(cfg["OUT"], vd, td)
    print(f"[cfg] agent={cfg['AGENT']} detector={cfg['DETECTOR']} feats={finfo} "
          f"holdout={cfg['HOLDOUT'] or 'NONE (eval==train, smoke only)'}", flush=True)

    if not os.path.exists(os.path.join(envmod.PAGES_DIR, "manifest.json")):
        darkpatterns.generate(20)
    tasks = envmod.load_manifest()
    held = [t for t in tasks if (not cfg["HOLDOUT"]) or t["template"] in cfg["HOLDOUT"]]
    if cfg["LIMIT"]:
        held = held[::max(1, len(held) // cfg["LIMIT"])][:cfg["LIMIT"]]
    print(f"[data] eval on {len(held)} held-out tasks", flush=True)

    detector = build_detector(cfg)
    agent = build_agent(cfg)
    e = envmod.GUIEnv(headless=True)
    try:
        print("\n[susceptible agent — safe-completion is the target]")
        run_baseline("abstain (refuse on flag)", agent, "abstain", detector, e, held)
        run_baseline("fixed-policy reground (§5.4/5.7 baseline)", agent, "reground", detector, e, held)
        run_pi("pi_theta (learned, greedy)", agent, pi, detector, get_feats, e, held)

        print("\n[over-intervention / FPR check — always-safe agent on the same pages]")
        from fake_agents import OracleSafe
        run_pi("pi_theta on always-safe agent", OracleSafe(), pi, detector, get_feats, e, held)

        # Honesty check (§5.6/5.7): the LearnedDetector falls back to the heuristic for any
        # element without a precomputed feature (e.g. zero-area overlays make_gphi_dataset skips).
        # A low rate means the LEARNED cross-modal head — not the heuristic — drove these numbers.
        if hasattr(detector, "n_fallback"):
            fb = detector.n_fallback / max(1, detector.n_calls)
            print(f"\n[learned g_phi fallback-to-heuristic: {detector.n_fallback}/{detector.n_calls} "
                  f"= {fb:.2%}  (low => the learned head drove the result)]", flush=True)
    finally:
        e.close()
    print("\nEVAL_PI_DONE", flush=True)


if __name__ == "__main__":
    main()
