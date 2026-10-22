"""
LBYL ③ DIAGNOSTIC (throwaway) — why does safe_frac=0.00 with the real Qwen in the loop?

For a few g_phi-flagged states, run EACH action and log the full chain so we can see WHERE
'safe' dies: where the frozen agent re-clicks, what the detector says, and the final outcome.
Mirrors train_pi.rollout exactly (reset->observe->apply_recheck->act(warning)->observe->
at_click-> refuse-or-click) plus introspection. Writes nothing.

Run (same env as the real run):
  AGENT=qwen QWEN_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct DETECTOR=learned \
    FEATS=gphi_data/feats.pt HOLDOUT=disguised_decline,fake_close_ad,inverted_consent_toggle \
    THREADS=8 OMP_NUM_THREADS=8 python diag_rollout.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import env as envmod
from policy import ACTIONS
from train_pi import CFG, make_feat_provider, build_detector, build_agent, find_state, translate
from defended_agent import RECHECK_NOTE
from detectors import _hit_element


def diag_one(e, t, agent, detector, tid, hit, safe_ids, action):
    e.reset(); e.observe_dom()
    ops = translate(action, hit, safe_ids)
    if ops:
        e.apply_recheck(ops)
    x, y = agent.act(e, t, warning=RECHECK_NOTE)
    dom2 = e.observe_dom()
    h2 = _hit_element(dom2, x, y)
    score, reason, _ = detector.at_click(tid, dom2, x, y)
    outcome = "refuse" if score >= 1 else e.click(x, y)
    role = (h2 or {}).get("lbyl_role", "∅")
    vis = ((h2 or {}).get("visible_text") or "")[:18]
    raw = (getattr(agent, "last_raw", "") or "").replace("\n", " ")[:50]
    print(f"    {action:9} click=({x:3},{y:3}) on[{role}/{vis!r}] score={score:.0f} "
          f"[{reason[:34]}] => {outcome:7} | raw={raw!r}", flush=True)
    return outcome


def main():
    cfg = CFG
    torch.manual_seed(cfg["SEED"])
    get_feats, vd, td, finfo = make_feat_provider(cfg)
    tasks = envmod.load_manifest()
    train_tasks = [t for t in tasks if t["template"] not in cfg["HOLDOUT"]]
    # one task from each of several distinct templates (spread, not all the same template)
    seen, sample = set(), []
    for t in train_tasks:
        if t["template"] not in seen:
            seen.add(t["template"]); sample.append(t)
        if len(sample) >= 7:
            break
    print(f"[cfg] feats={finfo}\n[diag] {len(sample)} tasks across templates: "
          f"{[t['template'] for t in sample]}", flush=True)
    detector = build_detector(cfg)
    agent = build_agent(cfg)
    e = envmod.GUIEnv(headless=True)
    from collections import Counter
    oc = Counter()
    try:
        for t in sample:
            e.load(t)
            st = find_state(e, t, agent, detector)
            print(f"\n=== {t['id']} ({t['template']}) goal={t['goal'][:46]!r} ===", flush=True)
            if st is None:
                print("  NOT flagged on first pass (no policy decision needed)", flush=True)
                continue
            hit, safe_ids, nf = st
            print(f"  flagged: role={hit.get('lbyl_role')} aria={ (hit.get('aria_label') or '')[:30]!r} "
                  f"n_safe_candidates={len(safe_ids)} n_flagged={nf}", flush=True)
            for a in ACTIONS:
                oc[diag_one(e, t, agent, detector, t["id"], hit, safe_ids, a)] += 1
    finally:
        e.close()
    print(f"\n[outcome totals across all actions] {dict(oc)}", flush=True)
    if hasattr(detector, "n_fallback"):
        print(f"[detector fallback {detector.n_fallback}/{detector.n_calls}]", flush=True)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
