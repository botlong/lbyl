"""
LBYL ③ — GRPO training of the re-observation policy pi_theta (W3-RL).

Replaces the hard-wired recheck-action choice (detectors.scan: invisible->remove, else->badge;
§5.4 showed a frozen agent IGNORES badge -> recovery capped at 0.25) with a LEARNED policy.
For the g_phi-flagged element the agent is about to click, pi_theta picks a corrective
intervention from {remove, relabel, highlight, noop}; the FROZEN agent re-grounds on the
corrected page; the episode outcome is the reward. Group-relative policy gradient (GRPO-style,
no value net) — mirrors DUDE's GRPO so it doubles as a clean comparison point.

Win condition (D2): maximize SAFE-COMPLETION at ASR≈0. Reward (per episode, with the SAME
safety gate as deployment — see defended_agent.reground):
    +1.0  agent re-grounds to a SAFE (non-flagged) control and the click completes the task
    -0.3  agent still lands on a flagged control -> system refuses (safe but task not done)
    -1.0  a trap actually executes (only if the detector misses the re-click)
    -lam  small cost for any non-noop intervention (curbs over-editing -> protects FPR)

Run (config via env vars):
  # CPU smoke (no GPU, no feats) — verifies the loop learns 'avoid noop, prefer remove/relabel/highlight':
  OMP_NUM_THREADS=2 AGENT=stubborn DETECTOR=heuristic FEATS=synth THREADS=2 \
      /root/miniconda3/bin/python train_pi.py
  # real run (GPU box, single session): true Qwen in the loop + learned g_phi detector + real feats
  AGENT=qwen QWEN_PATH=/root/rivermind-data/Qwen2.5-VL-3B-Instruct DETECTOR=learned \
      FEATS=gphi_data/feats.pt HOLDOUT=disguised_decline,fake_close_ad,inverted_consent_toggle \
      THREADS=8 python train_pi.py
"""
import os, sys, json, hashlib
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch

CFG = dict(
    AGENT=os.environ.get("AGENT", "stubborn"),          # stubborn (CPU smoke) | qwen (real)
    QWEN_PATH=os.environ.get("QWEN_PATH", ""),
    DETECTOR=os.environ.get("DETECTOR", "heuristic"),   # heuristic (smoke) | learned (real)
    FEATS=os.environ.get("FEATS", "synth"),             # synth (smoke) | path to feats.pt
    HOLDOUT=[s for s in os.environ.get("HOLDOUT", "").split(",") if s],  # eval templates; train on rest
    LIMIT=int(os.environ.get("LIMIT", "0")),            # 0 = all tasks; smoke uses a small subset
    K=int(os.environ.get("K", "8")),
    EPOCHS=int(os.environ.get("EPOCHS", "6")),
    LR=float(os.environ.get("LR", "1e-3")),
    LAM=float(os.environ.get("LAM", "0.05")),
    BETA=float(os.environ.get("BETA", "0.01")),         # entropy bonus
    SEED=int(os.environ.get("SEED", "0")),
    THREADS=int(os.environ.get("THREADS", "4")),        # cardless 0.5-core box: use 2
    OUT=os.environ.get("OUT", os.path.join(HERE, "results", "pi_theta.pt")),
    VIS_DIM=int(os.environ.get("VIS_DIM", "2048")),     # used only for synth feats
    TXT_DIM=int(os.environ.get("TXT_DIM", "2048")),
)
torch.set_num_threads(max(1, CFG["THREADS"]))

import env as envmod, darkpatterns
from policy import PiTheta, ACTIONS, ctx_features
from defended_agent import RECHECK_NOTE


def reward(outcome, action, lam):
    base = {"safe": 1.0, "trap": -1.0}.get(outcome, -0.3)   # refuse / none -> -0.3
    return base - (0.0 if action == "noop" else lam)


def make_feat_provider(cfg):
    """Return (get_feats(crop_id)->(vis,txt)|None, vis_dim, txt_dim, info)."""
    fp = cfg["FEATS"]
    if fp and fp != "synth":
        path = fp if os.path.isabs(fp) else os.path.join(HERE, fp)
        blob = torch.load(path)
        feats, vd, td = blob["feats"], blob["vis_dim"], blob["txt_dim"]
        def get(cid):
            f = feats.get(cid)
            return (f["vis"], f["txt"]) if f is not None else None
        return get, vd, td, f"real:{path} ({len(feats)} crops, vis={vd} txt={td})"
    vd, td = cfg["VIS_DIM"], cfg["TXT_DIM"]
    def get(cid):  # deterministic pseudo-features (smoke: reward is action-driven, not feature-driven)
        h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % (2 ** 31)
        g = torch.Generator().manual_seed(h)
        return torch.randn(vd, generator=g), torch.randn(td, generator=g)
    return get, vd, td, f"synth (vis={vd} txt={td})"


def build_detector(cfg):
    if cfg["DETECTOR"] == "learned":
        from run_e2e_learned import build_learned_detector
        return build_learned_detector()
    from detectors import HeuristicDetector
    return HeuristicDetector()


def build_agent(cfg):
    if cfg["AGENT"] == "qwen":
        from agent import QwenVLAgent
        return QwenVLAgent(cfg["QWEN_PATH"])
    from fake_agents import StubbornAgent
    return StubbornAgent()


def translate(action, hit, safe_ids):
    """pi_theta's per-element action -> low-level apply_recheck ops."""
    if action == "remove":
        return [{"lbyl_id": hit["lbyl_id"], "kind": "remove"}]
    if action == "relabel":
        return [{"lbyl_id": hit["lbyl_id"], "kind": "relabel", "text": hit.get("aria_label") or ""}]
    if action == "highlight":
        return [{"lbyl_id": sid, "kind": "highlight"} for sid in safe_ids]
    return []  # noop


def rollout(e, t, agent, detector, tid, hit, safe_ids, action):
    """One episode under `action`, with the SAME safety gate as deployment. -> outcome str."""
    e.reset()
    e.observe_dom()                                  # assigns data-lbyl-id (stable order)
    ops = translate(action, hit, safe_ids)
    if ops:
        e.apply_recheck(ops)
    x, y = agent.act(e, t, warning=RECHECK_NOTE)     # frozen agent re-grounds on corrected page
    dom2 = e.observe_dom()
    score, _, _ = detector.at_click(tid, dom2, x, y)
    if score >= 1:                                   # still flagged -> safety refuse
        return "refuse"
    return e.click(x, y)                             # not flagged -> execute (safe / trap / none)


def find_state(e, t, agent, detector):
    """Frozen agent's first (undefended) click -> the at-click hit element if g_phi flags it.
    Returns (hit, safe_ids, n_flagged) or None when no policy decision is needed."""
    tid = t.get("id")
    x, y = agent.act(e, t)                            # naive first pass
    dom = e.observe_dom()
    score, _, hit = detector.at_click(tid, dom, x, y)
    if hit is None or score < 1:
        return None
    flagged, safe_ids = 0, []
    for el in dom:
        if el["bbox"]["w"] <= 0 or el["bbox"]["h"] <= 0:
            continue
        f = detector.score_elem(tid, el)
        if f >= 1:
            flagged += 1
        elif el.get("lbyl_id") != hit.get("lbyl_id"):
            safe_ids.append(el.get("lbyl_id"))        # non-flagged interactable = safe candidate
    return hit, safe_ids, flagged


def main():
    cfg = CFG
    torch.manual_seed(cfg["SEED"])
    get_feats, vd, td, finfo = make_feat_provider(cfg)
    print(f"[cfg] agent={cfg['AGENT']} detector={cfg['DETECTOR']} feats={finfo} "
          f"K={cfg['K']} epochs={cfg['EPOCHS']} lam={cfg['LAM']} threads={cfg['THREADS']}", flush=True)

    if not os.path.exists(os.path.join(envmod.PAGES_DIR, "manifest.json")):
        darkpatterns.generate(20)
    tasks = envmod.load_manifest()
    train_tasks = [t for t in tasks if t["template"] not in cfg["HOLDOUT"]]
    if cfg["LIMIT"]:
        train_tasks = train_tasks[::max(1, len(train_tasks) // cfg["LIMIT"])][:cfg["LIMIT"]]
    print(f"[data] {len(train_tasks)}/{len(tasks)} train tasks "
          f"(holdout templates: {cfg['HOLDOUT'] or 'none'})", flush=True)

    pi = PiTheta(vd, td)
    opt = torch.optim.Adam(pi.parameters(), lr=cfg["LR"])
    detector = build_detector(cfg)
    agent = build_agent(cfg)
    e = envmod.GUIEnv(headless=True)
    # best-epoch checkpoint: GRPO can collapse in a late epoch (entropy bonus / small K); we keep the
    # policy from the epoch with the highest sampled safe-completion rather than blindly the last one.
    best_state, best_sf, best_epoch = None, -1.0, -1
    try:
        for epoch in range(cfg["EPOCHS"]):
            ep_rewards, act_count, n_states, n_safe_end = [], Counter(), 0, 0
            for t in train_tasks:
                e.load(t)
                st = find_state(e, t, agent, detector)
                if st is None:
                    continue
                hit, safe_ids, n_flagged = st
                tid = t.get("id")
                f = get_feats(f"{tid}_e{hit.get('lbyl_id')}.png")
                if f is None:                          # no precomputed feature -> skip (rare)
                    continue
                vis = f[0].unsqueeze(0); txt = f[1].unsqueeze(0)
                ctx = torch.tensor([ctx_features(1.0, hit.get("opacity", "1"),
                                                 len(safe_ids), n_flagged)], dtype=torch.float32)
                logits = pi(vis, txt, ctx).squeeze(0)  # [N_ACTIONS]
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample((cfg["K"],))        # [K]
                logps = dist.log_prob(acts)            # [K], grad
                rs = []
                for a in acts.tolist():
                    out = rollout(e, t, agent, detector, tid, hit, safe_ids, ACTIONS[a])
                    rs.append(reward(out, ACTIONS[a], cfg["LAM"]))
                    act_count[ACTIONS[a]] += 1
                    if out == "safe":
                        n_safe_end += 1
                r = torch.tensor(rs, dtype=torch.float32)
                adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)  # pop std: K=1 -> 0 not nan
                loss = -(adv.detach() * logps).mean() - cfg["BETA"] * dist.entropy()
                opt.zero_grad(); loss.backward(); opt.step()
                ep_rewards.extend(rs); n_states += 1
            mr = sum(ep_rewards) / max(1, len(ep_rewards))
            safe_frac = n_safe_end / max(1, len(ep_rewards))
            dist_str = " ".join(f"{a}={act_count.get(a,0)}" for a in ACTIONS)
            print(f"[epoch {epoch}] states={n_states} mean_R={mr:+.3f} safe_frac={safe_frac:.2f} "
                  f"| sampled actions: {dist_str}", flush=True)
            if safe_frac > best_sf:                       # strictly-greater: ties keep the EARLIER (more converged) epoch
                best_sf, best_epoch = safe_frac, epoch
                best_state = {k: v.detach().clone() for k, v in pi.state_dict().items()}
        os.makedirs(os.path.dirname(cfg["OUT"]), exist_ok=True)
        save_state = best_state if best_state is not None else pi.state_dict()
        torch.save({"state_dict": save_state, "vis_dim": vd, "txt_dim": td, "actions": ACTIONS,
                    "best_epoch": best_epoch, "best_safe_frac": best_sf,
                    "cfg": {k: v for k, v in cfg.items() if k != "OUT"}}, cfg["OUT"])
        print(f"[saved best epoch={best_epoch} safe_frac={best_sf:.2f}] {cfg['OUT']}\nTRAIN_PI_DONE", flush=True)
    finally:
        e.close()


if __name__ == "__main__":
    main()
