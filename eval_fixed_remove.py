"""常量-remove 基线 — 回应评审问题②"π_θ 的 +23 是否一条固定规则即可复刻"。

与 eval_indist.log 完全同协议(同 280 任务池 levels=(0,)、同 60 任务 stride、同 heuristic
检测器、同 frozen Qwen2.5-VL-3B、同 find_state→rollout 通路),唯一区别:把 π_θ 的 argmax
换成常量 "remove"。前两行 abstain / fixed-badge-reground 为锚点,应复现 eval_indist.log 的
0.217/0.233/0.450 与 0.217/0.450/0.233;若锚点吻合而 CONST-REMOVE == π_θ 的 0.683/0.000,
则 +23 无需 RL(RL 的贡献=自动选出 remove)。

用法(等 GPU 空闲后由 runner 调用):
  AGENT=qwen QWEN_PATH=... DETECTOR=heuristic LIMIT=60 HOLDOUT=<11 个训练模板> \
  python3 eval_fixed_remove.py
常量策略不查特征,FEATS 无需设置(默认 synth 仅影响 CFG 打印)。
"""
import os, sys
from collections import defaultdict

M0 = os.environ.get("M0_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M0)

import torch
import env as envmod, darkpatterns
from train_pi import CFG, build_detector, build_agent, find_state, rollout
from eval_pi import run_baseline, report


def run_const(label, agent, detector, e, tasks, action="remove", show_first_raw=False):
    """π_θ 的评测通路(find_state→rollout),动作恒为 `action`,不查特征不走策略网络。"""
    tally = defaultdict(int)
    n_flagged = 0
    for i, t in enumerate(tasks):
        e.load(t)
        st = find_state(e, t, agent, detector)
        if show_first_raw and i == 0 and hasattr(agent, "last_raw"):
            print(f"      [preflight] task0={t['id']} raw_model_output={agent.last_raw!r}", flush=True)
        if st is None:
            x, y = agent.act(e, t)
            tally[e.click(x, y)] += 1
            continue
        hit, safe_ids, nf = st
        n_flagged += 1
        out = rollout(e, t, agent, detector, t.get("id"), hit, safe_ids, action)
        tally["refused" if out == "refuse" else out] += 1
    r = report(label, tally, len(tasks))
    print(f"      (flagged states: {n_flagged}/{len(tasks)})", flush=True)
    return r


def main():
    import transformers
    print(f"[env] torch={torch.__version__} transformers={transformers.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    cfg = CFG
    darkpatterns.generate(20)          # 强制重生成(确定性;避免残留 manifest 口径不一)
    tasks = envmod.load_manifest()
    held = [t for t in tasks if (not cfg["HOLDOUT"]) or t["template"] in cfg["HOLDOUT"]]
    if cfg["LIMIT"]:
        held = held[::max(1, len(held) // cfg["LIMIT"])][:cfg["LIMIT"]]
    print(f"[data] {len(tasks)} tasks total, eval on {len(held)}  "
          f"(agent={cfg['AGENT']} detector={cfg['DETECTOR']})", flush=True)

    detector = build_detector(cfg)
    agent = build_agent(cfg)
    e = envmod.GUIEnv(headless=True)
    try:
        print("\n[susceptible agent] — 锚点两行应复现 eval_indist.log 的 0.233/0.450 行", flush=True)
        run_baseline("abstain (refuse on flag)", agent, "abstain", detector, e, held)
        run_baseline("fixed-policy reground (badge rule)", agent, "reground", detector, e, held)
        run_const("CONST-REMOVE (no RL, same harness)", agent, detector, e, held, show_first_raw=True)

        print("\n[over-intervention check — always-safe agent]", flush=True)
        from fake_agents import OracleSafe
        run_const("CONST-REMOVE on always-safe agent", OracleSafe(), detector, e, held)
    finally:
        e.close()
    print("\nEVAL_FIXED_REMOVE_DONE", flush=True)


if __name__ == "__main__":
    main()
