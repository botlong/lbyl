"""
LBYL (Neurocomputing revision) — E3: can the live learned detector be made stable?

Re-run after the P1 fixes (browser hit test, computed accessible names, live click-time encoding,
channel-balanced set v2). The three remedies and their parameters are unchanged from the plan
committed before the first run (results/neucom_20261025/E3_PLAN.md):

  E3a  ensemble   average the eight heads' P(mismatch), flag at 0.5            (primary; no knobs)
  E3b  mixed      per held-out template T: train on realistic rows of the other 13 types plus
                  environment-page rows of the other 13 types, run live on T's tasks only
                  (leave-one-template-out end to end), eight seeds
  E3c  fallback   when a head's P(mismatch) is in [0.3, 0.7], ask the zero-shot prompted VLM
                  (prompted_baseline MODE=neutral) about the same live crop and accessible name;
                  eight seeds, and once on top of the E3a ensemble

All thresholds are fixed in advance (tau 0.5, band [0.3, 0.7]); nothing is tuned on these tasks.
Agents are the deterministic scripted agents of run_e2e_learned.py.

Run:  python e3_stabilize.py       (needs gphi_data_realistic_v2/feats.pt and gphi_data/feats.pt)
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import torch.nn as nn
import env as envmod
import train_gphi
from train_gphi import train_head
from fake_agents import OracleSafe, SalientThenSafe
from detectors import LearnedDetector
from run_e2e_learned import run, get_encoder
from stats_util import mean_ci
import prompted_baseline as pb

SEEDS = list(range(int(os.environ.get("SEEDS", "8"))))
CM_DIR = os.path.join(HERE, os.environ.get("CM_DIR", "gphi_data_realistic_v2"))
ENV_DIR = os.path.join(HERE, "gphi_data")
BAND = (0.3, 0.7)
ARMS = os.environ.get("ARMS", "E3a,E3b,E3c").split(",")
E3C_RUNS = os.environ.get("E3C_RUNS", "all")    # e.g. "0,1,2" or "ens", to split E3c over processes


def load(d):
    blob = torch.load(os.path.join(d, "feats.pt"))
    rows = [json.loads(l) for l in open(os.path.join(d, "gphi_data.jsonl"))]
    return [r for r in rows if r["crop"] in blob["feats"]], blob


class EnsembleHead(nn.Module):
    """Averages member heads' probabilities; returns the logit of the mean (so sigmoid -> mean p)."""
    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, vis, txt, num):
        p = torch.stack([torch.sigmoid(h(vis, txt, num)) for h in self.heads]).mean(0)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))


class FallbackDetector(LearnedDetector):
    """LearnedDetector that asks the prompted VLM about the same live crop and accessible name
    whenever P(mismatch) is in BAND. Verdicts are cached by the live input hash."""
    def __init__(self, head, encoder, vlm, band=BAND):
        super().__init__(head, encoder, no_num=True, tau=0.5)
        self.vlm = vlm
        self.band = band
        self.n_deferred = 0
        self.verdicts = {}

    def prob(self, env, task_id, e, png=None):
        self._p = super().prob(env, task_id, e, png)
        return self._p

    def _score_elem(self, env, task_id, e, png=None):
        s, r = super()._score_elem(env, task_id, e, png)
        if r.startswith("[fallback"):
            return s, r
        p = self._p          # unrounded; the reason string shows p to two decimals only
        if self.band[0] <= p <= self.band[1]:
            self.n_deferred += 1
            img = env.crop(e, png or self._png or env.screenshot())
            key = self.log[-1]["key"]
            if key not in self.verdicts:
                self.verdicts[key] = pb.judge(self.vlm, img, e.get("ax_name", ""), "neutral")
            v = float(self.verdicts[key])
            return v, f"prompted fallback (g_phi P={p:.2f}) verdict={int(v)}"
        return s, r


def evaluate(name, detector, e, tasks):
    res = dict(
        reground=run(f"{name}: reground", SalientThenSafe, "reground", e, tasks, detector),
        safe_reground=run(f"{name}: always-safe reground", OracleSafe, "reground", e, tasks, detector))
    fb = detector.n_fallback
    extra = f", deferred to prompt {detector.n_deferred}" if isinstance(detector, FallbackDetector) else ""
    print(f"    [lookups {detector.n_calls}, heuristic fallback {fb}{extra}]", flush=True)
    return res


def summarize(tag, per_seed):
    print(f"\n=== {tag}: {len(per_seed)} runs (mean +- t 95% half-width) ===", flush=True)
    for key, field in [("reground", "asr"), ("reground", "safe"), ("safe_reground", "safe"),
                       ("safe_reground", "refused")]:
        xs = [r[key][field] for r in per_seed]
        m, h, _ = mean_ci(xs)
        print(f"  {key:14} {field:8} {m:.3f} +- {h:.3f}   per-run {[round(x, 3) for x in xs]}",
              flush=True)


def pooled(results_by_tpl):
    """Combine per-template runs (each on its own tasks) into one task-weighted result."""
    out = {}
    for key in ("reground", "safe_reground"):
        n = sum(r["n"] for r in results_by_tpl)
        out[key] = {f: sum(r[key][f] * r["n"] for r in results_by_tpl) / n
                    for f in ("asr", "safe", "refused")}
    return out


def main():
    train_gphi.NO_NUM = True
    train_gphi.ZERO_VIS = train_gphi.ZERO_TXT = False
    cm_rows, cm = load(CM_DIR)
    env_rows, envb = load(ENV_DIR)
    assert (cm["vis_dim"], cm["txt_dim"]) == (envb["vis_dim"], envb["txt_dim"])
    vd, td = cm["vis_dim"], cm["txt_dim"]
    feats_all = {**cm["feats"], **envb["feats"]}
    enc = get_encoder()
    import darkpatterns
    darkpatterns.generate(20, clear=False)          # the 280-task deployment suite (deterministic)
    tasks = envmod.load_manifest()
    templates = sorted({t["template"] for t in tasks})
    print(f"tasks={len(tasks)} templates={len(templates)} cm_rows={len(cm_rows)} "
          f"env_rows={len(env_rows)} seeds={SEEDS} band={BAND} arms={ARMS}", flush=True)
    e = envmod.GUIEnv(headless=True)
    try:
        heads = [train_head(cm_rows, vd, td, cm["feats"], seed=s).eval() for s in SEEDS]

        if "E3a" in ARMS:
            print("\n##### E3a: ensemble of the deployed heads #####", flush=True)
            res = evaluate("ensemble", LearnedDetector(EnsembleHead(heads), enc,
                                                       no_num=True, tau=0.5), e, tasks)
            summarize("E3a ensemble (single deterministic run)", [res])

        if "E3b" in ARMS:
            print("\n##### E3b: mixed training, leave-one-template-out end to end #####", flush=True)
            per_seed = []
            for s in SEEDS:
                by_tpl = []
                for T in templates:
                    tr = [r for r in cm_rows if r["template"] != T] + \
                         [r for r in env_rows if r["template"] != T]
                    h = train_head(tr, vd, td, feats_all, seed=s).eval()
                    tt = [t for t in tasks if t["template"] == T]
                    det = LearnedDetector(h, enc, no_num=True, tau=0.5)
                    r = dict(reground=run(f"seed{s} {T}: reground", SalientThenSafe, "reground",
                                          e, tt, det),
                             safe_reground=run(f"seed{s} {T}: always-safe reground", OracleSafe,
                                               "reground", e, tt, det), n=len(tt))
                    by_tpl.append(r)
                p = pooled(by_tpl)
                print(f"  seed{s} pooled: ASR={p['reground']['asr']:.3f} "
                      f"safe={p['reground']['safe']:.3f} always-safe={p['safe_reground']['safe']:.3f}",
                      flush=True)
                per_seed.append(p)
            summarize("E3b mixed-training LOTO", per_seed)

        if "E3c" in ARMS:
            print("\n##### E3c: prompted-VLM fallback in the uncertainty band #####", flush=True)
            vlm = pb.VLM(enc.model, enc.proc)       # the same frozen Qwen, prompted zero-shot
            per_seed = []
            for s, h in zip(SEEDS, heads):
                if E3C_RUNS != "all" and str(s) not in E3C_RUNS.split(","):
                    continue
                per_seed.append(evaluate(f"seed{s}+fallback",
                                         FallbackDetector(h, enc, vlm), e, tasks))
            if per_seed:
                summarize("E3c per-seed head + prompted fallback", per_seed)
            if E3C_RUNS == "all" or "ens" in E3C_RUNS.split(","):
                res = evaluate("ensemble+fallback",
                               FallbackDetector(EnsembleHead(heads), enc, vlm), e, tasks)
                summarize("E3c ensemble + prompted fallback (single deterministic run)", [res])
    finally:
        e.close()
    print("\nE3_DONE", flush=True)


if __name__ == "__main__":
    main()
