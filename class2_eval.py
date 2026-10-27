"""
LBYL (Neurocomputing revision) — E1 evaluation: zero-shot Class-2 detection.

Trains the deployed g_phi configuration (realistic channel-balanced set, numeric features removed)
over SEEDS training seeds and scores the Class-2 probe set from class2_pages.py. Reports, per
variant, the flag rate (recall for label-1 variants, FPR for honest ones) as mean +- 95% CI
half-width (normal approximation, as in the paper), plus the keyword heuristic.

Run after:  python class2_pages.py 12
            GPHI_DATA_DIR=gphi_data_class2 python extract_feats.py
"""
import os, sys, json, math
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import train_gphi
from train_gphi import train_head, heuristic_pred

TRAIN_DIR = os.path.join(HERE, os.environ.get("CM_DIR", "gphi_data_realistic_v2"))
TEST_DIR = os.path.join(HERE, "gphi_data_class2")
SEEDS = int(os.environ.get("SEEDS", "8"))
ORDER = ["honest_A", "honest_B", "swap_A", "swap_B", "append", "hidden"]


def load(d):
    blob = torch.load(os.path.join(d, "feats.pt"))
    rows = [json.loads(l) for l in open(os.path.join(d, "gphi_data.jsonl"))]
    rows = [r for r in rows if r["crop"] in blob["feats"]]
    return rows, blob


def ci(xs):
    from stats_util import mean_ci
    m, h, _ = mean_ci(xs)       # t interval over seeds
    return m, h


def main():
    tr, tb = load(TRAIN_DIR)
    te, eb = load(TEST_DIR)
    assert (tb["vis_dim"], tb["txt_dim"]) == (eb["vis_dim"], eb["txt_dim"]), "feature dims differ"
    print(f"train {TRAIN_DIR.split('/')[-1]}: {len(tr)} rows | test class2: {len(te)} rows | "
          f"seeds={SEEDS} NO_NUM=1", flush=True)
    train_gphi.NO_NUM = True
    by_var = defaultdict(list)
    for r in te:
        by_var[r["variant"]].append(r)
    rates = defaultdict(list)          # variant -> [flag rate per seed]
    by_tpl = defaultdict(lambda: defaultdict(list))
    for seed in range(SEEDS):
        head = train_head(tr, tb["vis_dim"], tb["txt_dim"], tb["feats"], seed=seed).eval()
        line = []
        for v in ORDER:
            vis, txt, num, _ = train_gphi.tensors(by_var[v], eb["feats"])
            with torch.no_grad():
                flag = (torch.sigmoid(head(vis, txt, num)) >= 0.5).tolist()
            rates[v].append(sum(flag) / len(flag))
            for r, f in zip(by_var[v], flag):
                by_tpl[v][r["template"]].append(int(f))
            line.append(f"{v}={rates[v][-1]:.2f}")
        print(f"  seed{seed}: " + "  ".join(line), flush=True)
    print(f"\n=== Class-2 zero-shot flag rate, {SEEDS} seeds (mean +- 95% CI) ===")
    print(f"  {'variant':9} {'label':>5} {'n':>4} {'g_phi':>14} {'heuristic':>10}")
    for v in ORDER:
        rs = by_var[v]
        m, h = ci(rates[v])
        heur = sum(heuristic_pred(r) for r in rs) / len(rs)
        print(f"  {v:9} {rs[0]['label']:>5} {len(rs):>4} {m:>8.2f} +- {h:.2f} {heur:>10.2f}")
    print("\n  per-template flag rate pooled over seeds (types with any variant < 0.5 flagged):")
    for v in ORDER:
        lab = by_var[v][0]["label"]
        weak = {t: sum(x) / len(x) for t, x in by_tpl[v].items()}
        bad = {t: f for t, f in weak.items() if (f < 0.5 if lab else f >= 0.5)}
        if bad:
            print(f"   {v}: " + ", ".join(f"{t}={f:.2f}" for t, f in sorted(bad.items())))
    print("CLASS2_EVAL_DONE")


if __name__ == "__main__":
    main()
