"""Task 2 / B⑦ — multi-seed CI for the learned head's flagship LOTO numbers (§5.5).

Runs the leave-one-template-out (cross-type) protocol RUNS times, each with a different head-init
seed, on the DE-NUMERIC config (the reported flagship). Reports the mean ± 95% CI of the LOTO-average
recall / FPR across seeds, so §5.5's 0.93/0.10 gets an interval instead of a single point.

Reuses train_gphi.{load, train_head, evaluate} verbatim (no change to the tested trainer). Variance
comes from head init + dropout during training; the LOTO data splits are deterministic.

Run on the GPU box:
  GPHI_MODEL=... GPHI_DATA_DIR=gphi_data_55 GPHI_NO_NUMERIC=1 CI_RUNS=8 python ci_gphi.py
"""
import os, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

# point train_gphi at the requested dataset BEFORE importing-time globals are used (DATA is read in
# load(); NO_NUMERIC is read at import, so GPHI_NO_NUMERIC must be set in the environment).
import train_gphi as tg
_d = os.environ.get("GPHI_DATA_DIR")
if _d:
    tg.DATA = _d if os.path.isabs(_d) else os.path.join(HERE, _d)

RUNS = int(os.environ.get("CI_RUNS", "8"))


def loto_once(rows, feats, vd, td, off):
    """One full LOTO pass with head-init seeds offset by `off`; returns LOTO-avg (recall, FPR)."""
    templates = sorted(set(r["template"] for r in rows))
    rec = fpr = 0.0
    for i, T in enumerate(templates):
        tr = [r for r in rows if r["template"] != T]
        te = [r for r in rows if r["template"] == T]
        m = tg.train_head(tr, vd, td, feats, seed=1000 * off + i)
        L, _ = tg.evaluate(m, te, feats)
        rec += L["recall"] / len(templates)
        fpr += L["fpr"] / len(templates)
    return rec, fpr


def ci(xs):
    from stats_util import mean_ci
    m, h, _ = mean_ci(xs)       # t interval over seeds
    return m, h


def main():
    rows, feats, vd, td = tg.load()
    print(f"CI over {RUNS} seeds  data={os.path.basename(tg.DATA)} n={len(rows)} "
          f"NO_NUM={tg.NO_NUM}  (mismatch={sum(int(r['label']) for r in rows)})", flush=True)
    R, F = [], []
    for s in range(RUNS):
        r, f = loto_once(rows, feats, vd, td, s)
        R.append(r); F.append(f)
        print(f"  seed{s}: LOTO recall={r:.3f} FPR={f:.3f}", flush=True)
    mr, hr = ci(R); mf, hf = ci(F)
    print(f"\n=== LOTO (de-numeric, {RUNS} seeds): "
          f"recall {mr:.3f} ± {hr:.3f}   FPR {mf:.3f} ± {hf:.3f}   (mean ± t 95% half-width) ===", flush=True)
    print("CI_DONE", flush=True)


if __name__ == "__main__":
    main()
