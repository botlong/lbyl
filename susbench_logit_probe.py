"""
Read-only diagnostic (2026-07-08 cold-repro): regenerate §5.8's head-health logit
numbers — training mean logits (honest vs mismatch) and per-class mean logits on
SusBench — for the SAME FULL-channel de-numeric head susbench_eval.py trains
(train_head on gphi_data_realistic_cm, EPOCHS=120, seed=0). Writes nothing into
any dataset dir; output goes to stdout only (redirect into the repro log dir).
Paper targets (§5.8): train honest ≈ −15.6, train mismatch ≈ +12.5, susbench
visual-class ≈ +2.29 (the archived susbench-audit stdout was lost with the box).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
torch.set_num_threads(min(64, os.cpu_count() or 8))
import train_gphi
from train_gphi import train_head, tensors
from susbench_eval import load, TRAIN_DIR, SB_DIRS, EPOCHS


def mean_logit(m, rows, feats, idx):
    vis, txt, num, _ = tensors([rows[i] for i in idx], feats)
    with torch.no_grad():
        return m(vis, txt, num).mean().item()


def main():
    tr_rows, tr_feats, vd, td, tr_model, tr_mode = load(TRAIN_DIR)
    print(f"probe provenance: model={tr_model} txt_mode={tr_mode} "
          f"train={len(tr_rows)} epochs={EPOCHS}", flush=True)
    # identical config to susbench_eval.py's FULL channel: de-numeric, both channels on
    train_gphi.NO_NUM = True
    train_gphi.ZERO_VIS = train_gphi.ZERO_TXT = False
    m = train_head(tr_rows, vd, td, tr_feats, epochs=EPOCHS, seed=0)
    m.eval()
    hon = [i for i, r in enumerate(tr_rows) if int(r["label"]) == 0]
    mis = [i for i, r in enumerate(tr_rows) if int(r["label"]) == 1]
    print(f"train mean logit: honest(label0,n={len(hon)}) = "
          f"{mean_logit(m, tr_rows, tr_feats, hon):+.1f}   "
          f"mismatch(label1,n={len(mis)}) = {mean_logit(m, tr_rows, tr_feats, mis):+.1f}")
    for d in SB_DIRS:
        rows, feats, *_ = load(os.path.join(HERE, d))
        for c in sorted(set(r["dp_class"] for r in rows)):
            idx = [i for i, r in enumerate(rows) if r["dp_class"] == c]
            print(f"[{d}] class={c:16} n={len(idx):3d} "
                  f"mean logit = {mean_logit(m, rows, feats, idx):+.2f}")
    train_gphi.NO_NUM = False
    print("SUSBENCH_LOGIT_PROBE_DONE")


if __name__ == "__main__":
    main()
