"""
LBYL W3-X — evaluate g_phi on the CROSS-MODAL dataset (the aria-shortcut-proof task).

By construction (crossmodal.py) aria alone and pixels alone are each exactly 50/50 on the
label, so:
  * if the FULL head scores high while ARIA-ONLY and PIXEL-ONLY both sit at chance (~0.5),
    the head is doing a genuine pixel-vs-structure consistency check -> novelty leg #2 is real.
  * if the full head ALSO sits near chance, the frozen mean-pooled features are too weak for
    cross-modal matching (an honest negative -> need better text features / fine-tuning).

Two generalization splits:
  RANDOM   : stratified 70/30 over elements (same concepts, unseen paraphrase/seed/layout).
  HOLDOUT  : leave-out concepts — test = any element whose visible OR aria concept is held out;
             train uses only the other concepts. Tests whether the learned agreement function
             generalizes to concepts never seen during training.

Run after: extract feats with GPHI_DATA_DIR=gphi_data_crossmodal python extract_feats.py
Then:      python crossmodal_eval.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import train_gphi
from train_gphi import train_head, metrics

DATA = os.path.join(HERE, os.environ.get("CM_DIR", "gphi_data_crossmodal"))
FEATS = os.environ.get("FEATS_FILE", "feats.pt")   # "feats.pt"(embed) | "feats_hidden.pt"(LLM hidden)
# concepts/types held out for the HOLDOUT split. Matching is PREFIX-aware: a held-out token
# "subscribe" matches cv=="subscribe"; for the realistic set cv looks like "disguised_decline:A",
# so a held-out token "disguised_decline" matches both its :A and :B controls.
_HO_DEFAULT = "subscribe,download_ad,allow_notif"
HOLDOUT_CONCEPTS = set(os.environ.get("CM_HOLDOUT", _HO_DEFAULT).split(","))


def _held(c):
    return c in HOLDOUT_CONCEPTS or c.split(":")[0] in HOLDOUT_CONCEPTS


def load():
    blob = torch.load(os.path.join(DATA, FEATS))
    print(f"feats={FEATS} txt_mode={blob.get('txt_mode','?')}", flush=True)
    feats = blob["feats"]
    rows = [json.loads(l) for l in open(os.path.join(DATA, "gphi_data.jsonl"))]
    rows = [r for r in rows if r["crop"] in feats]
    return rows, feats, blob["vis_dim"], blob["txt_dim"]


def evaluate_rows(m, rows, feats):
    vis, txt, num, y = train_gphi.tensors(rows, feats)   # honors ZERO_VIS/ZERO_TXT/NO_NUM
    m.eval()
    with torch.no_grad():
        p = (torch.sigmoid(m(vis, txt, num)) >= 0.5).tolist()
    return metrics(y.tolist(), p)


CONFIGS = [   # (name, zero_vis, zero_txt, no_num)
    ("FULL  (vis+txt)", False, False, False),
    ("ARIA-ONLY     ", True,  False, True),
    ("PIXEL-ONLY    ", False, True,  True),
]


def run_split(name, tr, te, vd, td, feats):
    print(f"\n========== split={name}  (train={len(tr)}, test={len(te)}, "
          f"test mismatch={sum(r['label'] for r in te)}) ==========")
    print(f"  {'channel':16} {'recall':>7} {'FPR':>7} {'acc':>7}   (chance acc≈0.50)")
    for cname, zv, zt, nn in CONFIGS:
        train_gphi.ZERO_VIS, train_gphi.ZERO_TXT, train_gphi.NO_NUM = zv, zt, nn
        m = train_head(tr, vd, td, feats, seed=0)
        M = evaluate_rows(m, te, feats)
        print(f"  {cname} {M['recall']:>7.2f} {M['fpr']:>7.2f} {M['acc']:>7.2f}")
    train_gphi.ZERO_VIS = train_gphi.ZERO_TXT = train_gphi.NO_NUM = False


def main():
    rows, feats, vd, td = load()
    pos = sum(r["label"] for r in rows)
    print(f"loaded {len(rows)} crossmodal elements (mismatch={pos} / match={len(rows)-pos}), "
          f"vis_dim={vd} txt_dim={td}", flush=True)

    # RANDOM stratified 70/30 (deterministic: interleave by index within each label)
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]
    def strat(split_rows, frac=0.7):
        k = int(len(split_rows) * frac)
        return split_rows[:k], split_rows[k:]
    trp, tep = strat(pos_rows); trn, ten = strat(neg_rows)
    run_split("RANDOM-70/30 (unseen paraphrase/layout)", trp + trn, tep + ten, vd, td, feats)

    # HOLDOUT concepts: test touches a held-out concept, train avoids them entirely
    te = [r for r in rows if _held(r["cv"]) or _held(r["ca"])]
    tr = [r for r in rows if not _held(r["cv"]) and not _held(r["ca"])]
    run_split(f"HOLDOUT {sorted(HOLDOUT_CONCEPTS)}", tr, te, vd, td, feats)
    print("\nCROSSMODAL_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
