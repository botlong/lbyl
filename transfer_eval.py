"""
LBYL W3 — appearance-shift TRANSFER probe (gap #1, synthetic version).

Reviewer worry about LOTO 0.95: all 14 templates share ONE renderer look (Arial,
the same button palette / corner-radius / background), so "cross-template" is still
within-appearance — a head could be reading this renderer's idiosyncrasies, not the
true pixel-vs-structure mismatch. This probe answers that WITHOUT downloading anything:

  train the head on the DEFAULT skin (all templates, all levels)
  test it on a HELD-OUT skin (unseen font / palette / corner-radius / background),
  same mismatch semantics (visible_text vs aria unchanged).

A head that learned the real relation should transfer; one that memorized the look
collapses. We report learned vs heuristic (which is skin-blind, a useful control),
overall + per-level (L2 = the subtle level where the learned head must carry the load),
and with/without the numeric structural scalars (the de-numeric ablation, to confirm
the transfer isn't a role/opacity shortcut).

Run AFTER:
  python make_gphi_dataset.py 20 default            # -> gphi_data/  (skin A; already exists)
  python make_gphi_dataset.py 20 serif_slate        # -> gphi_data_serif_slate/
  GPHI_DATA_DIR=gphi_data            python extract_feats.py   # feats for A (already exists)
  GPHI_DATA_DIR=gphi_data_serif_slate python extract_feats.py # feats for B
  python transfer_eval.py serif_slate
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from learned_gphi import numeric_feats
from train_gphi import train_head, metrics, heuristic_pred
import train_gphi


def load_dir(d):
    blob = torch.load(os.path.join(d, "feats.pt"))
    feats = blob["feats"]
    rows = [json.loads(l) for l in open(os.path.join(d, "gphi_data.jsonl"))]
    rows = [r for r in rows if r["crop"] in feats]
    return rows, feats, blob["vis_dim"], blob["txt_dim"]


def eval_on(m, rows, feats):
    # mirror train_gphi.tensors but for an explicit row/feat set (honors NO_NUM)
    vis = torch.stack([feats[r["crop"]]["vis"] for r in rows])
    txt = torch.stack([feats[r["crop"]]["txt"] for r in rows])
    num = torch.tensor([numeric_feats(r) for r in rows], dtype=torch.float32)
    if train_gphi.NO_NUM:
        num = torch.zeros_like(num)
    if train_gphi.ZERO_VIS:
        vis = torch.zeros_like(vis)
    if train_gphi.ZERO_TXT:
        txt = torch.zeros_like(txt)
    y = [int(r["label"]) for r in rows]
    m.eval()
    with torch.no_grad():
        p = (torch.sigmoid(m(vis, txt, num)) >= 0.5).tolist()
    learned = metrics(y, p)
    heur = metrics(y, [heuristic_pred(r) for r in rows])
    return learned, heur


def show(tag, L, H):
    print(f"\n[{tag}]  (test n={L['n']}, mismatch={L['pos']})")
    print(f"  {'':9} {'recall':>7} {'FPR':>7} {'prec':>7} {'acc':>7}")
    print(f"  heuristic {H['recall']:>7.2f} {H['fpr']:>7.2f} {H['prec']:>7.2f} {H['acc']:>7.2f}")
    print(f"  LEARNED   {L['recall']:>7.2f} {L['fpr']:>7.2f} {L['prec']:>7.2f} {L['acc']:>7.2f}")


def main():
    skin = sys.argv[1] if len(sys.argv) > 1 else "serif_slate"
    train_dir = os.path.join(HERE, "gphi_data")
    test_dir = os.path.join(HERE, f"gphi_data_{skin}")
    tr_rows, tr_feats, vd, td = load_dir(train_dir)
    te_rows, te_feats, vd2, td2 = load_dir(test_dir)
    assert (vd, td) == (vd2, td2), f"feat dims differ: {(vd,td)} vs {(vd2,td2)}"
    print(f"TRANSFER train=default ({len(tr_rows)} elems) -> test={skin} "
          f"({len(te_rows)} elems), NO_NUMERIC={train_gphi.NO_NUM}", flush=True)

    # train ONE head on the full default skin (all templates, all levels)
    m = train_head(tr_rows, vd, td, tr_feats, seed=0)

    show(f"TRANSFER default->{skin} (ALL)", *eval_on(m, te_rows, te_feats))
    for lvl in (0, 1, 2):                      # per-level; L2 is the subtle, head-critical level
        sub = [r for r in te_rows if r["level"] == lvl]
        if sub:
            show(f"TRANSFER default->{skin} L{lvl}", *eval_on(m, sub, te_feats))

    # within-skin reference: how the SAME head scores back on default (held-out look removed)
    show(f"reference default->default (ALL, same head)", *eval_on(m, tr_rows, tr_feats))
    print("TRANSFER_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
