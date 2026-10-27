"""
LBYL W3 — train & evaluate the learned consistency head g_phi vs the keyword heuristic.

The W3 claim to prove: a LEARNED screenshot-vs-DOM head catches perceptual dark patterns
that the keyword heuristic MISSES (esp. subtle ones), and GENERALIZES to held-out
difficulty / unseen templates with a low false-positive rate.

Protocols (each trains a fresh head, then compares learned vs heuristic on the held-out
split — generalization, not memorization):
  level->L2 : train on blatant+moderate (L0,L1), test on SUBTLE (L2)
  LOTO      : leave-one-template-out, averaged over the 14 templates

Loads gphi_data/feats.pt (frozen VLM features) + gphi_data.jsonl. Tiny head, CPU-fine.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import torch.nn as nn
from learned_gphi import GphiHead, numeric_feats
from consistency import gphi as heuristic_gphi

DATA = os.path.join(HERE, os.environ.get("GPHI_DATA_DIR", "gphi_data"))
# ablation: zero out the structural SCALARS (opacity/bbox/role one-hot) to test whether the
# learned head's gain is a numeric shortcut (e.g. role==div) vs real pixel-vs-struct alignment.
NO_NUM = os.environ.get("GPHI_NO_NUMERIC", "0") == "1"
# channel ablations: which modality actually carries the detection? Zeroing the VIS (pixel)
# channel leaves aria-text only -> if recall stays high, the head is solving via an aria/
# semantic shortcut, NOT pixel-vs-structure. Zeroing TXT leaves pixels only.
ZERO_VIS = os.environ.get("GPHI_ZERO_VIS", "0") == "1"
ZERO_TXT = os.environ.get("GPHI_ZERO_TXT", "0") == "1"


def load():
    blob = torch.load(os.path.join(DATA, os.environ.get("FEATS_FILE", "feats.pt")))
    feats = blob["feats"]
    rows = [json.loads(l) for l in open(os.path.join(DATA, "gphi_data.jsonl"))]
    rows = [r for r in rows if r["crop"] in feats]          # keep extracted ones
    return rows, feats, blob["vis_dim"], blob["txt_dim"]


def tensors(rows, feats):
    vis = torch.stack([feats[r["crop"]]["vis"] for r in rows])
    txt = torch.stack([feats[r["crop"]]["txt"] for r in rows])
    num = torch.tensor([numeric_feats(r) for r in rows], dtype=torch.float32)
    if NO_NUM:
        num = torch.zeros_like(num)           # ablate structural scalars -> vision+text only
    if ZERO_VIS:
        vis = torch.zeros_like(vis)           # pixel channel off -> aria-text-only probe
    if ZERO_TXT:
        txt = torch.zeros_like(txt)           # structural-text channel off -> pixel-only probe
    y = torch.tensor([float(r["label"]) for r in rows])
    return vis, txt, num, y


def heuristic_pred(r):
    elem = {"visible_text": r.get("visible_text", ""), "aria_label": r.get("aria", ""),
            "ax_name": r.get("ax_name", r.get("aria", "")),
            "opacity": r.get("opacity", "1"), "bg": ""}
    s, _ = heuristic_gphi(elem)
    return s >= 1


def metrics(y, pred):
    y = [int(v) for v in y]; pred = [bool(v) for v in pred]
    tp = sum(a == 1 and b for a, b in zip(y, pred)); fp = sum(a == 0 and b for a, b in zip(y, pred))
    fn = sum(a == 1 and not b for a, b in zip(y, pred)); tn = sum(a == 0 and not b for a, b in zip(y, pred))
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(recall=rec, fpr=fpr, prec=prec, acc=(tp + tn) / max(1, len(y)), pos=tp + fn, n=len(y))


def train_head(tr, vis_dim, txt_dim, feats, epochs=200, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    vis, txt, num, y = tensors(tr, feats)
    m = GphiHead(vis_dim, txt_dim)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    # class-balanced BCE
    pos = max(1.0, y.sum().item()); neg = max(1.0, (len(y) - pos))
    posw = torch.tensor(neg / pos)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(vis, txt, num), y)
        loss.backward(); opt.step()
    return m


def evaluate(m, te, feats):
    m.eval()                                   # turn OFF dropout -> deterministic test metrics
    vis, txt, num, y = tensors(te, feats)
    with torch.no_grad():
        p = (torch.sigmoid(m(vis, txt, num)) >= 0.5).tolist()
    learned = metrics(y.tolist(), p)
    heur = metrics(y.tolist(), [heuristic_pred(r) for r in te])
    return learned, heur


def show(tag, learned, heur):
    print(f"\n[{tag}]  (test n={learned['n']}, mismatch={learned['pos']})")
    print(f"  {'':9} {'recall':>7} {'FPR':>7} {'prec':>7} {'acc':>7}")
    print(f"  heuristic {heur['recall']:>7.2f} {heur['fpr']:>7.2f} {heur['prec']:>7.2f} {heur['acc']:>7.2f}")
    print(f"  LEARNED   {learned['recall']:>7.2f} {learned['fpr']:>7.2f} {learned['prec']:>7.2f} {learned['acc']:>7.2f}")


def main():
    rows, feats, vd, td = load()
    print(f"loaded {len(rows)} elements, vis_dim={vd} txt_dim={td}, "
          f"mismatch={sum(r['label'] for r in rows)}, NO_NUMERIC={NO_NUM}", flush=True)

    # protocol 1: train L0+L1 -> test L2 (subtle)
    tr = [r for r in rows if r["level"] in (0, 1)]
    te = [r for r in rows if r["level"] == 2]
    m = train_head(tr, vd, td, feats)
    show("level->L2 (generalize to SUBTLE)", *evaluate(m, te, feats))
    from stats_util import input_key
    seen = {input_key(feats[r["crop"]]) for r in tr}
    te_new = [r for r in te if input_key(feats[r["crop"]]) not in seen]
    print(f"  ({len(te) - len(te_new)}/{len(te)} L2 test inputs also occur in L0/L1 training; "
          f"de-duplicated test below)")
    show("level->L2, test inputs unseen in training", *evaluate(m, te_new, feats))

    # protocol 2: leave-one-template-out, averaged
    templates = sorted(set(r["template"] for r in rows))
    agg_l = {"recall": 0, "fpr": 0}; agg_h = {"recall": 0, "fpr": 0}
    for i, T in enumerate(templates):          # deterministic seed (was hash(T), not reproducible)
        tr = [r for r in rows if r["template"] != T]
        te = [r for r in rows if r["template"] == T]
        m = train_head(tr, vd, td, feats, seed=i)
        L, H = evaluate(m, te, feats)
        show(f"LOTO test={T}", L, H)
        for k in agg_l:
            agg_l[k] += L[k] / len(templates); agg_h[k] += H[k] / len(templates)
    print(f"\n=== LOTO average ===  heuristic recall={agg_h['recall']:.2f} FPR={agg_h['fpr']:.2f}  |  "
          f"LEARNED recall={agg_l['recall']:.2f} FPR={agg_l['fpr']:.2f}")
    print("TRAIN_GPHI_DONE", flush=True)


if __name__ == "__main__":
    main()
