"""
LBYL W3-X — evaluate g_phi on the channel-balanced (cross-modal) sets.

By construction aria alone and pixels alone are each exactly 50/50 on the label, so a full head that
scores high while ARIA-ONLY and PIXEL-ONLY sit at chance is genuinely comparing the two channels.

Every split asserts that no test input (the exact pair of vis and txt features) occurs in training;
the v1 realistic set failed this (404/404 test inputs were training inputs), which is why it is
superseded by realistic_v2. SPLIT selects the protocol:

  v2     (realistic_v2)  new-instance : train wording families 0-2 on skins 0-3,
                                        test wording family 3 on skin 4 (both unseen)
                         new-wording  : train families 0-2, test family 3 (all skins)
                         new-skin     : train skins 0-3, test skin 4 (all families)
                         held-out types (CM_HOLDOUT, three types): train on the other 11 types
  pair   (grid)          group split by (visible text, accessible name) pair: 70/30 of the
                         distinct pairs per label; the test pairs never occur in training
  random (legacy v1)     the old stratified 70/30 by generation order; reports the overlap
  sweep                  all C(14,3) = 364 three-type hold-outs on the given set (seed 0)

Each configuration is trained over SEEDS head seeds (default 8) and reported as mean +- t-interval.
Run:  CM_DIR=gphi_data_realistic_v2 SPLIT=v2 python crossmodal_eval.py
"""
import os, sys, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import train_gphi
from train_gphi import train_head, metrics
from stats_util import fmt, mean_ci, overlap

DATA = os.path.join(HERE, os.environ.get("CM_DIR", "gphi_data_realistic_v2"))
FEATS = os.environ.get("FEATS_FILE", "feats.pt")   # "feats.pt"(embed) | "feats_hidden.pt"(LLM hidden)
SPLIT = os.environ.get("SPLIT", "v2")
SEEDS = list(range(int(os.environ.get("SEEDS", "8"))))
_HO_DEFAULT = "disguised_decline,fake_close_ad,inverted_consent_toggle"
HOLDOUT_CONCEPTS = set(os.environ.get("CM_HOLDOUT", _HO_DEFAULT).split(","))
ALLOW_OVERLAP = SPLIT == "random"


def _held(c, held=None):
    held = HOLDOUT_CONCEPTS if held is None else held
    return c in held or c.split(":")[0] in held


def load():
    blob = torch.load(os.path.join(DATA, FEATS))
    print(f"data={DATA} feats={FEATS} txt_mode={blob.get('txt_mode','?')} struct={blob.get('struct','?')}",
          flush=True)
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
    ("FULL (vis+txt+num)", False, False, False),
    ("FULL, no numeric  ", False, False, True),
    ("ARIA-ONLY         ", True,  False, True),
    ("PIXEL-ONLY        ", False, True,  True),
    ("NUM-ONLY          ", True,  True,  False),
]


def run_split(name, tr, te, vd, td, feats, seeds=SEEDS):
    ov = overlap(tr, te, feats)
    print(f"\n========== split={name}  (train={len(tr)}, test={len(te)}, "
          f"test mismatch={sum(r['label'] for r in te)}, test inputs also in train={ov}) ==========",
          flush=True)
    if ov and not ALLOW_OVERLAP:
        raise AssertionError(f"{ov} test inputs occur in training under split {name}")
    print(f"  {'channel':19} {'acc':>14} {'recall':>14} {'FPR':>14}   (chance acc 0.50; {len(seeds)} seeds)")
    out = {}
    for cname, zv, zt, nn in CONFIGS:
        train_gphi.ZERO_VIS, train_gphi.ZERO_TXT, train_gphi.NO_NUM = zv, zt, nn
        Ms = [evaluate_rows(train_head(tr, vd, td, feats, seed=s), te, feats) for s in seeds]
        out[cname.strip()] = Ms
        print(f"  {cname} {fmt([M['acc'] for M in Ms]):>14} {fmt([M['recall'] for M in Ms]):>14} "
              f"{fmt([M['fpr'] for M in Ms]):>14}   per-seed acc {[round(M['acc'], 3) for M in Ms]}",
              flush=True)
    train_gphi.ZERO_VIS = train_gphi.ZERO_TXT = train_gphi.NO_NUM = False
    return out


def split_v2(rows):
    nf = max(r["family"] for r in rows); nk = max(r["skin_idx"] for r in rows)
    yield ("new-instance (unseen wording family AND unseen skin)",
           [r for r in rows if r["family"] != nf and r["skin_idx"] != nk],
           [r for r in rows if r["family"] == nf and r["skin_idx"] == nk])
    yield ("new-wording (unseen wording family, all skins)",
           [r for r in rows if r["family"] != nf], [r for r in rows if r["family"] == nf])
    yield ("new-skin (unseen skin, all wording families)",
           [r for r in rows if r["skin_idx"] != nk], [r for r in rows if r["skin_idx"] == nk])


def split_pair(rows, frac=0.7):
    tr, te = [], []
    for lab in (0, 1):
        pairs = sorted({(r["visible_text"], r["aria"]) for r in rows if r["label"] == lab})
        k = int(len(pairs) * frac)
        trp = set(pairs[:k])
        for r in rows:
            if r["label"] == lab:
                (tr if (r["visible_text"], r["aria"]) in trp else te).append(r)
    return tr, te


def split_random(rows, frac=0.7):
    pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]
    kp, kn = int(len(pos) * frac), int(len(neg) * frac)
    return pos[:kp] + neg[:kn], pos[kp:] + neg[kn:]


def holdout(rows, held=None):
    te = [r for r in rows if _held(r["cv"], held) or _held(r["ca"], held)]
    tr = [r for r in rows if not _held(r["cv"], held) and not _held(r["ca"], held)]
    return tr, te


def sweep(rows, vd, td, feats):
    types = sorted({r["cv"].split(":")[0] for r in rows})
    trios = list(itertools.combinations(types, 3))
    res = {"with numeric": [], "no numeric": []}
    for i, trio in enumerate(trios):
        tr, te = holdout(rows, set(trio))
        for key, nn in (("with numeric", False), ("no numeric", True)):
            train_gphi.NO_NUM = nn
            res[key].append(evaluate_rows(train_head(tr, vd, td, feats, seed=0), te, feats)["acc"])
        train_gphi.NO_NUM = False
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(trios)} trios", flush=True)
    for key, xs in res.items():
        m, _, sd = mean_ci(xs)
        print(f"  {len(xs)} three-type hold-outs, {key}: mean acc {m:.3f}, sd {sd:.3f}, "
              f"range {min(xs):.2f}-{max(xs):.2f}", flush=True)


def main():
    rows, feats, vd, td = load()
    pos = sum(r["label"] for r in rows)
    print(f"loaded {len(rows)} elements (mismatch={pos} / match={len(rows)-pos}), "
          f"distinct inputs={len({json.dumps([r['visible_text'], r['aria'], r.get('skin'), r['bbox'][2:]]) for r in rows})}, "
          f"vis_dim={vd} txt_dim={td}", flush=True)
    if SPLIT == "v2":
        for name, tr, te in split_v2(rows):
            run_split(name, tr, te, vd, td, feats)
        run_split(f"held-out types {sorted(HOLDOUT_CONCEPTS)}", *holdout(rows), vd, td, feats)
    elif SPLIT == "pair":
        run_split("pair-disjoint 70/30 (unseen visible/name pairs)", *split_pair(rows), vd, td, feats)
        run_split(f"held-out concepts {sorted(HOLDOUT_CONCEPTS)}", *holdout(rows), vd, td, feats)
    elif SPLIT == "random":
        run_split("legacy RANDOM-70/30 by generation order", *split_random(rows), vd, td, feats)
    elif SPLIT == "sweep":
        sweep(rows, vd, td, feats)
    else:
        sys.exit(f"unknown SPLIT={SPLIT}")
    print("\nCROSSMODAL_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
