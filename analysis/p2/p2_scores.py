"""Second-round review (B-2/B-4/B-5): per-row head probabilities for ROC, template bootstrap and the
level split, using train_gphi's trainer unchanged.

  TASK=loto     environment set, leave-one-template-out, head seed 1000*SEED+i (as ci_gphi.py)
  TASK=level    environment set, train L0+L1, test L2, head seed SEED
  TASK=newinst  realistic_v2 new-instance split (unseen wording family and skin), head seed SEED
  CONFIG=full|name|pixel   (numeric scalars always zeroed: the deployed configuration)
  FEATS_FILE=feats.pt|feats_internvl.pt

  TASK=loto CONFIG=full SEED=0 THREADS=2 python analysis/p2/p2_scores.py
Writes results/neucom_p2/scores/<task>_<feats>_<config>_s<seed>.json.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
M0 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, M0)
import torch
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
import train_gphi as tg

TASK, CONFIG = os.environ["TASK"], os.environ["CONFIG"]
SEED = int(os.environ["SEED"])
FEATS = os.environ.get("FEATS_FILE", "feats.pt")
OUT = os.path.join(M0, "results", "neucom_p2", "scores")


def set_config():
    tg.NO_NUM = True
    tg.ZERO_VIS = CONFIG == "name"      # name-only: pixel channel zeroed
    tg.ZERO_TXT = CONFIG == "pixel"     # pixel-only: structural channel zeroed
    assert CONFIG in ("full", "name", "pixel")


def probs(m, te, feats):
    m.eval()
    vis, txt, num, y = tg.tensors(te, feats)
    with torch.no_grad():
        return torch.sigmoid(m(vis, txt, num)).tolist()


def record(te, p, fold):
    return [dict(crop=r["crop"], template=r.get("template", r.get("cv")), level=r.get("level"), label=int(r["label"]),
                 p=float(q), heur=bool(tg.heuristic_pred(r)) if TASK != "newinst" else None, fold=fold)
            for r, q in zip(te, p)]


def main():
    t0 = time.time()
    tg.DATA = os.path.join(M0, "gphi_data_realistic_v2" if TASK == "newinst" else "gphi_data_55")
    os.environ["FEATS_FILE"] = FEATS
    rows, feats, vd, td = tg.load()
    set_config()
    out = []
    if TASK == "loto":
        for i, T in enumerate(sorted(set(r["template"] for r in rows))):
            tr = [r for r in rows if r["template"] != T]
            te = [r for r in rows if r["template"] == T]
            m = tg.train_head(tr, vd, td, feats, seed=1000 * SEED + i)
            out += record(te, probs(m, te, feats), T)
    elif TASK == "level":
        tr = [r for r in rows if r["level"] in (0, 1)]
        te = [r for r in rows if r["level"] == 2]
        m = tg.train_head(tr, vd, td, feats, seed=SEED)
        out = record(te, probs(m, te, feats), "L2")
    elif TASK == "newinst":
        nf = max(r["family"] for r in rows); nk = max(r["skin_idx"] for r in rows)
        tr = [r for r in rows if r["family"] != nf and r["skin_idx"] != nk]
        te = [r for r in rows if r["family"] == nf and r["skin_idx"] == nk]
        m = tg.train_head(tr, vd, td, feats, seed=SEED)
        out = record(te, probs(m, te, feats), "new-instance")
    else:
        raise SystemExit(f"unknown TASK {TASK}")
    os.makedirs(OUT, exist_ok=True)
    tag = os.path.splitext(FEATS)[0]
    path = os.path.join(OUT, f"{TASK}_{tag}_{CONFIG}_s{SEED}.json")
    json.dump(dict(task=TASK, config=CONFIG, seed=SEED, feats=FEATS, rows=out), open(path, "w"))
    print(f"P2_SCORES {TASK} {tag} {CONFIG} s{SEED} n={len(out)} {time.time() - t0:.0f}s -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
