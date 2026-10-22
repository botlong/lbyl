"""
LBYL real-transfer eval — test the synthetically-trained g_phi ZERO-SHOT on SusBench's real
dark-pattern widgets (rendered offline by susbench_probe.py).

The head is trained on the shortcut-proof realistic cross-modal set (gphi_data_realistic_cm,
§5.6 — the SAME training data as the flagship), then frozen and run on SusBench elements.
The head depends ONLY on the training data + channel config, NOT on the test set, so we train
ONE head per channel config and evaluate it on BOTH structural-channel test sets:
  gphi_data_susbench       -> aria channel (what g_phi was TRAINED on; ~empty in the wild)
  gphi_data_susbench_dest  -> navigation-target channel (the real "what it does" signal)

Per dark-pattern class we report:
  flag_rate         = P(g_phi predicts mismatch) over ALL elements of the class.
  recall(decep)     = P(g_phi fires | element is SusBench-labeled deceptive, label==1).
Reading: action_mismatch (disguised) WANT recall high (esp. dest channel); visual/linguistic
WANT flag_rate LOW (specificity -> complementary to a pure-visual detector like DUDE).
"""
import os, sys, json, time
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
torch.set_num_threads(min(64, os.cpu_count() or 8))
import train_gphi
from train_gphi import train_head, tensors, heuristic_pred

TRAIN_DIR = os.path.join(HERE, os.environ.get("TRAIN_DIR", "gphi_data_realistic_cm"))
SB_DIRS = [s for s in os.environ.get("SB_DIRS", "gphi_data_susbench,gphi_data_susbench_dest").split(",") if s]
EPOCHS = int(os.environ.get("GPHI_EPOCHS", "120"))
CONFIGS = [("FULL", False, False), ("ARIA-ONLY(zero vis)", True, False), ("PIXEL-ONLY(zero txt)", False, True)]
RESULTS = os.path.join(HERE, "results", "susbench_transfer.txt")


def load(d):
    blob = torch.load(os.path.join(d, "feats.pt"))
    feats = blob["feats"]
    rows = [json.loads(l) for l in open(os.path.join(d, "gphi_data.jsonl"))]
    rows = [r for r in rows if r["crop"] in feats]
    return rows, feats, blob["vis_dim"], blob["txt_dim"], blob.get("model"), blob.get("txt_mode")


def main():
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    out = open(RESULTS, "w")
    def emit(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()

    tr_rows, tr_feats, vd, td, tr_model, tr_mode = load(TRAIN_DIR)
    tests = []
    for d in SB_DIRS:
        rows, feats, vd2, td2, mdl, mode = load(os.path.join(HERE, d))
        assert (vd, td) == (vd2, td2), f"dim mismatch {d}: {(vd, td)} vs {(vd2, td2)}"
        # provenance guard (codex Q3): a model/txt_mode mismatch would make any OOD effect a
        # TRIVIAL extraction artifact, not a real finding. Refuse to compare apples to oranges.
        assert (mdl, mode) == (tr_model, tr_mode), (
            f"feature provenance mismatch {d}: (model,txt_mode)={(mdl, mode)} != train {(tr_model, tr_mode)} "
            f"-> re-extract feats with identical GPHI_MODEL/TXT_MODE before trusting transfer numbers")
        tests.append((d, rows, feats))
    emit(f"feature provenance OK: model={tr_model} txt_mode={tr_mode}")
    emit(f"train={len(tr_rows)} (realistic_cm)  threads={torch.get_num_threads()} epochs={EPOCHS}")
    emit(f"test sets: " + ", ".join(f"{d}(n={len(r)})" for d, r, _ in tests))
    emit("test by class: " + str(dict(Counter(r["dp_class"] for r in tests[0][1]))))

    train_gphi.NO_NUM = True   # flagship is de-numeric (no bbox/role shortcut)
    for cname, zv, zt in CONFIGS:
        train_gphi.ZERO_VIS, train_gphi.ZERO_TXT = zv, zt
        t0 = time.time()
        m = train_head(tr_rows, vd, td, tr_feats, epochs=EPOCHS, seed=0)
        m.eval()
        emit(f"\n############## channel = {cname}   (trained in {time.time()-t0:.0f}s) ##############")
        for d, te_rows, te_feats in tests:
            vis, txt, num, _ = tensors(te_rows, te_feats)
            with torch.no_grad():
                pred = (torch.sigmoid(m(vis, txt, num)) >= 0.5).tolist()
            classes = sorted(set(r["dp_class"] for r in te_rows))
            emit(f"  [{d}]   {'class':16} {'n':>4} {'flag_rate':>10} {'recall(decep)':>14}")
            for c in classes:
                idx = [i for i, r in enumerate(te_rows) if r["dp_class"] == c]
                pos = [i for i in idx if te_rows[i]["label"] == 1]
                flag = sum(pred[i] for i in idx) / len(idx)
                rec = (sum(pred[i] for i in pos) / len(pos)) if pos else float("nan")
                emit(f"  {'':9}   {c:16} {len(idx):>4} {flag:>10.2f} {rec:>14.2f}")
            if cname == "FULL":
                hp = [heuristic_pred(r) for r in te_rows]
                hl = "  ".join(f"{c}={sum(hp[i] for i, r in enumerate(te_rows) if r['dp_class']==c)/max(1,sum(1 for r in te_rows if r['dp_class']==c)):.2f}"
                              for c in classes)
                emit(f"  {'':9}   [heuristic flag_rate] {hl}")
    train_gphi.ZERO_VIS = train_gphi.ZERO_TXT = train_gphi.NO_NUM = False
    emit("\nSUSBENCH_EVAL_DONE")
    out.close()


if __name__ == "__main__":
    main()
