"""Second-round review statistics (pure Python): reads results/neucom_p2 and prints
  B-2  AUROC and recall at matched false-positive rates (environment-set LOTO, new-instance split)
  B-4  template-level paired comparisons (LOTO recall/FPR; scripted-agent ASR, mixed training vs heuristic)
  B-5  level split, eight seeds, both backbones, full / name-only / pixel-only
  B-1/B-6  RL rules table
  python3 m0/analysis/p2/p2_analyze.py
"""
import glob, json, os, random, re, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
M0 = os.path.dirname(os.path.dirname(HERE))
P2 = os.path.join(M0, "results", "neucom_p2")
P1 = os.path.join(M0, "results", "neucom_p1")
T975_7 = 2.3646


def mci(xs):
    m = statistics.mean(xs)
    return m, (T975_7 * statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def f(xs, d=2):
    m, h = mci(xs)
    return f"{m:.{d}f} ± {h:.{d}f}"


def auroc(y, p):
    pairs = sorted(zip(p, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(y); nneg = len(y) - npos
    rpos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab)
    return (rpos - npos * (npos + 1) / 2) / (npos * nneg)


def recall_at_fpr(y, p, target):
    """Highest recall whose FPR does not exceed `target` (threshold sweep on the test scores)."""
    neg = sorted((q for q, lab in zip(p, y) if not lab), reverse=True)
    pos = [q for q, lab in zip(p, y) if lab]
    k = int(target * len(neg))                 # number of negatives allowed above threshold
    thr = neg[k] if k < len(neg) else -1.0
    return sum(q > thr for q in pos) / len(pos)


def load(task, feats, cfg):
    out = {}
    for path in glob.glob(os.path.join(P2, "scores", f"{task}_{feats}_{cfg}_s*.json")):
        d = json.load(open(path)); out[d["seed"]] = d["rows"]
    return [out[s] for s in sorted(out)]


def macro(rows, key):
    """LOTO-average recall / FPR as ci_gphi.py computes it: mean over held-out templates."""
    by = {}
    for r in rows:
        by.setdefault(r["fold"], []).append(r)
    rec, fpr = [], []
    for T, rs in by.items():
        pos = [r for r in rs if r["label"]]; neg = [r for r in rs if not r["label"]]
        hit = (lambda r: r["p"] >= 0.5) if key == "p" else (lambda r: r["heur"])
        rec.append(sum(map(hit, pos)) / len(pos)); fpr.append(sum(map(hit, neg)) / len(neg))
    return statistics.mean(rec), statistics.mean(fpr), by


def boot(diffs, n=20000, seed=0):
    rnd = random.Random(seed); k = len(diffs)
    ms = sorted(statistics.mean(rnd.choice(diffs) for _ in range(k)) for _ in range(n))
    return statistics.mean(diffs), ms[int(0.025 * n)], ms[int(0.975 * n)]


def sign_test(diffs):
    from math import comb
    pos = sum(d > 0 for d in diffs); neg = sum(d < 0 for d in diffs); n = pos + neg
    k = min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return pos, neg, p


def section_loto():
    seeds = load("loto", "feats", "full")
    if not seeds:
        return print("LOTO: no scores yet")
    print(f"\n=== B-2/B-4 environment-set LOTO, full head no numeric, {len(seeds)} seeds ===")
    R, F, A, RH = [], [], [], []
    for rows in seeds:
        r, fp, _ = macro(rows, "p"); R.append(r); F.append(fp)
        y = [x["label"] for x in rows]; p = [x["p"] for x in rows]
        A.append(auroc(y, p))
        hy = [x for x in rows if not x["label"]]
        hfpr = sum(x["heur"] for x in hy) / len(hy)
        RH.append(recall_at_fpr(y, p, hfpr))
    hr, hf, _ = macro(seeds[0], "heur")
    rows0 = seeds[0]
    hpool_r = sum(x["heur"] for x in rows0 if x["label"]) / sum(x["label"] for x in rows0)
    hpool_f = sum(x["heur"] for x in rows0 if not x["label"]) / sum(not x["label"] for x in rows0)
    print(f"  learned @0.5, LOTO-avg: recall {f(R)}  FPR {f(F)}   (per-seed recall {[round(x, 3) for x in R]})")
    print(f"  heuristic LOTO-avg: recall {hr:.3f} FPR {hf:.3f};  pooled recall {hpool_r:.3f} FPR {hpool_f:.3f}")
    print(f"  learned pooled AUROC {f(A, 3)}")
    print(f"  learned pooled recall at the heuristic's pooled FPR ({hpool_f:.3f}): {f(RH)}")
    # pooled ROC mixes fourteen fold heads; per-fold versions (post-hoc, threshold chosen per test fold)
    FA, FR, FF = [], [], []
    for rows in seeds:
        _, _, by = macro(rows, "p")
        a, rr, ff = [], [], []
        for T, rs in by.items():
            y = [x["label"] for x in rs]; p = [x["p"] for x in rs]
            neg = [x for x in rs if not x["label"]]
            tf = sum(x["heur"] for x in neg) / len(neg)
            a.append(auroc(y, p)); rr.append(recall_at_fpr(y, p, tf))
            negp = sorted((x["p"] for x in neg), reverse=True); k = int(tf * len(negp))
            thr = negp[k] if k < len(negp) else -1.0
            ff.append(sum(q > thr for q in negp) / len(negp))
        FA.append(statistics.mean(a)); FR.append(statistics.mean(rr)); FF.append(statistics.mean(ff))
    print(f"  per-fold AUROC, macro over templates: {f(FA, 3)}")
    print(f"  per-fold recall at each fold's heuristic FPR (post hoc), macro: {f(FR)} at FPR {f(FF)}")
    print(f"  actual pooled FPR at the matched threshold, per seed: "
          f"{[round(sum(q > (sorted((x['p'] for x in r if not x['label']), reverse=True) + [-1.0])[int(hpool_f * sum(not x['label'] for x in r))] for q in (x['p'] for x in r if not x['label'])) / sum(not x['label'] for x in r), 3) for r in seeds]}")
    # template-level paired comparison: learned (seed-mean) vs heuristic, per held-out template
    dr, df = [], []
    per = {}
    for rows in seeds:
        _, _, by = macro(rows, "p")
        for T, rs in by.items():
            pos = [x for x in rs if x["label"]]; neg = [x for x in rs if not x["label"]]
            per.setdefault(T, []).append((sum(x["p"] >= .5 for x in pos) / len(pos),
                                          sum(x["p"] >= .5 for x in neg) / len(neg),
                                          sum(x["heur"] for x in pos) / len(pos),
                                          sum(x["heur"] for x in neg) / len(neg)))
    print("  per template (learned recall, heuristic recall | learned FPR, heuristic FPR):")
    for T in sorted(per):
        v = per[T]; lr = statistics.mean(a[0] for a in v); lf = statistics.mean(a[1] for a in v)
        print(f"    {T:26} {lr:.2f} {v[0][2]:.2f} | {lf:.2f} {v[0][3]:.2f}")
        dr.append(lr - v[0][2]); df.append(lf - v[0][3])
    m, lo, hi = boot(dr); p, n, sp = sign_test(dr)
    print(f"  recall diff learned-heuristic over 14 templates: {m:+.3f} [bootstrap 95% {lo:+.3f}, {hi:+.3f}]  "
          f"sign test {p}+/{n}- p={sp:.3f}")
    m, lo, hi = boot(df); p, n, sp = sign_test(df)
    print(f"  FPR diff learned-heuristic over 14 templates:    {m:+.3f} [bootstrap 95% {lo:+.3f}, {hi:+.3f}]  "
          f"sign test {p}+/{n}- p={sp:.3f}")


def section_newinst():
    seeds = load("newinst", "feats", "full")
    if not seeds:
        return print("new instance: no scores yet")
    print(f"\n=== B-2 new-instance split, {len(seeds)} seeds ===")
    A, ACC, R54, ACC54 = [], [], [], []
    for rows in seeds:
        y = [x["label"] for x in rows]; p = [x["p"] for x in rows]
        A.append(auroc(y, p))
        ACC.append(sum((q >= .5) == bool(l) for q, l in zip(p, y)) / len(y))
        r = recall_at_fpr(y, p, 0.54); R54.append(r)
        ACC54.append((r * sum(y) + (1 - 0.54) * (len(y) - sum(y))) / len(y))
    print(f"  accuracy @0.5 {f(ACC)}  AUROC {f(A, 3)}")
    print(f"  recall at FPR 0.54 (prompted VLM: recall 0.99): {f(R54)}  -> accuracy {f(ACC54)}")


def section_level():
    print("\n=== B-5 level split L0+L1 -> L2, 8 seeds ===")
    for feats in ("feats", "feats_internvl"):
        for cfg in ("full", "name", "pixel"):
            seeds = load("level", feats, cfg)
            if not seeds:
                continue
            R = [sum(x["p"] >= .5 for x in s if x["label"]) / sum(x["label"] for x in s) for s in seeds]
            F = [sum(x["p"] >= .5 for x in s if not x["label"]) / sum(not x["label"] for x in s) for s in seeds]
            A = [auroc([x["label"] for x in s], [x["p"] for x in s]) for s in seeds]
            print(f"  {feats:15} {cfg:6} recall {f(R)}  FPR {f(F)}  AUROC {f(A, 3)}  "
                  f"per-seed recall {[round(x, 2) for x in R]}")


def section_e2e():
    hp = os.path.join(P2, "e2e_heuristic_by_tpl.json")
    if not os.path.exists(hp):
        return print("\nE2E heuristic per-template: not yet")
    heur = json.load(open(hp))["reground"]["by_tpl"]
    hasr = {T: c.get("trap", 0) / sum(c.values()) for T, c in heur.items()}
    txt = open(os.path.join(P1, "e3.log"), errors="replace").read()
    sec = txt[txt.index("##### E3b"):txt.index("##### E3c")]
    mixed = {}
    for m in re.finditer(r"seed(\d+) (\w+): reground\s+ASR\(trap\)=([\d.]+)", sec):
        mixed.setdefault(m.group(2), []).append(float(m.group(3)))
    print("\n=== B-4 scripted-agent E2E: mixed training (E3b, 8 seeds) vs heuristic, per template ASR ===")
    d = []
    for T in sorted(hasr):
        mm = statistics.mean(mixed[T]); d.append(mm - hasr[T])
        print(f"    {T:26} mixed {mm:.2f}  heuristic {hasr[T]:.2f}")
    m, lo, hi = boot(d); p, n, sp = sign_test(d)
    print(f"  ASR diff mixed-heuristic: {m:+.3f} [bootstrap 95% {lo:+.3f}, {hi:+.3f}]  sign test {p}+/{n}- p={sp:.3f}")
    print(f"  heuristic overall ASR {statistics.mean(hasr.values()):.3f} (Table 6: 0.14)")
    per = {}
    for r in json.load(open(os.path.join(P1, "e2e_8seed.json")))["results"]:
        for T, c in r["reground"]["by_tpl"].items():
            per.setdefault(T, []).append(c.get("trap", 0) / sum(c.values()))
    d = [statistics.mean(per[T]) - hasr[T] for T in sorted(hasr)]
    m, lo, hi = boot(d); p, n, sp = sign_test(d)
    print(f"  single deployed head (8 seeds) minus heuristic: {m:+.3f} [bootstrap 95% {lo:+.3f}, {hi:+.3f}]  "
          f"sign test {p}+/{n}- p={sp:.3f}")


def section_rl():
    print("\n=== B-1/B-6 RL rules (heuristic gate) ===")
    for path in sorted(glob.glob(os.path.join(P2, "rl_*.json"))):
        d = json.load(open(path))
        for k, v in d["results"].items():
            none = 1 - v["asr"] - v["safe"] - v["refused"]
            print(f"  {d['split']:8} n={d['n']:3} {k:16} ASR {v['asr']:.3f} safe {v['safe']:.3f} "
                  f"refused {v['refused']:.3f} neither {none:.3f}")


if __name__ == "__main__":
    section_loto(); section_newinst(); section_level(); section_e2e(); section_rl()
