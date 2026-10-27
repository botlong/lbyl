"""Task 3 / B⑧ (throwaway diagnostic) — where does the heuristic gate over-flag SAFE env controls?

§13's π_θ eval saw an always-safe agent score only 0.683 (≈43% of safe pages re-observed) because the
HEURISTIC detector flags safe controls. This tallies the heuristic's false positives on the env's SAFE
(label==0) elements, broken down by template + element properties, to explain the source and say
whether it is a fixable rule artifact or an inherent precision/recall trade-off. CPU, reads only.

Reuses train_gphi.heuristic_pred (the exact rule used as the gate). Run:
  GPHI_DATA_DIR=gphi_data python fpr_diag.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import train_gphi as tg
from collections import defaultdict

_d = os.environ.get("GPHI_DATA_DIR", "gphi_data")
DATA = _d if os.path.isabs(_d) else os.path.join(HERE, _d)
rows = [json.loads(l) for l in open(os.path.join(DATA, "gphi_data.jsonl"))]
safe = [r for r in rows if int(r["label"]) == 0]
fp = [r for r in safe if tg.heuristic_pred(r)]
print(f"env={os.path.basename(DATA)} total={len(rows)} safe(label0)={len(safe)}")
print(f"heuristic FLAGS {len(fp)}/{len(safe)} safe controls = element-level FPR "
      f"{len(fp)/max(1,len(safe)):.3f}", flush=True)

by_t = defaultdict(lambda: [0, 0])
for r in safe:
    by_t[r["template"]][1] += 1
    if tg.heuristic_pred(r):
        by_t[r["template"]][0] += 1
print("\nper-template  FP/safe  (rate):")
for t in sorted(by_t, key=lambda t: -by_t[t][0] / max(1, by_t[t][1])):
    c, n = by_t[t]
    print(f"  {t:30} {c:3}/{n:<3} = {c/max(1,n):.2f}")

print("\nsample flagged-safe elements (why the rule fires):")
for r in fp[:12]:
    print(f"  tmpl={r.get('template'):22} role={r.get('role')} lbyl={r.get('lbyl_role')} "
          f"aria={(r.get('aria') or '')[:32]!r} vtext={(r.get('visible_text') or '')[:22]!r} "
          f"op={r.get('opacity')}")
print("FPR_DIAG_DONE", flush=True)
