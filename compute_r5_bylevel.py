import json
p = "/root/autodl-tmp/lbyl/m0/results/m1_results.json"
r = json.load(open(p))
print("models in json:", list(r.keys()))
for name, md in r.items():
    for mode, mr in md.items():
        pt = mr.get("per_task", [])
        print(f"\n== {name} / mode={mode} ==  n={len(pt)}  "
              f"agg grounded-ASR={mr.get('grounded_asr'):.3f} raw-ASR={mr.get('asr'):.3f} "
              f"miss={mr.get('none_rate'):.3f} safe={mr.get('safe'):.3f}")
        for l in [0, 1, 2]:
            sub = [x for x in pt if x.get("level") == l]
            n = len(sub) or 1
            trap = sum(x["outcome"] == "trap" for x in sub)
            safe = sum(x["outcome"] == "safe" for x in sub)
            none = sum(x["outcome"] == "none" for x in sub)
            hits = trap + safe
            print(f"   L{l}: n={len(sub):3d}  grounded-ASR={trap/hits if hits else 0:.3f}  "
                  f"raw-ASR={trap/n:.3f}  miss={none/n:.3f}  safe={safe/n:.3f}  "
                  f"(trap={trap} safe={safe} none={none})")
