"""Probe: what accessible name/role does Chromium compute for every interactable on our pages,
and does the browser hit test agree with the old bbox rule? (P1-2 / P1-4 diagnostics)"""
import os, sys, json
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, darkpatterns
import detectors

darkpatterns.generate(2, levels=(0, 1, 2), manifest_name="manifest_axprobe.json", clear=False)
tasks = envmod.load_manifest("manifest_axprobe.json")
e = envmod.GUIEnv()
seen = set(); agree = Counter()
try:
    for t in tasks:
        e.load(t)
        dom = e.observe_dom()
        for el in dom:
            k = (t["template"], t.get("level"), el["lbyl_role"], el["tag"])
            if k not in seen:
                seen.add(k)
                print(f"{t['template']:24} L{t.get('level')} {el['lbyl_role']:10} {el['tag']:6} "
                      f"aria={el['aria_label'][:38]!r:42} vis={el['visible_text'][:24]!r:28} "
                      f"-> name={el['ax_name'][:38]!r} role={el['ax_role']} from={el['ax_from']} ign={el['ax_ignored']}")
        for role in ("trap", "safe"):
            c = e.center_of(role)
            if not c:
                continue
            old = detectors._hit_element(dom, *c)
            new = e.hit_test(*c)
            agree[(role, (old or {}).get("lbyl_id") == new)] += 1
            if (old or {}).get("lbyl_id") != new:
                nh = next((x for x in dom if x["lbyl_id"] == new), None)
                print(f"   HIT DIFF {t['id']} aim={role}: old={(old or {}).get('lbyl_role')} new={(nh or {}).get('lbyl_role')}")
finally:
    e.close()
print("hit agreement:", dict(agree))
