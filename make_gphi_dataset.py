"""
LBYL W3 — build the supervised dataset for the LEARNED consistency head g_phi.

g_phi compares the PIXEL channel (what a control LOOKS like — its rendered crop) with
the STRUCTURAL channel (what it actually does — DOM/a11y: aria_label, role, opacity,
geometry). The learned head must flag a *perceptual discrepancy* (类1 label deception /
类2 hidden-overlay) that the keyword heuristic misses at subtle difficulty (e.g. the
fake_close_ad ✕ whose aria says "install …" without the exact keyword phrase, or a
transparent overlay whose crop shows the button beneath it but whose structural action
is "Subscribe").

Output per interactable element:
  crops/<task>_e<id>.png   the rendered crop (the PIXEL channel)
  gphi_data.jsonl          {crop, aria, role, opacity, bbox, visible_text, topmost_role,
                            template, level, threat_class, lbyl_role, label}
  label = 1  iff  lbyl_role == "trap" AND threat_class in {1,2}   (real pixel-vs-DOM mismatch)
        = 0  for safe / distractor, and for 类3 trap (the consistent-deception BLIND SPOT,
             which g_phi is designed NOT to catch — only the W6 task-context signal does).

Pure Playwright + PIL crop -> GPU-FREE (run on the cardless instance). Feature extraction
with the frozen VLM + head training happen later on GPU.
"""
import os, sys, io, json
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, darkpatterns
from PIL import Image

OUT = os.path.join(HERE, "gphi_data")
CROPS = os.path.join(OUT, "crops")


def label_for(elem, task):
    if elem.get("lbyl_role") == "trap" and task.get("threat_class") in (1, 2):
        return 1
    return 0


def main(n=20, levels=(0, 1, 2), skin="default", out_dir=None, manifest_name="manifest.json"):
    out = out_dir or OUT
    crops = os.path.join(out, "crops")
    os.makedirs(crops, exist_ok=True)
    for f in os.listdir(crops):
        if f.endswith(".png"):
            os.remove(os.path.join(crops, f))
    darkpatterns.generate(n, levels=levels, skin=skin, manifest_name=manifest_name,
                          clear=False)                 # deterministic overwrite; keep other sets' pages
    tasks = envmod.load_manifest(manifest_name)
    e = envmod.GUIEnv(headless=True)
    rows = []
    try:
        for t in tasks:
            e.load(t)
            img = Image.open(io.BytesIO(e.screenshot())).convert("RGB")
            for el in e.observe_dom():
                b = el["bbox"]
                if b["w"] < 4 or b["h"] < 4:          # need a real region to crop
                    continue
                pad = 6
                box = (max(0, b["x"] - pad), max(0, b["y"] - pad),
                       min(img.width, b["x"] + b["w"] + pad), min(img.height, b["y"] + b["h"] + pad))
                cid = f"{t['id']}_e{el['lbyl_id']}.png"
                img.crop(box).save(os.path.join(crops, cid))
                rows.append(dict(
                    crop=cid, template=t["template"], level=t.get("level", 0),
                    threat_class=t.get("threat_class"), lbyl_role=el.get("lbyl_role", ""),
                    visible_text=el.get("visible_text", ""), aria=el.get("aria_label", ""),
                    role=el.get("tag", ""), opacity=el.get("opacity", "1"),
                    ax_name=el.get("ax_name", ""), ax_role=el.get("ax_role", ""),
                    ax_from=el.get("ax_from", ""),
                    bbox=[b["x"], b["y"], b["w"], b["h"]],
                    topmost_role=el.get("topmost_role", ""),
                    label=label_for(el, t)))
    finally:
        e.close()
    with open(os.path.join(out, "gphi_data.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pos = sum(r["label"] for r in rows)
    print(f"\nwrote {len(rows)} elements  ({pos} mismatch / {len(rows)-pos} consistent)  -> {out}")
    print("label balance by (template, level):")
    bal = Counter((r["template"], r["level"]) for r in rows)
    posb = Counter((r["template"], r["level"]) for r in rows if r["label"] == 1)
    for k in sorted(bal):
        print(f"  {k[0]:22} L{k[1]}  total={bal[k]:3}  mismatch={posb.get(k,0)}")
    print("MAKE_GPHI_DATA_DONE")


if __name__ == "__main__":
    # usage: make_gphi_dataset.py [n] [skin]
    #   skin="default" -> gphi_data/ (the training distribution, default look)
    #   skin!="default" -> gphi_data_<skin>/ + manifest_<skin>.json (held-out appearance)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    skin = sys.argv[2] if len(sys.argv) > 2 else "default"
    if skin == "default":
        main(n)
    else:
        main(n, skin=skin, out_dir=os.path.join(HERE, f"gphi_data_{skin}"),
             manifest_name=f"manifest_{skin}.json")
