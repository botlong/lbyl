"""
LBYL W3-X — render the cross-modal dataset (crossmodal.py) into crops + jsonl for g_phi.

Mirrors make_gphi_dataset's render/crop loop but reads the per-button CROSS-MODAL fields
(data-cv / data-ca = visible/aria concept ids, data-lbyl = match/mismatch) so the eval can
do a concept-pair-disjoint split. Pure Playwright + PIL -> GPU-free; feature extraction
(extract_feats.py with GPHI_DATA_DIR=gphi_data_crossmodal) + training happen later.

Output: gphi_data_crossmodal/{crops/*.png, gphi_data.jsonl}
  row = {crop, aria, role, opacity, bbox, visible_text, cv, ca, template, level,
         threat_class, label}   label = 1 iff visible concept != aria concept (mismatch)
"""
import os, sys, io, json
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env as envmod, crossmodal
from ax import ax_by_attr
from PIL import Image

OUT = os.path.join(HERE, "gphi_data_crossmodal")
CROPS = os.path.join(OUT, "crops")

# variant registry: name -> (generator_module, manifest_name, out_dirname)
VARIANTS = {
    "crossmodal": ("crossmodal", "manifest_crossmodal.json", "gphi_data_crossmodal"),
    "realistic":  ("realistic_crossmodal", "manifest_realistic_cm.json", "gphi_data_realistic_cm"),
    # same pages re-rendered into a separate dir: render/feature-pipeline consistency check on a
    # new machine (does a head trained on the archived feats still score the re-render correctly?)
    "realistic_rerender": ("realistic_crossmodal", "manifest_realistic_cm.json", "gphi_data_realistic_rerender"),
    # P1-1 fix: four wording families x five skins x per-page geometry and cell order
    "realistic_v2": ("realistic_v2", "manifest_realistic_v2.json", "gphi_data_realistic_v2"),
}

_CM_JS = r"""
() => {
  const out = [];
  document.querySelectorAll('button.cm').forEach((el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    out.push({
      cmid: el.getAttribute('data-cmid'),
      bbox: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)},
      visible_text: (el.innerText || '').trim().slice(0, 80),
      aria: el.getAttribute('aria-label') || '',
      cv: el.getAttribute('data-cv') || '',
      ca: el.getAttribute('data-ca') || '',
      lbyl_role: el.getAttribute('data-lbyl') || '',
      role: el.tagName.toLowerCase(),
      opacity: cs.opacity,
    });
  });
  return out;
}
"""


def main(seeds=30, variant="crossmodal"):
    import importlib
    gen_modname, manifest_name, out_dirname = VARIANTS[variant]
    gen = importlib.import_module(gen_modname)
    out = os.path.join(HERE, out_dirname)
    crops = os.path.join(out, "crops")
    os.makedirs(crops, exist_ok=True)
    for f in os.listdir(crops):
        if f.endswith(".png"):
            os.remove(os.path.join(crops, f))
    gen.generate(seeds)
    tasks = envmod.load_manifest(manifest_name)
    e = envmod.GUIEnv(headless=True)
    rows = []
    try:
        for t in tasks:
            e.load(t)
            img = Image.open(io.BytesIO(e.screenshot())).convert("RGB")
            ax = ax_by_attr(e.page, "data-cmid")
            for el in e.page.evaluate(_CM_JS):
                b = el["bbox"]
                if b["w"] < 4 or b["h"] < 4:
                    continue
                pad = 6
                box = (max(0, b["x"] - pad), max(0, b["y"] - pad),
                       min(img.width, b["x"] + b["w"] + pad), min(img.height, b["y"] + b["h"] + pad))
                cid = f"{t['id']}_e{el['cmid']}.png"
                img.crop(box).save(os.path.join(crops, cid))
                rows.append(dict(
                    crop=cid, template=t.get("template", "crossmodal"), level=0, threat_class=1,
                    lbyl_role=el["lbyl_role"], visible_text=el["visible_text"], aria=el["aria"],
                    role=el["role"], opacity=el["opacity"],
                    bbox=[b["x"], b["y"], b["w"], b["h"]], topmost_role=el["lbyl_role"],
                    cv=el["cv"], ca=el["ca"], page=t["id"], skin=t.get("skin"),
                    skin_idx=t.get("skin_idx"), family=t.get("family"),
                    label=1 if el["lbyl_role"] == "trap" else 0, **ax[el["cmid"]]))
    finally:
        e.close()
    with open(os.path.join(out, "gphi_data.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pos = sum(r["label"] for r in rows)
    print(f"\nwrote {len(rows)} elements  ({pos} mismatch / {len(rows)-pos} match)  -> {out}")
    # paranoia: confirm the aria string is NOT predictive of the label in the rendered data
    bal = Counter((r["ca"], r["label"]) for r in rows)
    print("aria-concept label balance (each ~50/50 -> aria alone cannot solve it):")
    for c in sorted(set(r["ca"] for r in rows)):
        print(f"  {c:18} label0={bal.get((c,0),0):3} label1={bal.get((c,1),0):3}")
    print("MAKE_CROSSMODAL_DONE")


if __name__ == "__main__":
    # usage: make_crossmodal.py [seeds] [variant=crossmodal|realistic]
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    variant = sys.argv[2] if len(sys.argv) > 2 else "crossmodal"
    main(seeds, variant)
