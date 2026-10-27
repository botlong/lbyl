"""
LBYL W3 — extract FROZEN features for the learned g_phi head.

For each element in gphi_data.jsonl:
  vis feat = Qwen2.5-VL vision tower over the rendered crop (the PIXEL channel),
             mean-pooled over patches -> [hidden].
  txt feat = Qwen2.5-VL input-embeddings over the structural string (STRUCTURAL channel):
             the browser-computed accessible name + role (ax.struct_text), mean-pooled -> [hidden].
The VLM is FROZEN and used only as a feature extractor (no grad). Saves feats.pt =
{crop_id: {"vis": tensor, "txt": tensor}} + dims. Set LIMIT=N to validate on N rows first.

Runs on CPU (cardless, free, slow) or GPU. Uses the small 3B by default.
"""
import os, sys, io, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from PIL import Image
from encoder import FrozenEncoder, DEFAULT_MODEL
from ax import struct_text

MODEL = os.environ.get("GPHI_MODEL", DEFAULT_MODEL)
# GPHI_DATA_DIR lets us point the SAME extractor at a held-out appearance set
# (e.g. gphi_data_serif_slate) without touching the default training set.
DATA = os.environ.get("GPHI_DATA_DIR", os.path.join(HERE, "gphi_data"))
LIMIT = int(os.environ.get("LIMIT", "0"))   # 0 = all
TXT_MODE = os.environ.get("TXT_MODE", "embed")   # "embed" (bag-of-token) | "hidden" (LLM last hidden)


def main():
    t0 = time.time()
    enc = FrozenEncoder(MODEL, txt_mode=TXT_MODE)
    print(f"device={enc.dev} dtype={enc.dtype} model={MODEL.split('/')[-1]} "
          f"loaded in {time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(os.path.join(DATA, "gphi_data.jsonl"))]
    if LIMIT:
        rows = rows[:LIMIT]
    feats = {}
    t1 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(DATA, "crops", r["crop"])).convert("RGB")
        f, _ = enc.encode(img, struct_text(r))
        feats[r["crop"]] = f
        if i == 0:
            print(f"[probe] vis_feat={tuple(f['vis'].shape)} txt_feat={tuple(f['txt'].shape)} "
                  f"struct={struct_text(r)!r}", flush=True)
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  {i+1}/{len(rows)}  ({(time.time()-t1)/(i+1):.2f}s/row)", flush=True)
    out = os.path.join(DATA, os.environ.get("FEATS_OUT", "feats.pt"))
    vd, td = f["vis"].shape[-1], f["txt"].shape[-1]
    torch.save(dict(feats=feats, vis_dim=vd, txt_dim=td, model=MODEL, txt_mode=TXT_MODE,
                    struct="ax_name+ax_role"), out)
    print(f"saved {len(feats)} feats -> {out}\nEXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
