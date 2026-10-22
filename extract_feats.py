"""
LBYL W3 — extract FROZEN features for the learned g_phi head.

For each element in gphi_data.jsonl:
  vis feat = Qwen2.5-VL vision tower over the rendered crop (the PIXEL channel),
             mean-pooled over patches -> [hidden].
  txt feat = Qwen2.5-VL input-embeddings over the aria/role string (STRUCTURAL channel),
             mean-pooled over tokens -> [hidden].
The VLM is FROZEN and used only as a feature extractor (no grad). Saves feats.pt =
{crop_id: {"vis": tensor, "txt": tensor}} + dims. Set LIMIT=N to validate on N rows first.

Runs on CPU (cardless, free, slow) or GPU. Uses the small 3B by default.
"""
import os, sys, io, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL = os.environ.get("GPHI_MODEL", "/root/rivermind-data/models/Qwen2.5-VL-3B-Instruct")
# GPHI_DATA_DIR lets us point the SAME extractor at a held-out appearance set
# (e.g. gphi_data_serif_slate) without touching the default training set.
DATA = os.environ.get("GPHI_DATA_DIR", os.path.join(HERE, "gphi_data"))
LIMIT = int(os.environ.get("LIMIT", "0"))   # 0 = all
TXT_MODE = os.environ.get("TXT_MODE", "embed")   # "embed" (bag-of-token) | "hidden" (LLM last hidden)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 + low_cpu_mem_usage keeps the load peak ~8-10GB to fit the cardless 16GB cgroup cap
    # (default fp32 load peaks ~24GB -> OOM-kills the container). bf16 vision/embed fwd is fine.
    dtype = torch.bfloat16
    print(f"device={dev} dtype={dtype} model={MODEL.split('/')[-1]}", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=dtype, low_cpu_mem_usage=True).to(dev)
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    visual = getattr(model, "visual", None) or model.model.visual
    embed = model.get_input_embeddings()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(os.path.join(DATA, "gphi_data.jsonl"))]
    if LIMIT:
        rows = rows[:LIMIT]
    feats = {}
    t1 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(DATA, "crops", r["crop"])).convert("RGB")
        vp = proc.image_processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            vout = visual(vp["pixel_values"].to(dev, dtype), grid_thw=vp["image_grid_thw"].to(dev))
            # model.visual may return a tensor OR a BaseModelOutputWithPooling
            vt = getattr(vout, "pooler_output", None)
            if vt is None:
                vt = getattr(vout, "last_hidden_state", vout)
            vis_feat = vt.reshape(-1, vt.shape[-1]).mean(0).float().cpu()
            ids = proc.tokenizer((r.get("aria") or " ") + " " + (r.get("role") or ""),
                                 return_tensors="pt").input_ids.to(dev)
            if TXT_MODE == "hidden":
                # contextualized: run the LLM over the aria tokens, take last hidden state.
                # (mean-pooled INPUT embeddings are bag-of-tokens — too weak for semantic
                #  cross-modal matching to NEW concepts; this should help HOLDOUT.)
                hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1]
                txt_feat = hs[0].mean(0).float().cpu()
            else:
                txt_feat = embed(ids)[0].mean(0).float().cpu()
        feats[r["crop"]] = {"vis": vis_feat, "txt": txt_feat}
        if i == 0:
            print(f"[probe] vis_feat={tuple(vis_feat.shape)} txt_feat={tuple(txt_feat.shape)} "
                  f"grid_thw={vp['image_grid_thw'].tolist()}", flush=True)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  {i+1}/{len(rows)}  ({(time.time()-t1)/(i+1):.2f}s/row)", flush=True)
    out = os.path.join(DATA, os.environ.get("FEATS_OUT", "feats.pt"))
    torch.save(dict(feats=feats, vis_dim=vis_feat.shape[-1], txt_dim=txt_feat.shape[-1],
                    model=MODEL, txt_mode=TXT_MODE), out)
    print(f"saved {len(feats)} feats -> {out}\nEXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
