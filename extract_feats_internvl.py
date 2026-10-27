"""
LBYL W3 / R4 — extract FROZEN features from a SECOND VLM family (InternVL) for the g_phi head.

Second-family re-validation: the learned cross-modal head is claimed to be architecture-agnostic
(a controller on ANY frozen VLM), not a Qwen artifact. This mirrors extract_feats.py exactly but
swaps the backbone to InternVL (InternViT vision encoder + a different LLM). Same feats.pt contract
({crop_id: {"vis","txt"}} + vis_dim/txt_dim) -> train_gphi.py / crossmodal_eval.py consume it
unchanged (the head is rebuilt from the declared dims; vis_dim != txt_dim is fine, both are projected
to a common space before the pixel-vs-struct comparison).

Backbone path = transformers-NATIVE InternVL (InternVLForConditionalGeneration, model_type "internvl",
the *-hf checkpoints). We use the native API because on transformers 5.x the OpenGVLab remote-code
checkpoints no longer load: their custom modeling_internvl_chat.py targets 4.x internals and dies on
`InternVLChatModel has no attribute all_tied_weights_keys`. The native path is the supported one and
yields frozen second-family features:
  vis feat = model.get_image_features(pixel_values) — the frozen InternViT vision-encoder representation
             of the crop. On transformers 5.x this returns the vision-tower output (1024-d for the
             InternViT-300M encoder), i.e. one stage BEFORE the pixel-shuffle + multi_modal_projector,
             mean-pooled over patches -> [1024]. That is a DIFFERENT feature level from the Qwen extractor
             (whose model.visual is post-merger, LLM-space), and that is fine: g_phi is dim-agnostic (it
             projects vis_dim and txt_dim SEPARATELY to a shared space before comparing), so vis_dim != Qwen
             is expected. What the second-family claim needs is exactly this — genuinely different-family
             frozen visual features (InternViT, not Qwen-ViT) on which the SAME head still separates
             pixel-vs-aria mismatch. If anything, the pure vision-encoder channel is a HARDER, cleaner test
             than Qwen's already-text-aligned post-merger channel.
  txt feat = input-embeddings over the structural string (computed accessible name + role, ax.py),
             mean-pooled over tokens -> [3584 for the LLM].
Vision preprocessing is a single 448x448 ImageNet-normalised tile — the canonical InternViT input, with
no dynamic multi-tiling (our crops are one small control). This is NOT byte-identical to the Qwen
extractor (which feeds its OWN processor output + image_grid_thw): each backbone necessarily gets its own
native normalisation. The comparison stays fair because each VLM sees its canonical single-crop input and
both reduce to the same downstream shape — a mean-pooled projected-vision vector vs a mean-pooled text
vector; only the backbone differs.

Runs on GPU (an 8B forward is impractically slow on the cardless box). Env vars mirror extract_feats.py:
  GPHI_MODEL, GPHI_DATA_DIR, LIMIT, TXT_MODE(embed|hidden), FEATS_OUT (defaults feats_internvl.pt so it
  does NOT clobber the Qwen feats.pt sitting in the same data dir; downstream is pointed at it via
  FEATS_FILE=feats_internvl.pt / CM_DIR + FEATS_FILE).
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ax import struct_text
MODEL = os.environ.get("GPHI_MODEL", "/root/lbyl-neucom/models/InternVL3-8B-hf")
DATA = os.environ.get("GPHI_DATA_DIR", os.path.join(HERE, "gphi_data"))
LIMIT = int(os.environ.get("LIMIT", "0"))          # 0 = all
TXT_MODE = os.environ.get("TXT_MODE", "embed")     # "embed" (bag-of-token) | "hidden" (LLM last hidden)
INPUT_SIZE = int(os.environ.get("INTERNVL_INPUT", "448"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    # standard InternViT image transform, single tile (our crops are one small control)
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def pool_image_features(vf):
    """model.get_image_features may return a Tensor ([*, tok, hidden] or [tok, hidden]) OR an output
    object; normalise to ONE mean-pooled [hidden] vector over all projected image tokens."""
    if hasattr(vf, "last_hidden_state"):
        vf = vf.last_hidden_state
    if isinstance(vf, (tuple, list)):
        vf = vf[0]
    return vf.reshape(-1, vf.shape[-1]).mean(0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"device={dev} dtype={dtype} model={MODEL.split('/')[-1]} txt_mode={TXT_MODE}", flush=True)
    if dev == "cpu":
        print("[warn] no CUDA — an InternVL 8B forward on CPU is impractically slow; extract on GPU.",
              flush=True)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    assert hasattr(model, "get_image_features"), (
        f"{type(model).__name__} has no get_image_features — not a native InternVL "
        f"(model_type={getattr(model.config, 'model_type', '?')}); point GPHI_MODEL at a *-hf checkpoint")
    tok = AutoTokenizer.from_pretrained(MODEL)
    embed = model.get_input_embeddings()
    # native InternVL nests the LLM at model.model.language_model; some variants expose it top-level.
    lang = getattr(model, "language_model", None) or getattr(getattr(model, "model", None), "language_model", None)
    tf = build_transform(INPUT_SIZE)
    print(f"loaded {type(model).__name__} (model_type={model.config.model_type}) in {time.time()-t0:.0f}s "
          f"| native get_image_features", flush=True)

    rows = [json.loads(l) for l in open(os.path.join(DATA, "gphi_data.jsonl"))]
    if LIMIT:
        rows = rows[:LIMIT]
    assert rows, f"no data rows in {DATA}/gphi_data.jsonl (nothing to extract)"
    feats = {}
    t1 = time.time()
    vis_feat = txt_feat = None
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(DATA, "crops", r["crop"])).convert("RGB")
        pv = tf(img).unsqueeze(0).to(dev, dtype)                       # [1,3,H,W] single tile
        text = struct_text(r)          # computed accessible name + role, as for Qwen (ax.py)
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            vis_feat = pool_image_features(model.get_image_features(pixel_values=pv)).float().cpu()
            if TXT_MODE == "hidden":
                assert lang is not None, "TXT_MODE=hidden but language_model submodule not found"
                hs = lang(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states[-1]
                txt_feat = hs[0].mean(0).float().cpu()
            else:
                txt_feat = embed(ids)[0].mean(0).float().cpu()
        feats[r["crop"]] = {"vis": vis_feat, "txt": txt_feat}
        if i == 0:
            print(f"[probe] vis_feat={tuple(vis_feat.shape)} txt_feat={tuple(txt_feat.shape)}", flush=True)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  {i+1}/{len(rows)}  ({(time.time()-t1)/(i+1):.2f}s/row)", flush=True)
    out = os.path.join(DATA, os.environ.get("FEATS_OUT", "feats_internvl.pt"))
    torch.save(dict(feats=feats, vis_dim=vis_feat.shape[-1], txt_dim=txt_feat.shape[-1],
                    model=MODEL, txt_mode=TXT_MODE), out)
    print(f"saved {len(feats)} feats -> {out}\nEXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
