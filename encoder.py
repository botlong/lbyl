"""
LBYL — the frozen Qwen2.5-VL feature extractor, shared by offline extraction (extract_feats.py)
and the live click-time detector (detectors.LiveLearnedDetector), so both produce the same features.

  vis = vision tower over the control's crop, mean-pooled over patches
  txt = input embeddings (TXT_MODE=embed) or last LLM hidden state (TXT_MODE=hidden) over the
        structural string (ax.struct_text: computed accessible name + computed role), mean-pooled

encode() caches by (crop bytes, text) so a page state that recurs (e.g. the initial page across
policy rollouts) is encoded once, while any change to the pixels or the name is re-encoded.
"""
import io, hashlib
import torch

DEFAULT_MODEL = "/root/lbyl-neucom/models/Qwen2.5-VL-3B-Instruct"


class FrozenEncoder:
    def __init__(self, model_path=DEFAULT_MODEL, txt_mode="embed", device=None):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype, low_cpu_mem_usage=True).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.visual = getattr(self.model, "visual", None) or self.model.model.visual
        self.embed = self.model.get_input_embeddings()
        self.model_path, self.txt_mode = model_path, txt_mode
        self.cache = {}
        self.n_encoded = 0

    @classmethod
    def from_model(cls, model, proc, model_path, txt_mode="embed"):
        """Wrap an already loaded Qwen2.5-VL (e.g. the agent's) instead of loading a second copy."""
        self = cls.__new__(cls)
        self.model, self.proc = model, proc
        self.dev, self.dtype = next(model.parameters()).device, torch.bfloat16
        self.visual = getattr(model, "visual", None) or model.model.visual
        self.embed = model.get_input_embeddings()
        self.model_path, self.txt_mode = model_path, txt_mode
        self.cache, self.n_encoded = {}, 0
        return self

    @torch.no_grad()
    def encode_vis(self, img):
        vp = self.proc.image_processor(images=[img.convert("RGB")], return_tensors="pt")
        vout = self.visual(vp["pixel_values"].to(self.dev, self.dtype),
                           grid_thw=vp["image_grid_thw"].to(self.dev))
        vt = getattr(vout, "pooler_output", None)        # tensor or BaseModelOutputWithPooling
        if vt is None:
            vt = getattr(vout, "last_hidden_state", vout)
        return vt.reshape(-1, vt.shape[-1]).mean(0).float().cpu()

    @torch.no_grad()
    def encode_txt(self, text):
        ids = self.proc.tokenizer(text, return_tensors="pt").input_ids.to(self.dev)
        if self.txt_mode == "hidden":
            hs = self.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1]
            return hs[0].mean(0).float().cpu()
        return self.embed(ids)[0].mean(0).float().cpu()

    def encode(self, img, text):
        buf = io.BytesIO(); img.save(buf, format="PNG")
        key = hashlib.sha1(buf.getvalue() + b"\0" + text.encode()).hexdigest()
        f = self.cache.get(key)
        if f is None:
            f = {"vis": self.encode_vis(img), "txt": self.encode_txt(text)}
            self.cache[key] = f
            self.n_encoded += 1
        return f, key
