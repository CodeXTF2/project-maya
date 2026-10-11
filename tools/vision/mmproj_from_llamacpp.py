"""tools/vision/mmproj_from_llamacpp.py - a llama.cpp-style GLM-5.3-Flash mmproj (projector type "glm5v", e.g.
Unsloth's mmproj-model-bf16.gguf) as the "glm5next" mmproj strata-vision reads.

The tensors are the same ones glm5next_mmproj.py writes from the checkpoint (same names, same layout); what differs is
the projector type, three keys (swiglu_clamp -> ffn_clamp, image_min/max_pixels -> image_min/max_tokens) and the
storage: BF16 matrices go out as F16 (glm5next_mmproj.py's choice; refused when a weight does not fit F16), norms and
biases as F32.

    python tools/vision/mmproj_from_llamacpp.py --src mmproj-model-bf16.gguf --out mmproj-GLM-5.3-Flash-F16.gguf
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import add_gguf_py  # noqa: E402

add_gguf_py()
import gguf  # noqa: E402

TOKEN_PX = 28 * 28   # one image token: a 2x2 merge of 14x14 patches


def field_value(f):
    """A reader field's value: a str, a number, a bool, or a list of them."""
    t = f.types[0]
    if t == gguf.GGUFValueType.ARRAY:
        return [f.parts[i].tolist()[0] if f.types[1] != gguf.GGUFValueType.STRING else bytes(f.parts[i]).decode()
                for i in f.data]
    if t == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode()
    return f.parts[f.data[0]].tolist()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = gguf.GGUFReader(a.src)
    kv = {k: field_value(f) for k, f in r.fields.items() if not k.startswith("GGUF.")}
    if kv.get("clip.projector_type") not in ("glm5v", "glm5next"):
        print(f"{a.src}: projector type {kv.get('clip.projector_type')!r}, not GLM-5.3-Flash's (glm5v)")
        return 1

    w = gguf.GGUFWriter(a.out, arch="clip")
    w.add_string("general.type", "mmproj")
    w.add_string("general.name", "GLM-5.3-Flash vision")
    w.add_string("clip.projector_type", "glm5next")
    for k in ("clip.has_vision_encoder", "clip.use_silu"):
        w.add_bool(k, bool(kv.get(k, True)))
    for k in ("clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
              "clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
              "clip.vision.attention.head_count", "clip.vision.spatial_merge_size"):
        w.add_uint32(k, int(kv[k]))
    w.add_float32("clip.vision.attention.layer_norm_epsilon", float(kv["clip.vision.attention.layer_norm_epsilon"]))
    w.add_array("clip.vision.image_mean", [float(x) for x in kv["clip.vision.image_mean"]])
    w.add_array("clip.vision.image_std", [float(x) for x in kv["clip.vision.image_std"]])
    w.add_float32("clip.vision.ffn_clamp", float(kv.get("clip.vision.swiglu_clamp", kv.get("clip.vision.ffn_clamp", 10.0))))
    w.add_uint32("clip.vision.image_min_tokens", int(kv.get("clip.vision.image_min_pixels", 16 * TOKEN_PX)) // TOKEN_PX)
    w.add_uint32("clip.vision.image_max_tokens", int(kv.get("clip.vision.image_max_pixels", 8000 * TOKEN_PX)) // TOKEN_PX)

    big = 0.0
    for t in r.tensors:
        shape = tuple(int(x) for x in reversed(t.shape))          # numpy order (ggml's ne reversed)
        raw = np.asarray(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            x = (np.frombuffer(raw.tobytes(), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif t.tensor_type == gguf.GGMLQuantizationType.F16:
            x = np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32).reshape(shape)
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            x = np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(shape)
        else:
            print(f"{t.name}: storage type {t.tensor_type.name} not handled")
            return 1
        big = max(big, float(np.abs(x).max()))
        w.add_tensor(t.name, np.ascontiguousarray(x.astype(np.float16) if x.ndim >= 2 else x))
    if big > 60000:
        print(f"a weight of {big} does not fit F16")
        return 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.out}: {len(r.tensors)} tensors (largest |w| {big:.2f}), projector glm5next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
