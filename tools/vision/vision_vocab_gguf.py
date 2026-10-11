"""tools/vision/vision_vocab_gguf.py - the "text model" strata-vision loads (vocab only) for GLM-5.3-Flash: the
model GGUF's own tokenizer (every tokenizer.* key of its first shard, copied as they are) under llama.cpp's "glm4"
architecture - the pinned llama.cpp has no glm5-next, and glm4's pre-tokenizer is GLM-5.3's - with the
hyperparameters a vocab-only load still reads.  n_embd 4096 is what strata-vision reports as the width of the image
embeddings: the engine's.

    python tools/vision/vision_vocab_gguf.py --gguf <model>-00001-of-0000N.gguf --out GLM-5.3-Flash-vocab.gguf
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import add_gguf_py  # noqa: E402

add_gguf_py()
import gguf  # noqa: E402

V = gguf.GGUFValueType


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's first shard (its tokenizer.* keys)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-embd", type=int, default=4096)
    a = ap.parse_args()
    r = gguf.GGUFReader(a.gguf)
    arch = "glm4"
    w = gguf.GGUFWriter(a.out, arch)
    w.add_name("GLM-5.3-Flash vocab")
    w.add_context_length(131072)
    w.add_embedding_length(a.n_embd)
    w.add_block_count(1)
    w.add_feed_forward_length(4 * a.n_embd)
    w.add_head_count(32)
    w.add_head_count_kv(2)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(64)
    n_tok = 0
    for k, f in r.fields.items():
        if not k.startswith("tokenizer."):
            continue
        t = f.types[0]
        if t == V.ARRAY:
            sub = f.types[1]
            if sub == V.STRING:
                vals = [bytes(f.parts[i]).decode("utf-8", errors="surrogateescape") for i in f.data]
            else:
                vals = [f.parts[i].tolist()[0] for i in f.data]
            if k == "tokenizer.ggml.tokens":
                n_tok = len(vals)
            w.add_key_value(k, vals, V.ARRAY, sub_type=sub)
        elif t == V.STRING:
            w.add_key_value(k, bytes(f.parts[f.data[0]]).decode("utf-8"), t)
        else:
            w.add_key_value(k, f.parts[f.data[0]].tolist()[0], t)
    if n_tok == 0:
        print(f"{a.gguf}: no tokenizer.ggml.tokens")
        return 1
    w.add_uint32(arch + ".vocab_size", n_tok)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    print(f"{a.out}: {n_tok} tokens from {pathlib.Path(a.gguf).name}, arch {arch}, n_embd {a.n_embd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
