"""
Pull the ModernBERT-large encoder out of Convai's Laya checkpoint
(convaiinnovations/laya, Apache-2.0) into a plain Hugging Face folder, so it
can be fine-tuned as a teacher with train_mt.py. Only reads safetensors; the
repo's own Python files are never imported or run.

  python tools/extract_laya_encoder.py <laya_snapshot_dir> <out_dir>
"""

import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def main(src, out):
    os.makedirs(out, exist_ok=True)
    tensors = {}
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("encoder."):
                tensors[k[len("encoder."):]] = f.get_tensor(k)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    shutil.copy(os.path.join(src, "encoder", "config.json"), os.path.join(out, "config.json"))
    for fn in os.listdir(os.path.join(src, "tokenizer")):
        shutil.copy(os.path.join(src, "tokenizer", fn), os.path.join(out, fn))
    print("wrote", out, len(tensors), "tensors")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
