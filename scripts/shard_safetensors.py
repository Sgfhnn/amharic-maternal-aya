"""Split a single model.safetensors into ~SIZE_GB shards + model.safetensors.index.json (HF layout), tensor bytes
unchanged. Used so the 6.7 GB merged model uploads in pieces. Usage: python shard_safetensors.py DIR [SIZE_GB]"""
import json, os, sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

d = sys.argv[1]; lim = float(sys.argv[2] if len(sys.argv) > 2 else 1.4) * 1e9
src = f"{d}/model.safetensors"
with safe_open(src, "np") as f:
    keys = list(f.keys()); meta = f.metadata() or {}
    sizes = {}
    for k in keys:
        sl = f.get_slice(k); n = int(np.prod(sl.get_shape())) * 2  # all F16
        sizes[k] = n
    groups, cur, cs = [], [], 0
    for k in keys:
        if cur and cs + sizes[k] > lim:
            groups.append(cur); cur, cs = [], 0
        cur.append(k); cs += sizes[k]
    groups.append(cur)
    N = len(groups); wm = {}
    for i, g in enumerate(groups, 1):
        name = f"model-{i:05d}-of-{N:05d}.safetensors"
        if not os.path.exists(f"{d}/{name}"):  # resumable: shards are written whole or not at all (tmp + rename)
            save_file({k: f.get_tensor(k) for k in g}, f"{d}/{name}.tmp", metadata={"format": "pt", **meta})
            os.replace(f"{d}/{name}.tmp", f"{d}/{name}")
        for k in g:
            wm[k] = name
        print("wrote", name, len(g), "tensors", flush=True)
json.dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": wm}, open(f"{d}/model.safetensors.index.json", "w"), indent=2)
os.remove(src)
print("done", N, "shards; removed single file")
