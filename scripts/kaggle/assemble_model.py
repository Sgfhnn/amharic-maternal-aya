"""Assemble a byte-exact local snapshot of the gated CohereLabs/tiny-aya-earth repo inside a Kaggle kernel
without putting an HF token in the kernel.

- Small files (config.json, generation_config.json, tokenizer*.json, special_tokens_map.json,
  model.safetensors.index.json) were downloaded from CohereLabs/tiny-aya-earth with the authorised token
  (HF user 5LeaderW, access granted 2026-11-05) and shipped in the private Kaggle dataset as official__<name>.
- The two weight shards are fetched from the public mirror 1-800-LLMs/tiny-aya-earth and their SHA-256 is
  checked against the official repo's LFS oids below (taken from the official repo's tree API). Any mismatch aborts.

Usage: python assemble_model.py /tmp/tinyaya_official
"""
import glob, hashlib, json, os, shutil, sys

OFFICIAL_LFS = {
    "model-00001-of-00002.safetensors": "241f69384f874e2a6693d6de3c77224593514288218df0b8c432854bb4ae6336",
    "model-00002-of-00002.safetensors": "bea39b097358d078a62e88b25f367489d771b3b7f2c8206d4bb8f0df4e0c7af0",
    "tokenizer.json": "2227ea9c52e8afb3f98bfed2679008b275f2664de69dfde174b374389eb0225d",
}
SMALL = ["config.json", "generation_config.json", "model.safetensors.index.json", "special_tokens_map.json",
         "tokenizer_config.json", "tokenizer.json"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(dst):
    if os.path.exists(f"{dst}/.verified"):
        print("already assembled", dst); return
    os.makedirs(dst, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for fn in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
        p = hf_hub_download("1-800-LLMs/tiny-aya-earth", fn, local_dir=dst)
        got = sha256(p)
        assert got == OFFICIAL_LFS[fn], f"{fn} sha256 {got} != official {OFFICIAL_LFS[fn]}"
        print("verified", fn, got[:12])
    for fn in SMALL:
        src = (glob.glob(f"/kaggle/input/**/official__{fn}", recursive=True) or glob.glob(f"**/official__{fn}", recursive=True))[0]
        shutil.copy(src, f"{dst}/{fn}")
    assert sha256(f"{dst}/tokenizer.json") == OFFICIAL_LFS["tokenizer.json"]
    json.dump({"repo": "CohereLabs/tiny-aya-earth", "weights_sha256": OFFICIAL_LFS}, open(f"{dst}/.verified", "w"))
    print("assembled CohereLabs/tiny-aya-earth at", dst)


if __name__ == "__main__":
    main(sys.argv[1])
