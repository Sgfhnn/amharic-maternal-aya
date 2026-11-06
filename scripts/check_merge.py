"""Packaging: check the merged v4c model reproduces v4c predictions.

Compares, on the first 16 test rows (one generation batch):
  merged  = results/merge_check/merged/predictions.jsonl  (single fp16 model, official + B + v3 SFT + v4c merged)
  adapter = results/merge_check/adapter/predictions.jsonl (merged v3 SFT model + v4c adapter, same path as the v4c eval)
  stored  = results/dpo_v4c/eval/predictions.jsonl        (the v4c eval run, all 550 rows, length-sorted batches)
Reports exact-match counts and per-row chrF (sacrebleu) between each pair. Writes results/merge_check/merge_check.json/.md.
Usage: python scripts/check_merge.py [KAGGLE_OUTPUT_DIR]   (default results/kaggle_merge4c)
"""
import json, os, shutil, sys

from sacrebleu.metrics import CHRF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/kaggle_merge4c"
C = f"{ROOT}/results/merge_check"
os.makedirs(C, exist_ok=True)
for sub in ["merged", "adapter"]:
    src = f"{K}/merge_check/{sub}"
    if os.path.isdir(src):
        shutil.rmtree(f"{C}/{sub}", ignore_errors=True); shutil.copytree(src, f"{C}/{sub}")
for f in ["stack.json"]:
    if os.path.exists(f"{K}/merge_check/{f}"):
        shutil.copy(f"{K}/merge_check/{f}", f"{C}/{f}")
jl = lambda p: {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
merged, adapter = jl(f"{C}/merged/predictions.jsonl"), jl(f"{C}/adapter/predictions.jsonl")
stored = jl(f"{ROOT}/results/dpo_v4c/eval/predictions.jsonl")
ids = list(merged)
chrf = CHRF()


def cmp(a, b):
    exact = sum(a[i]["prediction"] == b[i]["prediction"] for i in ids)
    sc = [chrf.sentence_score(a[i]["prediction"], [b[i]["prediction"]]).score for i in ids]
    return {"exact": exact, "n": len(ids), "chrf_mean": round(sum(sc) / len(sc), 2), "chrf_min": round(min(sc), 2)}


res = {"rows": ids, "merged_vs_stored": cmp(merged, stored), "adapter_vs_stored": cmp(adapter, stored),
       "merged_vs_adapter": cmp(merged, adapter)}
# first-divergence position (characters) for rows that differ, merged vs stored
div = {}
for i in ids:
    a, b = merged[i]["prediction"], stored[i]["prediction"]
    if a != b:
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        div[i] = {"first_diff_char": k, "len_merged": len(a), "len_stored": len(b)}
res["merged_vs_stored_divergence"] = div
for name in ["merged", "adapter"]:
    p = f"{C}/{name}/run_meta_gen.json"
    if os.path.exists(p):
        res[f"{name}_run_meta"] = json.load(open(p))
json.dump(res, open(f"{C}/merge_check.json", "w"), indent=1, ensure_ascii=False)
md = ["# Merge check: merged v4c model vs v4c predictions", "",
      f"First {len(ids)} test rows, greedy, max 256 new tokens, fp16, Tesla T4.", "",
      "| Pair | Identical answers | Mean chrF | Min chrF |", "|---|---|---|---|"]
for k in ["merged_vs_stored", "adapter_vs_stored", "merged_vs_adapter"]:
    r = res[k]; md.append(f"| {k.replace('_', ' ')} | {r['exact']}/{r['n']} | {r['chrf_mean']} | {r['chrf_min']} |")
md += ["", "`stored` = the original v4c eval (all 550 rows in length-sorted batches of 16, so the first 16 rows sat in "
       "different batches with different padding); `adapter` = the same model loaded the way that eval loaded it "
       "(merged v3 SFT + v4c adapter merged on GPU) on this 16-row batch; `merged` = the single uploaded fp16 model."]
open(f"{C}/merge_check.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
