"""Packaging: stage the GitHub repo (code, kernel builders, configs, small results tables, docs, Space app) in
work/github_repo/amharic-maternal-aya. No weights, no datasets, no secrets. Run scripts/secret_scan.py on it before pushing.
"""
import glob, json, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = f"{ROOT}/work/github_repo/amharic-maternal-aya"
R = f"{ROOT}/results"
keep_git = os.path.isdir(f"{G}/.git")
if os.path.isdir(G):
    for x in os.listdir(G):
        if x != ".git":
            p = f"{G}/{x}"; shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
os.makedirs(G, exist_ok=True)
ign = shutil.ignore_patterns("__pycache__", "*.pyc", "*.safetensors", "*.bin")
shutil.copytree(f"{ROOT}/scripts", f"{G}/scripts", ignore=ign)
shutil.copytree(f"{ROOT}/docs", f"{G}/docs", ignore=ign)
shutil.copytree(f"{ROOT}/space", f"{G}/space", ignore=ign)
os.makedirs(f"{G}/data", exist_ok=True)
for f in ["README.md", "build_dataset.py", "build_log.txt", "stats.json"]:
    shutil.copy(f"{ROOT}/data/{f}", f"{G}/data/{f}")
for v in ["curated", "curated_v3", "curated_v4", "curated_v4b", "curated_v4c"]:
    os.makedirs(f"{G}/data/{v}", exist_ok=True)
    for f in ["curation_report.md", "curation_stats.json"]:
        shutil.copy(f"{ROOT}/data/{v}/{f}", f"{G}/data/{v}/{f}")
# configs: hyper-parameters and data counts of every stage, taken from train_info.json
os.makedirs(f"{G}/configs", exist_ok=True)
for name, src in [("sft_run2_B_curated", "sft_curated"), ("sft_run2_A_raw", "sft_raw"), ("sft_v3", "sft_v3"),
                  ("dpo_v3", "dpo_v3"), ("dpo_v4", "dpo_v4"), ("dpo_v4b", "dpo_v4b"), ("dpo_v4c", "dpo_v4c")]:
    t = json.load(open(f"{R}/{src}/train_info.json"))
    keep = {k: t[k] for k in ["framework", "hp", "final_dpo_args", "data", "model_id", "model_dir", "min_asst_tokens",
                              "final_eval", "nan_events", "peak_mem_gib", "train_seconds", "script"] if k in t}
    json.dump(keep, open(f"{G}/configs/{name}.json", "w"), indent=1)
json.dump({"decoding": "greedy", "max_new_tokens": 256, "chat_template": "stock Tiny Aya template, no extra system prompt",
           "batch_size": 16, "dtype": "float16", "gpu": "Tesla T4", "transformers": "5.15.0",
           "test": "data/test.jsonl, 550 rows (400 AmHQA + 150 Walelign SRH)",
           "mcq": {"belebele": "facebook/belebele amh_Ethi, 900, zero-shot argmax A-D",
                   "afrimmlu": "masakhane/afrimmlu amh test, 500, zero-shot argmax A-D"},
           "judge": {"model": "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra (HF Inference Providers)",
                     "sample": "results/judge_run3/sample_ids.json, 100 rows (60 Walelign + 40 AmHQA)"},
           "stack_v4c": ["CohereLabs/tiny-aya-earth", "run-2 B adapter", "v3 SFT adapter", "v4c DPO adapter"]},
          open(f"{G}/configs/eval_and_stack.json", "w"), indent=1)
# small results tables
os.makedirs(f"{G}/results", exist_ok=True)
for f in ["summary_table.md", "summary_table.json", "comparison.md", "comparison_run2.md", "comparison_run3.md",
          "comparison_run4.md", "comparison_run4_caveats.md"]:
    shutil.copy(f"{R}/{f}", f"{G}/results/{f}")
for f in ["referral_audit.md", "referral_audit.json"]:
    os.makedirs(f"{G}/results/referral_audit", exist_ok=True); shutil.copy(f"{R}/referral_audit/{f}", f"{G}/results/referral_audit/{f}")
for d in ["baseline_official", "sft_curated/eval", "sft_v3/eval", "dpo_v3/eval", "dpo_v4/eval", "dpo_v4b/eval", "dpo_v4c/eval"]:
    for f in ["metrics.json", "mcq_metrics.json"]:
        os.makedirs(f"{G}/results/metrics/{d}", exist_ok=True); shutil.copy(f"{R}/{d}/{f}", f"{G}/results/metrics/{d}/{f}")
for d in glob.glob(f"{R}/judge*"):
    n = os.path.basename(d); os.makedirs(f"{G}/results/judge/{n}", exist_ok=True)
    shutil.copy(f"{d}/judge_summary.json", f"{G}/results/judge/{n}/judge_summary.json")
if os.path.exists(f"{R}/merge_check/merge_check.json"):
    os.makedirs(f"{G}/results/merge_check", exist_ok=True)
    for f in glob.glob(f"{R}/merge_check/*.json") + glob.glob(f"{R}/merge_check/*.md"):
        shutil.copy(f, f"{G}/results/merge_check/")
for f in ["README.md", ".gitignore", "LICENSE", "NOTICE.md"]:
    src = f"{ROOT}/work/github_repo/{f}"
    if os.path.exists(src):
        shutil.copy(src, f"{G}/{f}")
n = sum(len(fs) for dp, _, fs in os.walk(G) if "/.git" not in dp)
sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(G) if "/.git" not in dp for f in fs)
print(f"staged {n} files, {sz / 1e6:.2f} MB at {G}")
