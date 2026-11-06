"""Packaging: stage the private HF repos under work/hf_stage/ (copies only; nothing is uploaded here).

  work/hf_stage/adapters  -> 5LeaderW/tiny-aya-earth-amharic-maternal-health-adapters (model repo, one folder per stage)
  work/hf_stage/ds_sft    -> 5LeaderW/amharic-maternal-health-sft-curated-v3 (dataset)
  work/hf_stage/ds_dpo    -> 5LeaderW/amharic-maternal-health-dpo-pairs (dataset)
  work/hf_stage/ds_eval   -> 5LeaderW/amharic-maternal-health-aya-eval-logs (dataset)
The merged model (work/hf_stage/model) is filled from the Kaggle merge kernel output.
"""
import os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = f"{ROOT}/work/hf_stage"
R, D = f"{ROOT}/results", f"{ROOT}/data"


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)


def fresh(name):
    p = f"{S}/{name}"
    shutil.rmtree(p, ignore_errors=True); os.makedirs(p)
    return p


# adapters
a = fresh("adapters")
shutil.copy(f"{S}/adapters_README.md", f"{a}/README.md")
for sub, src in [("run2_B_sft", "sft_curated"), ("v3_sft", "sft_v3"), ("v3_dpo", "dpo_v3"), ("v4b_dpo", "dpo_v4b"), ("v4c_dpo", "dpo_v4c")]:
    for f in ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja", "special_tokens_map.json",
              "tokenizer.json", "tokenizer_config.json"]:
        cp(f"{R}/{src}/adapter/{f}", f"{a}/{sub}/{f}")
    cp(f"{R}/{src}/train_info.json", f"{a}/{sub}/train_info.json")
    cp(f"{R}/{src}/train.log", f"{a}/{sub}/train.log")
# run-2 B was trained directly on the official base; record that in its config (others keep their local merged-model path)
import json
p = f"{a}/run2_B_sft/adapter_config.json"; c = json.load(open(p)); c["base_model_name_or_path"] = "CohereLabs/tiny-aya-earth"
json.dump(c, open(p, "w"), indent=2)

# SFT data
s = fresh("ds_sft")
shutil.copy(f"{S}/ds_sft_README.md", f"{s}/README.md")
for f in ["train.jsonl", "dev.jsonl", "rejected.jsonl", "curation_report.md", "curation_stats.json", "chat"]:
    cp(f"{D}/curated_v3/{f}", f"{s}/v3/{f}")
for f in ["quality_scores.jsonl", "referral_labels.jsonl", "rewrites.jsonl", "rewrite_checks.jsonl", "synth_questions.jsonl"]:
    cp(f"{ROOT}/work/run3/{f}", f"{s}/v3/llm_logs/{f}")
for f in ["train.jsonl", "dev.jsonl", "rejected.jsonl", "llm_checks.jsonl", "curation_report.md", "curation_stats.json", "chat"]:
    cp(f"{D}/curated/{f}", f"{s}/run2/{f}")
cp(f"{D}/README.md", f"{s}/README_dataset_build.md")

# DPO pairs
d = fresh("ds_dpo")
shutil.copy(f"{S}/ds_dpo_README.md", f"{d}/README.md")
for v in ["v3", "v4", "v4b", "v4c"]:
    for f in ["train.jsonl", "dev.jsonl"]:
        cp(f"{D}/curated_{v}/dpo/{f}", f"{d}/{v}/{f}")
    for f in ["curation_report.md", "curation_stats.json"]:
        cp(f"{D}/curated_{v}/{f}", f"{d}/{v}/{f}")

# eval results and logs
e = fresh("ds_eval")
shutil.copy(f"{S}/ds_eval_README.md", f"{e}/README.md")
for f in ["summary_table.md", "summary_table.json"]:
    cp(f"{R}/{f}", f"{e}/{f}")
for f in ["comparison.md", "comparison_run2.md", "comparison_run3.md", "comparison_run4.md", "comparison_run4_caveats.md"]:
    cp(f"{R}/{f}", f"{e}/comparisons/{f}")
SYS = [("baseline_official", "baseline_official", True), ("run1_sft", "sft", False), ("run2_A_raw_sft", "sft_raw", False),
       ("run2_B_curated_sft", "sft_curated", False), ("v3_sft", "sft_v3", False), ("v3_dpo", "dpo_v3", False),
       ("v4_dpo", "dpo_v4", False), ("v4b_dpo", "dpo_v4b", False), ("v4c_dpo", "dpo_v4c", False)]
for name, src, flat in SYS:
    ev = f"{R}/{src}" if flat else f"{R}/{src}/eval"
    for f in sorted(os.listdir(ev)):
        if os.path.isfile(f"{ev}/{f}"):
            cp(f"{ev}/{f}", f"{e}/systems/{name}/eval/{f}")
    if not flat:
        for f in ["train_info.json", "train.log"]:
            if os.path.exists(f"{R}/{src}/{f}"):
                cp(f"{R}/{src}/{f}", f"{e}/systems/{name}/{f}")
cp(f"{R}/dpo_v3_attempt1_oom", f"{e}/systems/v3_dpo_attempt1_oom")
J = [("run2_gemini3flash", "judge"), ("run3_test", "judge_run3_test"), ("run3_five_way", "judge_run3"),
     ("run3_v3sft_vs_v3dpo", "judge_run3_dpo"), ("run4_v3dpo_vs_v4", "judge_run4"), ("run4b_v3dpo_vs_v4b", "judge_run4b"),
     ("run4c_v4b_vs_v4c", "judge_run4c"), ("run4c_v4b_vs_v4c_reversed", "judge_run4c_rev")]
for name, src in J:
    cp(f"{R}/{src}", f"{e}/judge/{name}")
cp(f"{R}/referral_audit", f"{e}/referral_audit")
cp(f"{D}/probe_urgent_v4.jsonl", f"{e}/probe_urgent/questions.jsonl")
for f in sorted(os.listdir(f"{R}/probe_v4")):
    cp(f"{R}/probe_v4/{f}", f"{e}/probe_urgent/{f}")
cp(f"{D}/test.jsonl", f"{e}/test/test.jsonl")
cp(f"{ROOT}/work/run3/hf_spend.jsonl", f"{e}/spend/hf_spend.jsonl")
for name in ["adapters", "ds_sft", "ds_dpo", "ds_eval"]:
    n = sum(len(fs) for _, _, fs in os.walk(f"{S}/{name}"))
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(f"{S}/{name}") for f in fs)
    print(f"{name}: {n} files, {sz / 1e6:.1f} MB")
