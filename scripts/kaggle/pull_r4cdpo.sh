#!/usr/bin/env bash
# Run four, attempt c: pull neimashube/amh-tinyaya-r4c-dpo (v4c DPO + its held-out urgent probe answers), score v4c,
# re-run the referral audit (incl. false urgency on factual questions), judge v4b vs v4c head-to-head
# (DeepSeek-V4.1-Flash, same 100-row sample) and rewrite results/comparison_run4.md.
# Run from the Hark sandbox with shell credentials=[{provider: google, email: <your-google-account>}] (Kaggle routing);
# needs pip install sacrebleu sentencepiece auditkit curatorkit transformers jinja2.
# HF cap: logged spend was 0.790 before v4c; 0.92 logged ~= +$0.19 billed at the 1.44x ratio (Anu's $0.20 judging limit).
set -euo pipefail
cd "$(dirname "$0")/../.."
K=scripts/kaggle
st=$(python3 $K/kag.py status neimashube/amh-tinyaya-r4c-dpo); echo "r4c-dpo $st"
if [[ "$st" != *complete* ]]; then echo "r4c-dpo not complete"; exit 1; fi
if [[ ! -f results/dpo_v4c/eval/predictions.jsonl ]]; then
  rm -rf results/kaggle_r4cdpo && python3 $K/kag.py output neimashube/amh-tinyaya-r4c-dpo results/kaggle_r4cdpo >/dev/null
  if [[ -d results/kaggle_r4cdpo/results/dpo_v4c ]]; then rm -rf results/dpo_v4c && cp -r results/kaggle_r4cdpo/results/dpo_v4c results/dpo_v4c; fi
  mkdir -p results/probe_v4 && cp results/kaggle_r4cdpo/probe/*.jsonl results/probe_v4/ 2>/dev/null || true
fi
if [[ ! -f results/dpo_v4c/eval/predictions.jsonl ]]; then echo "no v4c predictions"; exit 2; fi
python3 scripts/score.py results/dpo_v4c/eval > /dev/null
python3 scripts/referral_audit.py
mkdir -p results/judge_run4c && cp -n results/judge_run3/sample_ids.json results/judge_run4c/sample_ids.json
HF_CAP_USD=${HF_CAP_USD:-0.92} python3 scripts/llm_judge.py --provider hf --model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra --out results/judge_run4c \
  --systems v4bdpo=results/dpo_v4b/eval/predictions.jsonl v4cdpo=results/dpo_v4c/eval/predictions.jsonl | tail -n 5
mkdir -p results/judge_run4c_rev && cp -n results/judge_run4c/sample_ids.json results/judge_run4c_rev/sample_ids.json
JUDGE_REVERSE=1 HF_CAP_USD=${HF_CAP_USD:-0.92} python3 scripts/llm_judge.py --provider hf --model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra --out results/judge_run4c_rev \
  --systems v4bdpo=results/dpo_v4b/eval/predictions.jsonl v4cdpo=results/dpo_v4c/eval/predictions.jsonl | tail -n 5
python3 scripts/make_comparison_run4.py > /dev/null && echo "wrote results/comparison_run4.md"
