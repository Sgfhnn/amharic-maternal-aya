#!/usr/bin/env bash
# Pull the run-three DPO retry kernel (neimashube/amh-tinyaya-r3-dpo), score v3 DPO, judge it head-to-head against
# v3 SFT (same DeepSeek-V4.1-Flash judge, same 100-row sample) and rewrite results/comparison_run3.md.
# Run from the Hark sandbox with shell credentials=[{provider: google, email: <your-google-account>}] (Kaggle routing);
# needs pip install sacrebleu sentencepiece auditkit curatorkit.
set -euo pipefail
cd "$(dirname "$0")/../.."
K=scripts/kaggle
st=$(python3 $K/kag.py status neimashube/amh-tinyaya-r3-dpo); echo "$st"
if [[ "$st" != *complete* ]]; then echo "r3-dpo not complete"; exit 1; fi
rm -rf results/kaggle_r3dpo && python3 $K/kag.py output neimashube/amh-tinyaya-r3-dpo results/kaggle_r3dpo >/dev/null
if [[ -d results/kaggle_r3dpo/results/dpo_v3 ]]; then rm -rf results/dpo_v3 && cp -r results/kaggle_r3dpo/results/dpo_v3 results/dpo_v3; fi
if [[ ! -f results/dpo_v3/eval/predictions.jsonl ]]; then echo "no DPO predictions"; exit 2; fi
python3 scripts/score.py results/dpo_v3/eval > /dev/null
mkdir -p results/judge_run3_dpo && cp -n results/judge_run3/sample_ids.json results/judge_run3_dpo/sample_ids.json
python3 scripts/llm_judge.py --provider hf --model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra --out results/judge_run3_dpo \
  --systems v3sft=results/sft_v3/eval/predictions.jsonl v3dpo=results/dpo_v3/eval/predictions.jsonl | tail -n 5
python3 scripts/make_comparison_run3.py > /dev/null && echo "wrote results/comparison_run3.md"
