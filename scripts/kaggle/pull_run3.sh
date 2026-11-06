#!/usr/bin/env bash
# Pull the run-three Kaggle kernel, score v3 SFT and v3 DPO, run the run-three LLM judge on all five systems and write
# results/comparison_run3.md. Run from the Hark sandbox with Kaggle requests routed through the Google credential
# <your-google-account> (shell credentials=[{provider: google, email: <your-google-account>}]); needs pip install sacrebleu sentencepiece auditkit curatorkit
set -euo pipefail
cd "$(dirname "$0")/../.."
K=scripts/kaggle
st=$(python3 $K/kag.py status neimashube/amh-tinyaya-r3); echo "$st"
if [[ "$st" != *complete* ]]; then echo "r3 not complete"; exit 1; fi
rm -rf results/kaggle_r3 && python3 $K/kag.py output neimashube/amh-tinyaya-r3 results/kaggle_r3 >/dev/null
for v in sft_v3 dpo_v3; do
  if [[ -d results/kaggle_r3/results/$v ]]; then rm -rf results/$v && cp -r results/kaggle_r3/results/$v results/$v; fi
  [[ -f results/$v/eval/predictions.jsonl ]] && python3 scripts/score.py results/$v/eval > /dev/null
done
SYSARGS="baseline=results/baseline_official/predictions.jsonl A=results/sft_raw/eval/predictions.jsonl B=results/sft_curated/eval/predictions.jsonl"
[[ -f results/sft_v3/eval/predictions.jsonl ]] && SYSARGS="$SYSARGS v3sft=results/sft_v3/eval/predictions.jsonl"
[[ -f results/dpo_v3/eval/predictions.jsonl ]] && SYSARGS="$SYSARGS v3dpo=results/dpo_v3/eval/predictions.jsonl"
mkdir -p results/judge_run3 && cp -n results/judge/sample_ids.json results/judge_run3/sample_ids.json
python3 scripts/llm_judge.py --provider hf --model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra --out results/judge_run3 --systems $SYSARGS | tail -n 5
python3 scripts/make_comparison_run3.py > /dev/null && echo "wrote results/comparison_run3.md"
