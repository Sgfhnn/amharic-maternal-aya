#!/usr/bin/env bash
# Pull run-two Kaggle outputs, score them, run the LLM judge and rebuild results/comparison_run2.md.
# Run from the project root inside the Hark sandbox with the Kaggle requests routed through the Google
# credential <your-google-account> (shell credentials=[{provider: google, email: <your-google-account>}]),
# otherwise Kaggle answers 409 ambiguous_account. Needs: pip install sacrebleu sentencepiece auditkit
set -euo pipefail
cd "$(dirname "$0")/../.."
K=scripts/kaggle
for v in raw curated; do
  python3 $K/kag.py status neimashube/amh-tinyaya-r2-$v
done
for v in raw curated; do
  st=$(python3 $K/kag.py status neimashube/amh-tinyaya-r2-$v)
  if [[ "$st" == *complete* ]]; then
    rm -rf results/kaggle_r2_$v && python3 $K/kag.py output neimashube/amh-tinyaya-r2-$v results/kaggle_r2_$v >/dev/null
    rm -rf results/sft_$v && cp -r results/kaggle_r2_$v/results/sft_$v results/sft_$v
    [[ -d results/kaggle_r2_$v/results/baseline_official ]] && rm -rf results/baseline_official && cp -r results/kaggle_r2_$v/results/baseline_official results/baseline_official
    [[ -f results/sft_$v/eval/predictions.jsonl ]] && python3 scripts/score.py results/sft_$v/eval
  else
    echo "r2-$v not complete: $st"
  fi
done
[[ -f results/baseline_official/predictions.jsonl ]] && python3 scripts/score.py results/baseline_official
BASE=results/baseline_official/predictions.jsonl; [[ -f $BASE ]] || BASE=results/baseline/predictions.jsonl
if [[ -f results/sft_raw/eval/predictions.jsonl && -f results/sft_curated/eval/predictions.jsonl ]]; then
  python3 scripts/llm_judge.py --systems baseline=$BASE A=results/sft_raw/eval/predictions.jsonl B=results/sft_curated/eval/predictions.jsonl
fi
python3 scripts/make_comparison_run2.py
