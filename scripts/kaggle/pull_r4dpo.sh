#!/usr/bin/env bash
# Run four: pull the DPO v4 kernel, the v4b kernel (neimashube/amh-tinyaya-r4b-dpo, also generates the v4b probe answers) (neimashube/amh-tinyaya-r4-dpo) and the held-out urgent probe kernel
# (neimashube/amh-tinyaya-r4-probe, pushed after r4-dpo completes), score v4, run the referral audit, judge v3 DPO vs v4
# head-to-head (DeepSeek-V4.1-Flash, same 100-row sample) and write results/comparison_run4.md.
# Run from the Hark sandbox with shell credentials=[{provider: google, email: <your-google-account>}] (Kaggle routing);
# needs pip install sacrebleu sentencepiece auditkit curatorkit transformers jinja2.
set -euo pipefail
cd "$(dirname "$0")/../.."
K=scripts/kaggle
st=$(python3 $K/kag.py status neimashube/amh-tinyaya-r4-dpo); echo "r4-dpo $st"
if [[ "$st" != *complete* ]]; then echo "r4-dpo not complete"; exit 1; fi
if [[ ! -f results/dpo_v4/eval/predictions.jsonl ]]; then
  rm -rf results/kaggle_r4dpo && python3 $K/kag.py output neimashube/amh-tinyaya-r4-dpo results/kaggle_r4dpo >/dev/null
  if [[ -d results/kaggle_r4dpo/results/dpo_v4 ]]; then rm -rf results/dpo_v4 && cp -r results/kaggle_r4dpo/results/dpo_v4 results/dpo_v4; fi
fi
if [[ ! -f results/dpo_v4/eval/predictions.jsonl ]]; then echo "no v4 predictions"; exit 2; fi
python3 scripts/score.py results/dpo_v4/eval > /dev/null
ps=$(python3 $K/kag.py status neimashube/amh-tinyaya-r4-probe || true); echo "r4-probe $ps"
if [[ "$ps" == *complete* ]]; then
  rm -rf results/kaggle_r4probe && python3 $K/kag.py output neimashube/amh-tinyaya-r4-probe results/kaggle_r4probe >/dev/null
  mkdir -p results/probe_v4 && cp results/kaggle_r4probe/probe/*.jsonl results/probe_v4/ 2>/dev/null || true
fi
bs=$(python3 $K/kag.py status neimashube/amh-tinyaya-r4b-dpo || true); echo "r4b-dpo $bs"
if [[ "$bs" == *complete* && ! -f results/dpo_v4b/eval/predictions.jsonl ]]; then
  rm -rf results/kaggle_r4bdpo && python3 $K/kag.py output neimashube/amh-tinyaya-r4b-dpo results/kaggle_r4bdpo >/dev/null
  if [[ -d results/kaggle_r4bdpo/results/dpo_v4b ]]; then rm -rf results/dpo_v4b && cp -r results/kaggle_r4bdpo/results/dpo_v4b results/dpo_v4b; fi
  mkdir -p results/probe_v4 && cp results/kaggle_r4bdpo/probe/*.jsonl results/probe_v4/ 2>/dev/null || true
fi
if [[ -f results/dpo_v4b/eval/predictions.jsonl ]]; then python3 scripts/score.py results/dpo_v4b/eval > /dev/null; fi
python3 scripts/referral_audit.py
mkdir -p results/judge_run4 && cp -n results/judge_run3/sample_ids.json results/judge_run4/sample_ids.json
HF_CAP_USD=${HF_CAP_USD:-1.00} python3 scripts/llm_judge.py --provider hf --model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra --out results/judge_run4 \
  --systems v3dpo=results/dpo_v3/eval/predictions.jsonl v4dpo=results/dpo_v4/eval/predictions.jsonl | tail -n 5
if [[ -f results/dpo_v4b/eval/predictions.jsonl ]]; then
  mkdir -p results/judge_run4b && cp -n results/judge_run3/sample_ids.json results/judge_run4b/sample_ids.json
  HF_CAP_USD=${HF_CAP_USD:-1.00} python3 scripts/llm_judge.py --provider hf --model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra --out results/judge_run4b \
    --systems v3dpo=results/dpo_v3/eval/predictions.jsonl v4bdpo=results/dpo_v4b/eval/predictions.jsonl | tail -n 5
fi
python3 scripts/make_comparison_run4.py > /dev/null && echo "wrote results/comparison_run4.md"
