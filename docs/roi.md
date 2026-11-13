# ROI: base Tiny Aya Earth vs the submitted v4c DPO model

Every measured number comes from `results/` (summary_table, comparison_run2/3/4, referral_audit, run_meta files, predictions.jsonl) and the project's HF spend log. Rows marked **estimate** are derived, and the derivation is given underneath.

## What we got

| Quality (550-question held-out Amharic health test unless noted) | Base tiny-aya-earth | v4c DPO | Change |
|---|---|---|---|
| Answers hitting the 256-token cap | 98.5% | 29.5% | -69.0 pts |
| Markdown in the answer | 96.0% | 0.0% | -96.0 pts |
| Mean answer length, Aya tokens (from predictions.jsonl) | 254.8 | 185.0 | -27% |
| Corpus chrF++ (550) / Walelign SRH chrF++ (150) | 10.3 / 17.8 | 12.3 / 19.0 | +2.0 / +1.2 |
| AuditKIT per-sample chrF | 18.8 | 22.7 | +3.9 |
| Referral on 82 held-out danger-sign questions | not run | 96.3% (92.7% urgent-worded) | n/a |
| Referral line on non-urgent questions | 0.7% | 5.8% | +5.1 pts |
| False "go now" on factual questions | 0.4% | 1.3% | +0.9 pts |
| Belebele Amharic (900) | 61.7 | 63.6 | +1.9 |
| AfriMMLU Amharic (500) | 46.4 | 44.4 | -2.0 |
| Judge correctness, 1-5 (**chained estimate**) | ~2.6 | ~3.0 | about +0.4 |
| Judge fluency, 1-5 (**chained estimate**) | ~3.8 | ~4.4 | about +0.6 |

Judge rows are chained, because base and v4c were never in the same judge pass. The passes are: the run-3 five-way (base 2.56 / 3.80, v3 SFT 2.58 / 3.66), then v3 SFT vs v3 DPO (2.65 / 3.68 -> 3.29 / 4.27), v3 DPO vs v4b (3.18 / 4.34 -> 3.12 / 4.47), and v4b vs v4c (3.13 / 4.50 -> 3.01 / 4.43). Each pass has about +-0.1 noise, and the judge is an LLM (DeepSeek-V4.1-Flash), not a clinician.

## What it cost

| Cost item | Amount | Source |
|---|---|---|
| GPU, submitted chain (run-2 SFT 5,666 s + v3 SFT 3,329 s + v4c DPO 1,388 s) | 2.9 h on Kaggle T4 x2 | train_info.json `train_seconds` |
| GPU, every training run including dead ends (9 runs) | 9.3 h on Kaggle T4 x2 | same |
| GPU, one full evaluation of a model (generation 819 s + MCQ 470 s) | 0.36 h on one T4 | run_meta_gen/mcq.json (v4c) |
| GPU, cash paid | $0 (free Kaggle quota) | Kaggle |
| GPU, market value if rented (**estimate**) | about $3.30 to $6.50 | 9.3 h x $0.35 per T4-hour (GCP on-demand, GPU only), x1 or x2 GPUs |
| HF Inference credit, curation (Gemma 4 31B + DeepSeek V4 Flash: classify, quality gate, rewrites, verify, synthetic questions) | $0.56 list | hf_spend.jsonl |
| HF Inference credit, triage labels for test and probe | $0.02 list | hf_spend.jsonl |
| HF Inference credit, LLM judging (624 calls) | $0.31 list | hf_spend.jsonl |
| HF Inference total | $0.88 list, **$1.03 billed** | hf_spend.jsonl; HF billing usage-v2 |
| Free promotional API credit (run-2 checks and judge, Cline) | about $0.20 | STATUS.md |
| **Total cash** | **about $1.03** | |

## What it costs to run

| Running cost | Value |
|---|---|
| Model size | 3.35B parameters, fp16 weights about 6.7 GB. Fits one 16 GB T4. |
| Measured throughput, v4c, one T4, batches of 16, max 256 new tokens | 550 answers in 819 s = 1.49 s per answer |
| GPU time per 1,000 answers | 0.41 T4-hours |
| Self-hosted cost per 1,000 answers (**estimate**) | about $0.15 (0.41 h x $0.35/h GCP T4, GPU only; VM, disk and idle time extra) |
| Demo Space | $0 (HF ZeroGPU) |
| Hosted large-model alternative per 1,000 answers (**estimate**) | about $0.13 list, $0.19 at our billed ratio, for DeepSeek-V4.1-Flash via HF/deepinfra |

How the hosted estimate was built: DeepSeek-V4.1-Flash list price is $0.20 per M input and $0.60 per M output tokens. That figure comes from a fit on our own 660 logged calls. We assume about 100 input tokens and 185 output tokens per answer, the same output length as v4c. HF billed 1.44x the logged list price. We never tested DeepSeek as the Amharic responder, so this compares cost only, not quality.

## What it means

- **The alignment was nearly free.** About $1.03 in cash and 2.9 hours of free GPU turned a model that rambles in Markdown into one that stops, writes plain Amharic, and refers 96% of danger-sign questions to care. It does this while flagging only 1.3% of factual questions as urgent. Including every dead end, the project used 9.3 free GPU hours.
- **The per-answer cost is not the win.** A cheap hosted flash model costs about the same per 1,000 answers as a rented T4 (about $0.15 vs about $0.19, both estimates). Both are a few cents per hundred users.
- **The win is where it can run.** The model's weights are open and 6.7 GB. It runs offline on one 16 GB GPU, has no per-call fee, and keeps reproductive-health questions on the operator's own hardware. A clinic network, an NGO, or a phone-app backend in Ethiopia can deploy it without a foreign API account, a card, or reliable international bandwidth.
- **The cost of the trade.** The model gave up 2.0 points of AfriMMLU general knowledge. v4c also scored 0.12 lower on judge correctness than v4b, which we accepted to remove false alarms. Answers are 27% shorter in tokens, but with fixed batches the wall time per answer did not fall (757 s base vs 819 s v4c for 550 answers). That is because a batch runs as long as its longest answer, and 29.5% of answers still hit the cap.
