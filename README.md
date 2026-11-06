# amharic-maternal-aya

**Tiny Aya Earth fine-tuned for Amharic maternal and reproductive health Q&A.**
Lexsi Labs hackathon, Track 1. Chain: **CuratorKIT -> AlignTune (SFT + DPO) -> AuditKIT**, all on free Kaggle T4 x2.

> Research prototype, not medical advice. Model and data are **non-commercial only (CC BY-NC 4.0 and the Tiny Aya licence)**; the code is MIT. See [Licence](#licence).

| Artifact | Where |
|---|---|
| Live demo (Gradio, ZeroGPU) | [spaces/5LeaderW/amharic-maternal-health-aya](https://huggingface.co/spaces/5LeaderW/amharic-maternal-health-aya) |
| Merged model (v4c DPO, fp16) | [5LeaderW/tiny-aya-earth-amharic-maternal-health](https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health) |
| All LoRA adapters (run-2 B, v3 SFT, v3 DPO, v4b, v4c) | [5LeaderW/tiny-aya-earth-amharic-maternal-health-adapters](https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health-adapters) |
| Curated SFT data (v3, licence-cleaned) + CuratorKIT reports | [5LeaderW/amharic-maternal-health-sft-curated-v3](https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-sft-curated-v3) |
| DPO pairs v3 / v4 / v4b / v4c (licence-cleaned) | [5LeaderW/amharic-maternal-health-dpo-pairs](https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-dpo-pairs) |
| Predictions, judge outputs, referral audit, urgent probe, training logs | [5LeaderW/amharic-maternal-health-aya-eval-logs](https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-aya-eval-logs) |
| Collections | [models](https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-models-6aedaabef1e7b1fa856a2eec), [data and logs](https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-data-and-logs-6aedaabfea4165a8357c79d6) |
| Write-up | [`docs/abstract.md`](docs/abstract.md), [`docs/article.md`](docs/article.md) (this README doubles as the article) |
| Demo app source | [`space/`](space/) |

## Problem

Amharic speakers ask maternal and reproductive health questions far from clinics. A small open model could answer
them cheaply, but it has to (1) answer briefly and correctly in natural Amharic, (2) tell people with danger signs to go
to a health facility **now**, and (3) not say that to everyone, or the warning stops meaning anything.

Out of the box, [CohereLabs/tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (3.35B) answered our test
questions in Markdown 96% of the time, 98.5% of answers ran into the 256-token cap, and it almost never referred anyone.

## Approach

1. **Dataset** (`data/build_dataset.py`, seed 42): five public Amharic sources merged and cleaned -> train 7,109,
   dev 374, held-out test 550 (400 AmHQA + 150 Walelign SRH). Sources and licences in [`data/README.md`](data/README.md).
2. **CuratorKIT, run 2** (`scripts/curate.py`): CuratorKIT TextCleaner, exact + MinHash dedup, SchemaGate, plus
   project gates on CuratorKIT `BaseGate` / `BaseNormalizer` (broken text, English-heavy, off-topic and Q/A mismatch
   with LLM confirmation, EthioNLP downsample, danger-sign referral line). 7,109 -> 4,806 rows.
3. **AlignTune SFT, run 2** (`scripts/train_sft.py`): LoRA r16 / alpha 32, lr 1e-4, 1 epoch -> adapter "B".
4. **CuratorKIT, v3** (`scripts/curate_v3.py`): LLM quality gate (Gemma 4 31B, fluency and correctness >= 4),
   480 verified referral rewrites, 123 synthetic danger-sign questions. 4,806 -> 2,691 rows + 763 DPO pairs.
5. **AlignTune SFT v3** from the merged run-2 model, then **AlignTune DPO v3** (`scripts/train_dpo_lowmem.py`, beta 0.1,
   lr 5e-5, batch 1 x 16, gradient checkpointing, precomputed reference log-probs, max length 768, model split over two T4s).
6. **AuditKIT + referral audit** (`scripts/score.py`, `scripts/referral_audit.py`, `scripts/llm_judge.py`): found that
   v3 DPO ended 86.5% of non-urgent answers with a boilerplate referral.
7. **CuratorKIT v4 / v4b / v4c** (`scripts/curate_v4*.py`, no LLM): counter-pairs built from the model's own bad
   outputs. v4b fixed the boilerplate but added false "go now" lines (19.2% of factual answers); **v4c** added 270
   false-urgency counter-pairs and is the submitted model.

The submitted model = official base + run-2 B adapter + v3 SFT adapter + v4c DPO adapter, merged
(`scripts/merge_adapter.py`, kernel `scripts/kaggle/merge4c_main.py`). Merge check on 16 test rows: the merged model gives
16/16 identical answers to the adapter path the v4c eval used, and 13/16 identical to the stored v4c predictions (the 3
differences appear identically in the adapter path, so they come from batch padding, not the merge).

## Results

Test = 550 held-out rows, greedy, max 256 new tokens. Referral rates: regex audit (`scripts/referral_lib.py`) sliced by
LLM triage labels (1 urgent / 102 routine / 447 factual). Urgent behaviour: 82 held-out danger-sign questions
(`data/probe_urgent_v4.jsonl`, never trained on). Full table: [`results/summary_table.md`](results/summary_table.md).

| Metric | Baseline | Run-2 B SFT | v3 SFT | v3 DPO | v4b DPO | **v4c DPO** |
|---|---|---|---|---|---|---|
| chrF++ test (550) | 10.3 | 9.1 | 9.8 | 10.7 | 12.3 | **12.3** |
| chrF++ AmHQA (400) | 9.2 | 8.3 | 8.9 | 9.5 | 11.5 | **11.4** |
| chrF++ Walelign SRH (150) | 17.8 | 15.1 | 16.6 | 19.0 | 18.4 | **19.0** |
| AuditKIT chrf, per-sample mean | 18.8 | 18.8 | 19.9 | 21.3 | 20.1 | **22.7** |
| Belebele amh_Ethi acc % (900) | 61.7 | 63.0 | 63.4 | 62.8 | 62.9 | **63.6** |
| AfriMMLU amh acc % (500) | 46.4 | 44.2 | 44.4 | 44.0 | 43.8 | **44.4** |
| Hit 256-token cap % | 98.5 | 44.7 | 35.3 | 14.0 | 34.5 | **29.5** |
| Answers with Markdown % | 96.0 | 0.0 | 0.0 | 0.0 | 0.2 | **0.0** |
| Referral line %, non-urgent test (549) | 0.7 | 0.0 | 12.8 | 86.5 | 23.0 | **5.8** |
| False "go now" line %, factual test (447) | 0.4 | 0.0 | 0.0 | 1.3 | 19.2 | **1.3** |
| Referral line %, urgent probe (82) | n/a | n/a | 67.1 | 100.0 | 98.8 | **96.3** |

Blind LLM judge (DeepSeek-V4.1-Flash, same 100 rows, correctness / fluency 1-5; each row is a separate pass):

| Pass | Baseline | Run-2 B | v3 SFT | v3 DPO | v4b | v4c |
|---|---|---|---|---|---|---|
| Five-way (run 3) | 2.56 / 3.80 | 2.38 / 3.37 | 2.58 / 3.66 | | | |
| v3 SFT vs v3 DPO | | | 2.65 / 3.68 | 3.29 / 4.27 | | |
| v3 DPO vs v4b | | | | 3.18 / 4.34 | 3.12 / 4.47 | |
| v4b vs v4c | | | | | 3.13 / 4.50 | 3.01 / 4.43 |
| v4b vs v4c, labels reversed | | | | | 3.13 / 4.50 | 3.00 / 4.39 |

v4c trades 0.12 judge correctness versus v4b for removing false "go now" lines (19.2% -> 1.3% of factual answers).

## Reproduce

Hardware: **free Kaggle T4 x2** (private kernels, account `neimashube`). No paid GPU, no HF Jobs.
Cost: LLM curation, triage labels, probe generation and judging used Hugging Face Inference Providers, **about $1.03
billed** in total (HF billing usage for 5-6 Nov 2026; under $1.20), plus about $0.20 of free promotional Cline credit for
the run-2 LLM checks and judge. Per-call list-price estimates: `work/run3/hf_spend.jsonl` (in the eval-logs dataset as `spend/hf_spend.jsonl`).

Software: transformers 5.15.0, peft >= 0.18, trl 1.7.1, aligntune 0.1.30 (`--no-deps`), curatorkit, auditkit 1.1.3,
sacrebleu. On Kaggle uninstall `torchao` (0.10 breaks PEFT LoRA).

```bash
# 0. local tools
pip install "transformers==5.15.0" peft accelerate sacrebleu sentencepiece auditkit curatorkit huggingface_hub
export KAGGLE_API_TOKEN=...   # Kaggle bearer token; never commit it
export HF_TOKEN=...           # needs access to the gated CohereLabs/tiny-aya-earth and inference credit

# 1. data
cd data && python build_dataset.py && cd ..          # needs the raw downloads listed in data/README.md

# 2. CuratorKIT run 2 -> data/curated/, then SFT run 2 (kernel amh-tinyaya-r2-curated)
python scripts/curate.py
cd scripts/kaggle && python build_kernels.py
python upload_ds.py amh-tinyaya-run2 <staged files>      # private Kaggle dataset; see scripts/README.md
python kag.py push build/r2_curated/meta.json build/r2_curated/kernel.py
bash pull_run2.sh                                       # pull, score, judge, results/comparison_run2.md
cd ../..

# 3. CuratorKIT v3 (LLM quality gate + rewrites), v3 SFT, v3 DPO (low-memory)
HF_CAP_USD=1.20 python scripts/curate_v3.py
python scripts/kaggle/kag.py push scripts/kaggle/build/r3/meta.json scripts/kaggle/build/r3/kernel.py
python scripts/kaggle/kag.py push scripts/kaggle/build/r3dpo/meta.json scripts/kaggle/build/r3dpo/kernel.py
bash scripts/kaggle/pull_run3.sh && bash scripts/kaggle/pull_r3dpo.sh

# 4. referral audit, urgent probe, v4 / v4b / v4c pairs and DPO
python scripts/label_test_care.py && python scripts/make_urgent_probe.py
python scripts/curate_v4.py && python scripts/curate_v4b.py && python scripts/curate_v4c.py
python scripts/kaggle/kag.py push scripts/kaggle/build/r4cdpo/meta.json scripts/kaggle/build/r4cdpo/kernel.py
bash scripts/kaggle/pull_r4cdpo.sh                      # scores, referral audit, judge, results/comparison_run4.md

# 5. merge the stack into one fp16 model + sanity check (kernel amh-tinyaya-merge-v4c)
python scripts/kaggle/upload_ds.py amh-tinyaya-merge4c work/kaggle_ds_merge4c/*
python scripts/kaggle/kag.py push scripts/kaggle/build/merge4c/meta.json scripts/kaggle/build/merge4c/kernel.py
python scripts/check_merge.py                           # compares merged vs adapter vs stored v4c predictions

# 6. tables
python scripts/make_summary_table.py
```

Every Kaggle kernel assembles the official model itself (`scripts/kaggle/assemble_model.py`): official config and
tokenizer files from the private dataset, weight shards SHA-256-checked against the official LFS object ids, so no HF
token ever sits in a kernel. Kernel wall times on T4 x2: run-2 SFT about 1.6 h, v3 SFT about 55 min, each DPO
28-44 min, merge + sanity check 6 min.

Single GPU (Colab / local), with the published adapters:
```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
m = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-earth", dtype=torch.float16)
for sub in ["run2_B_sft", "v3_sft", "v4c_dpo"]:
    m = PeftModel.from_pretrained(m, "5LeaderW/tiny-aya-earth-amharic-maternal-health-adapters", subfolder=sub).merge_and_unload()
```

## Repository layout

| Path | What |
|---|---|
| `scripts/` | curation (`curate*.py`), training (`train_sft.py`, `train_dpo_lowmem.py`), evaluation (`eval_lib.py`, `score.py`, `llm_judge.py`, `referral_audit.py`), merge, packaging, `secret_scan.py` |
| `scripts/kaggle/` | kernel builders (`common_driver.py`, `*_main.py`, `build_kernels.py`), built kernels in `build/`, Kaggle REST helpers, pull scripts |
| `configs/` | hyper-parameters, data counts and final metrics of every training stage; eval and stacking settings |
| `data/` | dataset build script, its log and stats, CuratorKIT reports (the data itself is on HF) |
| `results/` | summary and per-run comparison tables, metrics JSONs, judge summaries, referral audit, merge check (`results/merge_check/merge_check.md`) |
| `docs/` | abstract and HF article draft |
| `space/` | Gradio app for a later HF Space |

## Limitations

- Not a medical device; judge correctness is about 3 / 5. No clinician reviewed outputs.
- Judge noise is about +-0.1 between passes on the same rows.
- Most test references (AmHQA, much of Walelign) are machine translations; chrF++ against them is low for every system.
- The test set is mostly general health (150 in-domain SRH rows, 88 maternal by keyword).
- Residual over-referral: 5.8% of non-urgent answers (13.7% of routine ones) still end in a referral line.
- 3 of 82 held-out danger-sign answers miss the referral; 7.3% lack urgent wording.
- Synthetic danger-sign questions (123 in training, 82 in the probe) and synthetic referral edits (LLM rewrites and
  regex-built DPO pairs): the model learns surface patterns, not triage.
- 29.5% of answers still hit the 256-token cap (repetition loops).

## Licence

- **Code in this repo: MIT** ([`LICENSE`](LICENSE)). The MIT licence covers the code only.
- **Model weights and adapters: CC BY-NC 4.0, non-commercial only**, inherited from CohereLabs/tiny-aya-earth under the
  Tiny Aya licence (users must also follow Cohere Labs' Acceptable Use Policy). The model and data stay non-commercial
  whatever the code licence says.
- **Data: CC BY-NC 4.0, non-commercial only.** It mixes CC BY-NC 4.0 (Walelign SRH) and CC BY 4.0 (AfriHealth-QA,
  AmHQA) sources plus synthetic rows. Rows from danilior/amharic_maternal_health and EthioNLP/Amharic_Instruction_dataset
  (no stated licence) were removed from the published datasets; the models were trained before that removal, which the
  model cards disclose. Details in [`NOTICE.md`](NOTICE.md).
