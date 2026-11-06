# Reproducing the Tiny Aya Amharic health baseline and SFT run

Everything runs on free Kaggle GPUs (T4 x2). No paid compute was used.

## Pipeline
CuratorKIT -> AlignTune -> AuditKIT, all from the Lexsi Labs repos:
- **CuratorKIT** runs inside AlignTune's data manager (SchemaGate, chat normalisation) when the SFT data is loaded.
- **AlignTune 0.1.30** (TRL backend) does the LoRA SFT. `train_sft.py` falls back to plain TRL `SFTTrainer` with the same hyper-parameters if AlignTune fails, and `train_info.json` records which one ran.
- **AuditKIT 1.1.3** gives per-sample `chrf`, `bleu`, `rouge_l` and `f1_score` (from `ak.evaluate` replaying the saved predictions). AuditKIT has no chrF++ and no corpus BLEU, so `score.py` also computes those with sacrebleu.

## Files
| File | What |
|---|---|
| `../data/build_dataset.py` | Builds the dataset (seed 42). v2 drops `kaggle_afri_enhanced`, and the test split is byte-identical to v1 |
| `eval_lib.py` | Runs on the GPU. Greedy generation on `data/test.jsonl` (550 rows, max 256 new tokens, chat template, no system prompt beyond the model's built-in preamble), plus zero-shot Belebele `amh_Ethi` (900) and AfriMMLU `amh` test (500) multiple choice, scored by argmax over the next-token logits for A-D. It checks that fp16 output is finite and falls back to fp32 if not |
| `train_sft.py` | LoRA SFT on `data/chat/train.jsonl`, with eval on `data/chat/dev.jsonl`. Settings: r=16, alpha=32, dropout 0.05, all attention and MLP projections, lr 2e-4 cosine, warmup 3%, 1 epoch, max length 1024, batch 2 x grad-accum 8, fp16, gradient checkpointing. Loss covers assistant tokens only (the stock chat template is copied with `{% generation %}` tags, so the rendered text is unchanged). A time budget stops training cleanly |
| `score.py` | Turns predictions into `metrics.json`: corpus chrF++ / chrF / BLEU-13a / spBLEU-flores200, a breakdown by source and topic, AuditKIT per-sample means, and output diagnostics |
| `kaggle/common_driver.py`, `kaggle/*_main.py`, `kaggle/build_kernels.py` | Assemble the single-file Kaggle kernels into `kaggle/build/{baseline,sft}/kernel.py` |
| `kaggle/kag.py`, `kaggle/upload_ds.py` | Minimal Kaggle REST helpers: push a kernel, check status, download output, upload a private dataset |

## Model
`CohereLabs/tiny-aya-earth` lists Amharic, but the repo is gated, and the saved HF tokens are not on its authorized list (HTTP 403). The runs therefore load the ungated mirror `1-800-LLMs/tiny-aya-earth`. Its two safetensors shards and `tokenizer.json` have the same git/LFS object ids as the official repo, so the weights are identical. Only `config.json` and `tokenizer_config.json` differ, by 2 to 4 bytes. To switch to the official repo, accept the licence on its HF page, then pass `--model CohereLabs/tiny-aya-earth` (eval) or `--model_id` (train) with an `HF_TOKEN`.

## Steps
```bash
cd data && python build_dataset.py && cd ..
# stage the private Kaggle dataset
mkdir -p work/kaggle_ds && cp data/test.jsonl scripts/eval_lib.py scripts/train_sft.py work/kaggle_ds/
cp data/chat/train.jsonl work/kaggle_ds/chat_train.jsonl; cp data/chat/dev.jsonl work/kaggle_ds/chat_dev.jsonl
cp work/bench/belebele_amh.jsonl work/bench/afrimmlu_amh_test.tsv work/kaggle_ds/   # from HF facebook/belebele, masakhane/afrimmlu
cd scripts/kaggle
python upload_ds.py amh-health-tinyaya-eval ../../work/kaggle_ds/*       # private dataset
python build_kernels.py
python kag.py push build/baseline/meta.json build/baseline/kernel.py     # before
python kag.py push build/sft/meta.json build/sft/kernel.py               # train + after
python kag.py status neimashube/amh-tinyaya-sft
python kag.py output neimashube/amh-tinyaya-baseline ../../results/kaggle_baseline
python ../score.py ../../results/baseline ; python ../score.py ../../results/sft/eval
```
The Kaggle REST calls need `KAGGLE_API_TOKEN` (a bearer token for the Kaggle account `neimashube`). Both kernels and the dataset are private.

## Running on a single GPU (Colab or local)
```bash
pip install "transformers==5.15.0" peft accelerate sacrebleu auditkit
python scripts/eval_lib.py --task all --out results/baseline          # finds test.jsonl etc. under the cwd
pip install "trl==1.7.1" curatorkit && pip install --no-deps aligntune==0.1.30   # plus any missing light deps
python scripts/train_sft.py --out results/sft --stop_hours 4
python scripts/eval_lib.py --task all --adapter results/sft/adapter --out results/sft/eval
```

## Run two (2026-11-05): NaN fix, CuratorKIT curation, raw vs curated SFT
| File | What |
|---|---|
| `curate.py` | CuratorKIT curation of train/dev into `data/curated/` (test untouched). CuratorKIT TextCleaner, Exact/MinHash dedup and SchemaGate plus project gates on CuratorKIT's `BaseGate`/`BaseNormalizer`: broken text, URL-sentence strip, English-heavy, off-topic (keyword + LLM confirm), Q/A mismatch (char-n-gram cosine + acronym conflict -> LLM confirm), EthioNLP downsample, danger-sign referral line. Report: `data/curated/curation_report.md` |
| `llm_client.py` | Minimal Cline API client (streaming, OpenAI-compatible); stops on any billing/credit error |
| `llm_judge.py` | LLM judge on a fixed seeded 100-row test sample (60 SRH + 40 AmHQA): correctness 1-5, Amharic fluency 1-5, safe referral yes/no/NA; three systems per call, blind shuffled labels |
| `make_comparison_run2.py` | Builds `results/comparison_run2.md` |
| `train_sft.py` | Now: length guard (drops rows whose prompt leaves < 16 supervised tokens inside max_len 1024; this was the NaN cause), LoRA params forced fp32 with fp16 autocast, lr 1e-4, max_grad_norm 1.0, NaN watcher, `--model_dir/--train_file/--dev_file/--max_steps` |
| `kaggle/assemble_model.py` | Builds a byte-exact CohereLabs/tiny-aya-earth snapshot in the kernel: official config/tokenizer files shipped in the private dataset, weight shards fetched from the mirror and SHA-256-checked against the official LFS oids (no HF token in the kernel) |
| `kaggle/sft2_main.py`, `kaggle/build_kernels.py` | Kernels `r2_sanity` (50 steps), `r2_raw` (Model A), `r2_curated` (Model B + official baseline re-eval) |
| `kaggle/pull_run2.sh` | Pull outputs, score, judge, rebuild the comparison |
| `run1/` | Run-one copies of train_sft.py and the kernel driver |

Private Kaggle datasets: `neimashube/amh-tinyaya-run2` (code, official tokenizer/config files, raw chat train/dev, test, benchmarks), `neimashube/amh-tinyaya-curated-v2` (curated chat train/dev). `neimashube/amh-tinyaya-curated` is a stale first upload (pre danger-sign regex fix), unused.
