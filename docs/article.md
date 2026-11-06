---
title: "Teaching Tiny Aya when to say 'go now': Amharic maternal health Q&A with CuratorKIT, AlignTune and AuditKIT"
thumbnail: # TODO before publishing
authors:
- user: 5LeaderW
---

# Teaching Tiny Aya when to say "go now": Amharic maternal health Q&A with CuratorKIT, AlignTune and AuditKIT

*Lexsi Labs hackathon, Track 1. Every number below comes from the project's results files
(`results/summary_table.md`, `results/comparison_run{2,3,4}.md`, `results/referral_audit/`).*

> Research prototype, not medical advice. Non-commercial use only (CC BY-NC 4.0).

## The problem

A pregnant woman in rural Ethiopia who types a question in Amharic should get a short, correct answer, and if what she
describes is a danger sign (heavy bleeding, severe headache with blurred vision, convulsions, reduced fetal movement),
she should be told to go to a health facility now. She should not be told that for every question about nutrition or
family planning, because a model that always says "see a doctor" teaches people to ignore it.

[Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (3.35B parameters) lists Amharic and is small enough
to run cheaply. Out of the box, on our 550-question Amharic health test set, it answered in Markdown 96.0% of the time,
98.5% of its answers ran into the 256-token cap without stopping, and it almost never referred anyone to care.

## Data

We merged five public sources into one schema (seed 42, Unicode and Ethiopic punctuation normalisation, script filter,
homophone-aware deduplication):

| Source | Licence |
|---|---|
| Walelign/Amharic_SRH_Dataset | CC BY-NC 4.0 |
| ImhotepSystems/AfriHealth-QA (Amharic rows) | CC BY 4.0 |
| AmHQA (Bogale et al. 2026, Mendeley Data) | CC BY 4.0 |
| danilior/amharic_maternal_health | no licence stated |
| EthioNLP/Amharic_Instruction_dataset (general slice) | no licence stated |

Result: 7,109 train, 374 dev, and a held-out test set of 550 rows (400 AmHQA + 150 Walelign SRH). The intended
maternal benchmark was not available, so the test set is mostly general health, and much of it reads as machine
translation from English.

## Step 1: CuratorKIT

**Run 2.** CuratorKIT's TextCleaner, exact and MinHash deduplication and SchemaGate, plus project gates written on
CuratorKIT's `BaseGate` / `BaseNormalizer`: broken text (-22), English-heavy (-53), off-topic with LLM confirmation
(-15), question-answer mismatch with LLM confirmation (-36), and an EthioNLP downsample for the prompt budget (-2,139).
Train went from 7,109 to 4,806 rows.

**Run 3 (v3).** An LLM quality gate (Gemma 4 31B, keep only fluency >= 4 and correctness >= 4) cut 4,806 to 2,395 rows;
the machine-translated AmHQA answers mostly fell out (1,317 -> 142). Because the corpus had few real danger-sign
questions, we generated 140 synthetic Amharic danger-sign questions (123 kept after checks). Gemma rewrote 528 answers
to add an appropriate referral, and a second model (DeepSeek V4 Flash) verified them: 480 passed (182 urgent,
298 routine). Final v3 set: 2,691 rows, plus 763 DPO pairs.

## Step 2: AlignTune SFT, then DPO

All training used AlignTune 0.1.30 with the TRL backend, LoRA r16 / alpha 32 on every attention and MLP projection,
on free Kaggle T4 x2.

- **Run-2 SFT** on the 4,806 curated rows fixed format (Markdown 96% -> 0%) but cost fluency: the judge's fluency
  fell, which we traced to machine-translated training answers.
- **v3 SFT** (from the merged run-2 model) on the 2,691 quality-filtered rows recovered it.
- **v3 DPO** (beta 0.1, lr 5e-5) first ran out of memory on a T4. A low-memory recipe fixed it: batch 1 x grad-accum
  16, gradient checkpointing, reference log-probs precomputed once (no second model in memory), max length 768, the
  model split across both GPUs. Peak memory 7.9 GiB + 4.0 GiB.

v3 DPO was a big quality jump in blind head-to-head judging against v3 SFT (correctness 2.65 -> 3.29, fluency
3.68 -> 4.27), but the audit showed a catch: it ended **86.5% of non-urgent answers** with the same generic "go to a
health facility" sentence. The v3 pairs had only ever said "referral good".

## Step 3: AuditKIT and a referral audit, and two fixes

AuditKIT gave per-sample chrf / BLEU / ROUGE-L, and we added a regex referral audit (referral line, generic
boilerplate, urgent "go now" line) sliced by an LLM triage label for each test question (1 urgent, 102 routine,
447 factual). With only one urgent test question, urgent behaviour is measured on a held-out probe of 82 new Amharic
danger-sign questions that were never trained on.

- **v4b** kept the urgent "referral good" pairs, added routine counter-pairs (chosen = answer without the boilerplate,
  rejected = the same answer plus a sentence v3 DPO really produced), kept v3's fluency pairs with the boilerplate
  stripped, and added anti-repetition pairs. Non-urgent referral fell to 23.0%, but now **19.2% of factual answers
  ended in a false "go now" line** (for example on Bell's palsy treatment).
- **v4c** dropped the 90 urgent anti-repetition pairs and added 270 false-urgency counter-pairs: a factual or routine
  question, chosen = the clean answer, rejected = the same answer plus one of the "go now" lines v4b emitted. No LLM
  calls were needed for v4, v4b or v4c.

## Results

| Metric | Baseline | Run-2 SFT | v3 SFT | v3 DPO | v4b DPO | **v4c DPO** |
|---|---|---|---|---|---|---|
| chrF++ test (550) | 10.3 | 9.1 | 9.8 | 10.7 | 12.3 | **12.3** |
| chrF++ Walelign SRH (150) | 17.8 | 15.1 | 16.6 | 19.0 | 18.4 | **19.0** |
| AuditKIT chrf, per-sample mean | 18.8 | 18.8 | 19.9 | 21.3 | 20.1 | **22.7** |
| Belebele amh acc % | 61.7 | 63.0 | 63.4 | 62.8 | 62.9 | **63.6** |
| AfriMMLU amh acc % | 46.4 | 44.2 | 44.4 | 44.0 | 43.8 | **44.4** |
| Hit 256-token cap % | 98.5 | 44.7 | 35.3 | 14.0 | 34.5 | **29.5** |
| Referral line %, non-urgent test | 0.7 | 0.0 | 12.8 | 86.5 | 23.0 | **5.8** |
| False "go now" %, factual test | 0.4 | 0.0 | 0.0 | 1.3 | 19.2 | **1.3** |
| Referral %, 82 held-out danger-sign questions | n/a | n/a | 67.1 | 100.0 | 98.8 | **96.3** |

Blind LLM judge (DeepSeek-V4.1-Flash, same 100 rows, correctness / fluency out of 5; compare within a row):

| Pass | First system | Second system |
|---|---|---|
| v3 SFT vs v3 DPO | 2.65 / 3.68 | 3.29 / 4.27 |
| v3 DPO vs v4b | 3.18 / 4.34 | 3.12 / 4.47 |
| v4b vs v4c | 3.13 / 4.50 | 3.01 / 4.43 |
| v4b vs v4c, labels reversed | 3.13 / 4.50 | 3.00 / 4.39 |

We submitted **v4c**. It gives up 0.12 judge correctness against v4b (the same in both label orders, so not position
bias), in exchange for cutting false "go now" lines from 19.2% to 1.3% of factual answers while keeping 96.3% referral
on danger signs and the best Belebele score of any stage. For a health assistant, false alarms erode trust, so we took
that trade.

## Cost

All GPU work ran on free Kaggle T4 x2 in private kernels. LLM curation, triage labels, probe generation and judging
used Hugging Face Inference Providers: about $1.03 billed (HF billing usage, 5-6 November 2026), plus about $0.20 of
free promotional credit on another API for the run-2 checks and judge. No paid GPUs, no HF Jobs.

## What did not work, and what is left

- Run-2 SFT on machine-translated answers made the model less fluent; filtering by an LLM quality score fixed it.
- DPO on "referral good" pairs alone produced a boilerplate machine. Counter-pairs, built by regex from the model's own
  bad outputs, fixed it cheaply, but each fix exposed the next failure (blanket referral -> false urgency).
- 29.5% of answers still hit the length cap. Repetition loops are the main remaining failure.
- 3 of 82 danger-sign answers miss the referral. The judge's "safe referral when needed" rate fell from 76.9% (v4b) to
  55.1% (v4c), because on some routine questions it wanted a referral that v4c now leaves out.

## Limitations

The judge is an LLM, not a clinician, with about +-0.1 run-to-run noise; no human medical review was done. Most test
references are machine-translated. The danger-sign questions and referral edits are synthetic, so the model learns
surface patterns, not triage. Two training sources state no licence, so their rows were removed from the released datasets (the model was trained before that removal). The base model is CC BY-NC 4.0, so the model and data
are non-commercial only.

## Reproduce

- Code, Kaggle kernel builders and exact commands (MIT): [github.com/Sgfhnn/amharic-maternal-aya](https://github.com/Sgfhnn/amharic-maternal-aya)
- Try it: [demo Space](https://huggingface.co/spaces/5LeaderW/amharic-maternal-health-aya) (ZeroGPU)
- Models: [collection](https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-models-6aedaabef1e7b1fa856a2eec): [merged v4c model](https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health), [all LoRA adapters](https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health-adapters)
- Data and logs: [collection](https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-data-and-logs-6aedaabfea4165a8357c79d6): [curated SFT data](https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-sft-curated-v3), [DPO pairs](https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-dpo-pairs), [evaluation results and logs](https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-aya-eval-logs)

Model and data are CC BY-NC 4.0 (non-commercial, Tiny Aya licence); not medical advice.
