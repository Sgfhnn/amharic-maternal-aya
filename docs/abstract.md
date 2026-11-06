# Teaching a small multilingual model when to say "go now": Amharic maternal and reproductive health Q&A with Tiny Aya Earth

**Track 1, Lexsi Labs hackathon. Model: CohereLabs/tiny-aya-earth (3.35B). Chain: CuratorKIT -> AlignTune (SFT + DPO) -> AuditKIT.**

Amharic has tens of millions of speakers, and maternal and reproductive health questions are often asked far from a
clinic. Small open models that run cheaply could help, but out of the box Tiny Aya Earth answers Amharic health
questions in Markdown-heavy text that almost never stops (98.5% of test answers hit the 256-token cap), and it rarely
points people to care.

We merged five public Amharic sources (Walelign SRH, AfriHealth-QA, AmHQA, danilior maternal, an EthioNLP instruction
slice) into a 7,109-row training set with a 550-row held-out test set (400 AmHQA + 150 Walelign SRH). CuratorKIT
curation ran in two passes: rule and LLM-confirmed gates (deduplication, broken text, English-heavy, off-topic,
question-answer mismatch) cut the set to 4,806 rows, and an LLM quality gate plus verified referral rewrites and 123
synthetic danger-sign questions produced a 2,691-row v3 set. AlignTune trained LoRA SFT on the curated data, then DPO.
The first DPO taught "referral good" everywhere and put a generic facility line on 86.5% of non-urgent answers.
Two rounds of regex-built counter-pairs (v4b, then v4c) taught the model to refer when a question is urgent and stay
quiet otherwise. AuditKIT-style per-sample metrics, sacrebleu chrF++, a regex referral audit and a blind LLM judge
scored every stage.

The submitted model, v4c DPO, compared with the base model:

- Corpus chrF++ 10.3 -> 12.3; Walelign SRH chrF++ 17.8 -> 19.0; AuditKIT per-sample chrf 18.8 -> 22.7.
- Answers hitting the 256-token cap 98.5% -> 29.5%; Markdown 96.0% -> 0.0%.
- Belebele Amharic 61.7 -> 63.6; AfriMMLU Amharic 46.4 -> 44.4 (a 2-point loss in general knowledge).
- On 82 held-out danger-sign questions, 96.3% of answers refer the user to care (92.7% with urgent wording).
- Only 5.8% of non-urgent answers carry a referral line, and only 1.3% of factual answers carry a false "go now" line
  (the previous candidate, v4b, had 19.2%).

In blind head-to-head LLM judging (DeepSeek-V4.1-Flash, 100 rows), DPO lifted correctness from 2.65 to 3.29 and fluency
from 3.68 to 4.27 over v3 SFT. v4c scored 0.12 lower in correctness than v4b (3.01 vs 3.13, the same in both label
orders). We accepted that cost to remove false alarms.

All training ran on free Kaggle T4 x2 GPUs. LLM curation and judging cost about $1.03 of Hugging Face
Inference Providers credit (billed), plus about $0.20 of free promotional API credit.

Limitations: the judge is an LLM, not a clinician, with about +-0.1 run-to-run noise. Most test references are machine
translations from English. The urgent questions and referral edits are synthetic, so the model learns surface patterns,
not triage. 29.5% of answers still hit the length cap. Rows from two unlicensed sources were not released. The model and data are for
non-commercial research only (CC BY-NC 4.0) and are not medical advice.

---
Links (not part of the abstract text): model https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health · demo https://huggingface.co/spaces/5LeaderW/amharic-maternal-health-aya · models https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-models-6aedaabef1e7b1fa856a2eec · data and logs https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-data-and-logs-6aedaabfea4165a8357c79d6 · code https://github.com/Sgfhnn/amharic-maternal-aya
