# Amharic Health Instruction Dataset (merged, cleaned) — LexiLabs hackathon build

Local-only dataset for fine-tuning Cohere Tiny Aya (CuratorKIT → AlignTune → AuditKIT).
**Not uploaded anywhere. Licence of the merged set: NON-COMMERCIAL, research use only** (see Licences).

Build: `python build_dataset.py` from this folder (seed 42). Raw downloads live in `raw/`.

**v2 (2026-11-05): `kaggle_afri_enhanced` removed from train/dev** (LLM-synthetic, Markdown-heavy, CC BY-SA). The rows are still loaded and deduplicated so the random stream is unchanged: `test.jsonl` is byte-identical to v1 (verified with `cmp`); the EthioNLP slice is identical; train/dev are reshuffled. v1 files are kept in `../work/backup_v1/`. `EXCLUDE_SOURCES` in `build_dataset.py` controls this.

## Files
| File | What |
|---|---|
| `train.jsonl` / `dev.jsonl` / `test.jsonl` | Unified schema rows |
| `chat/{train,dev,test}.jsonl` | `messages` format (user/assistant) for AlignTune SFT; user turn = context + "\n\n" + question when context exists |
| `stats.json` | Raw rows, drops by reason, per-split/source/topic counts, avg lengths |
| `build_dataset.py`, `build_log.txt` | Reproducible pipeline and its output |

Schema: `id, source, license, question, answer, context, topic (maternal|srh|general_health|general), split, synthetic`
(`synthetic` = true for LLM-rewritten rows; extra field beyond the requested schema).

## Final counts
| Split | Rows |
|---|---|
| train | 7,109 |
| dev | 374 (5% of train pool) |
| test | 550 |

| Source | Raw | Kept (all splits) | train / dev / test |
|---|---|---|---|
| walelign_srh | 1,000 | 839 | 654 / 35 / 150 |
| afrihealth_qa (Amh_Eth) | 2,307 | 2,271 | 2,177 / 94 / 0 |
| danilior_maternal | 105 | 105 | 101 / 4 / 0 |
| amhqa_mendeley | 2,051 | 1,818 | 1,334 / 84 / 400 |
| ~~kaggle_afri_enhanced~~ | 2,307 | 0 (2,263 after cleaning, excluded in v2) | 0 / 0 / 0 |
| ethionlp_general | 5,000 sampled | 3,000 | 2,843 / 157 / 0 |

Topics (train): srh 2,322 · general 2,843 · general_health 1,172 · maternal 772. Avg answer length: train 235 chars, dev 264, test 507 (keyword heuristic, see below).

## Sources and attribution
1. **Walelign/Amharic_SRH_Dataset** (Hugging Face) — Amharic SRH QA, CC BY-NC 4.0.
2. **ImhotepSystems/AfriHealth-QA** (Hugging Face) — Zindi/ITU Multilingual Health QA challenge data curated by the HASH Consortium; Amh_Eth rows from train+validation only (test_holdout has no answers). CC BY 4.0.
3. **HanaMekonen/AmharicMaternal-and-NeonatalHealth** — NOT INCLUDED: repository returns "not found" both anonymously and with the user's HF token (user exists, 0 public datasets; probably private or not yet released). A related gated set, `israel/AmharicMaternalHealth`, exists but access is not granted to this account.
4. **danilior/amharic_maternal_health** (Hugging Face) — 105 maternal QA rows. No licence stated on the card (treat as all-rights-reserved until clarified).
5. **AmHQA** — Bogale, B., Tegegne, T., Abate, S. T., Belay, G. (2026). *AmHQA: An Amharic Health Question Answering Dataset.* Mendeley Data, V1, doi:10.17632/8cks7m5f8s.1. CC BY 4.0. Note the public file is a single `AmharicHQA.xlsx` (2,051 rows, columns qtype/ጥያቄ/መልስ), not the 1,600/400 CSVs the description mentions, so the 400-row test split here is a seeded random split, not the authors' official one.
6. **(Excluded in v2)** **Kaggle dokster/health-qa-in-low-resource-african-languages** ("Multilingual Health QA in Low-Resource African Languages", Uncharted Data Challenge writeup) — CC BY-SA 4.0. Its `input`/`output` columns are the same 2,307 AfriHealth Amharic rows (same IDs), so only its LLM-generated `enhanced_prompt`/`enhanced_completion` pairs (Adaption AI) are used, flagged `synthetic=true`.
7. **EthioNLP/Amharic_Instruction_dataset** (Walia-LLM SFT data, Azime et al. 2024, arXiv:2402.08015) — 3,000-row general slice from train, preferring rows with health keywords (2,424 of 3,000). Sub-tasks used: XL-Sum summarisation, title generation, Eng→Amh MT, story/poem generation, AmharicQA. Excluded: spellcheck, reverse summarisation, sentiment, NER, mezmur/zefen. No licence on the card; upstream components carry their own licences.

## Licences
Mixed (v2): CC BY-NC 4.0 + CC BY 4.0 + unspecified. The CC BY-SA Kaggle rows are no longer included, which removes the BY-SA/BY-NC conflict. Because of the NC component the merged set is **non-commercial only**. Update 2026-11-06: the danilior and EthioNLP licences were never clarified, so all of their rows were removed from the published Hugging Face dataset repos for licence reasons (this build description still lists them because the training runs used them). Attribute every source above in any model card.

## Cleaning
1. Unicode NFC; strip zero-width chars; collapse whitespace; `::` and `፡፡` typed for the Ethiopic full stop → `።`.
2. Drop empty question/answer; drop answers < 15 chars.
3. Script filter: drop rows where Ethiopic letters (U+1200–U+137F) are < 50% of letters in question+answer.
4. Dedup by priority (AmHQA > SRH > AfriHealth > danilior > Kaggle-enhanced > EthioNLP): exact/near duplicates via a normalised hash (lowercase, punctuation/space removed, Amharic homophone letters folded: ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ), then question-level dedup on the same normalisation.
5. Split: test = 400 AmHQA + 150 Walelign SRH (substitute for the unavailable HanaMekonen in-domain slice); any train/dev row whose normalised question matches a test question is removed (0 found after dedup); dev = random 5% of the rest.
6. Topic: `maternal` if pregnancy/birth/newborn/breastfeeding keywords appear; otherwise `srh` for SRH sources, `general_health` for AmHQA, `general` for EthioNLP.

## Known quality issues
- **Machine translation.** AmHQA reads as an Amharic translation of English consumer-health QA (MedQuAD-style: CDC/NIH topics such as LCM, angiostrongyliasis, rare diseases), with literal transliterations and awkward phrasing. Walelign SRH also looks translated from English clinical/SRH QA and includes some off-topic items (e.g. clinical-trial adverse-event grading).
- **Synthetic long-form (resolved in v2, source dropped).** `kaggle_afri_enhanced` (2,263 rows, ~1,650 chars avg) is LLM-written, heavy Markdown (headers, bold, bullets), with English terms in parentheses. It dominates answer length; consider down-weighting or stripping Markdown if you want short, plain answers.
- **Length imbalance.** AfriHealth answers ~106 chars, danilior ~92, SRH ~180, AmHQA ~700, EthioNLP ~1,300+.
- **Style.** danilior answers are chatty with many `!`; it is tiny (105). AfriHealth has typos and missing spaces (e.g. merged words) and ~40 rows with heavy English.
- **Test set is mostly general health** (367/550) because the maternal benchmark was unavailable; the 150 SRH rows are the in-domain portion. AmHQA test is a self-made split.
- Topic labels are keyword-based, not manually verified.

## Curated subset (run two, 2026-11-05)
`curated/` holds the CuratorKIT-curated train/dev (`train.jsonl` 4,806, `dev.jsonl` 249, plus `chat/` versions), `rejected.jsonl` with a reason per dropped row, `llm_checks.jsonl` with the LLM verdicts, and `curation_report.md` / `curation_stats.json` with counts per step. Built by `../scripts/curate.py`. The test split is unchanged. Same licence terms as the parent set.
