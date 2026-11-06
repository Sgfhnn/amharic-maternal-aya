# Run two: baseline vs raw-SFT vs curated-SFT (Tiny Aya Earth, Amharic maternal/SRH health)

| Metric | Baseline | Run 1 SFT (raw v2, lr 2e-4) | A: raw-SFT | B: curated-SFT |
|---|---|---|---|---|
| chrF++ test (corpus) | 10.3 | 9.0 | 8.9 | 9.1 |
| BLEU-13a test (corpus) | 0.9 | 1.0 | 0.9 | 1.4 |
| spBLEU flores200 test (corpus) | 5.2 | 3.7 | 3.6 | 3.9 |
| chrF++ AmHQA (400) | 9.2 | 8.2 | 8.0 | 8.3 |
| chrF++ Walelign SRH (150) | 17.8 | 15.4 | 15.6 | 15.1 |
| chrF++ maternal-topic rows (88) | 10.8 | 9.0 | 9.3 | 8.7 |
| AuditKIT chrf (per-sample mean) | 18.8 | 19.0 | 18.8 | 18.8 |
| AuditKIT rouge_l (per-sample mean) | 11.2 | 11.3 | 11.2 | 11.2 |
| Belebele amh_Ethi acc % (900) | 61.7 | 61.0 | 63.7 | 63.0 |
| AfriMMLU amh acc % (500) | 46.4 | 42.8 | 45.0 | 44.2 |
| Answers with Markdown % | 96.0 | 0.0 | 0.0 | 0.0 |
| Hit 256-token cap % | 98.5 | 46.2 | 44.2 | 44.7 |
| Referral keyword % (all test) | 4.2 | 2.5 | 2.2 | 1.6 |
| Avg answer chars (ref 507) | 436.7 | 247.0 | 243.5 | 254.2 |
| Ethiopic script share % | 84.8 | 98.4 | 98.3 | 98.4 |
| LLM judge correctness 1-5 | 2.62 | n/a | 2.86 | 2.87 |
| LLM judge Amharic fluency 1-5 | 3.99 | n/a | 3.44 | 3.46 |
| LLM judge safe referral when needed % | 27.30 | n/a | 24.20 | 29.40 |

Baseline column: `results/baseline_official` (official CohereLabs/tiny-aya-earth, re-run in run two). Run 1 column is the earlier SFT (lr 2e-4, NaN dev loss) for reference.
Generation: greedy, max 256 new tokens, chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH; never trained on). MCQ: zero-shot, argmax next-token logit over A-D. AuditKIT `chrf` is char 6-gram F1, not chrF++. "Hit 256-token cap" = share of answers that did not stop on their own.

LLM judge: google/gemini-3-flash via Cline API (free promo credit), 94/100 rows of a fixed seeded sample (60 Walelign SRH + 40 AmHQA), three answers per call under shuffled blind labels. Raw counts: {"baseline": {"n": 94, "correctness_mean": 2.617, "fluency_mean": 3.989, "safe_referral_yes": 9, "safe_referral_no": 24, "safe_referral_NA": 61, "safe_referral_rate_when_needed": 0.273}, "A": {"n": 94, "correctness_mean": 2.862, "fluency_mean": 3.436, "safe_referral_yes": 8, "safe_referral_no": 25, "safe_referral_NA": 61, "safe_referral_rate_when_needed": 0.242}, "B": {"n": 94, "correctness_mean": 2.872, "fluency_mean": 3.457, "safe_referral_yes": 10, "safe_referral_no": 24, "safe_referral_NA": 60, "safe_referral_rate_when_needed": 0.294}}

## Training (identical settings for A and B)

- A: framework aligntune (TRL backend); model CohereLabs/tiny-aya-earth; hp {'epochs': 1, 'lr': 0.0001, 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 1024, 'bs': 2, 'grad_acc': 8, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; train rows kept 5034 (length guard dropped 2075); dev rows 254; steps 315; train loss 1.5070935219053239; final dev eval_loss 1.4368178844451904; non-finite events 0; wall 6328 s
- B: framework aligntune (TRL backend); model CohereLabs/tiny-aya-earth; hp {'epochs': 1, 'lr': 0.0001, 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 1024, 'bs': 2, 'grad_acc': 8, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; train rows kept 4806 (length guard dropped 0); dev rows 249; steps 301; train loss 1.5010688803916754; final dev eval_loss 1.4542382955551147; non-finite events 0; wall 5666 s

## Notes

- NaN dev loss (run one): 119/374 dev rows and 2,046/7,109 train rows (all ethionlp_general, long XL-Sum/MT prompts) had a prompt (Cohere preamble ~360 tokens + article) that already filled max_length 1024. TRL truncates from the right, so every assistant label was masked; an eval batch with zero supervised tokens gives 0/0 = NaN and poisons the mean. Train loss stayed finite because TRL normalises by tokens across the accumulated batch, but ~29% of train rows contributed nothing. Fix in run two: a length guard (drop rows with < 16 supervised tokens inside 1024) applied identically to A and B, LoRA params fp32 under fp16 autocast, lr 1e-4, max_grad_norm 1.0. Sanity kernel (50 steps): eval_loss 1.670 -> 1.653, no non-finite values.
- Model: official CohereLabs/tiny-aya-earth (HF access granted to 5LeaderW). The kernel assembles it from the official config/tokenizer files plus weight shards whose SHA-256 matches the official LFS oids, so no token sits in the kernel.
- Curation (CuratorKIT): train 7,109 -> 4806, dev 374 -> 249; dedup 37, broken text 23, English-heavy 53, off-topic 15, Q/A mismatch 36, EthioNLP downsample 2139 (kept 653), referral line added to 15 of 17 danger-sign answers. Details: data/curated/curation_report.md
