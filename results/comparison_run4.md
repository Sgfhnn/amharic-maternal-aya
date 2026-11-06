# Run four: over-referral fix (DPO v4) vs v3 DPO vs v3 SFT (Tiny Aya Earth, Amharic maternal/SRH health)

| Metric | v3 SFT | v3 DPO | v4 DPO | v4b DPO | v4c DPO |
|---|---|---|---|---|---|
| chrF++ test (corpus) | 9.8 | 10.7 | 10.2 | 12.3 | 12.3 |
| chrF++ AmHQA (400) | 8.9 | 9.5 | 9.3 | 11.5 | 11.4 |
| chrF++ Walelign SRH (150) | 16.6 | 19.0 | 16.8 | 18.4 | 19.0 |
| AuditKIT chrf (per-sample mean) | 19.9 | 21.3 | 20.1 | 20.1 | 22.7 |
| Belebele amh_Ethi acc % (900) | 63.4 | 62.8 | 64.0 | 62.9 | 63.6 |
| AfriMMLU amh acc % (500) | 44.4 | 44.0 | 43.8 | 43.8 | 44.4 |
| Hit 256-token cap % | 35.3 | 14.0 | 40.7 | 34.5 | 29.5 |
| Avg answer chars (ref 507) | 251.2 | 267.6 | 263.2 | 321.2 | 309.5 |
| Referral line % (all 550 test) | 12.7 | 86.5 | 4.7 | 23.1 | 6.0 |
| Referral line % on non-urgent test questions (549) | 12.8 | 86.5 | 4.7 | 23.0 | 5.8 |
| Referral line % on factual test questions (447) | 10.1 | 84.1 | 2.5 | 21.3 | 4.0 |
| Referral line % on routine test questions (102) | 24.5 | 97.1 | 14.7 | 30.4 | 13.7 |
| Referral line % on urgent test questions (1) | 0.0 | 100.0 | 0.0 | 100.0 | 100.0 |
| Referral line % on held-out urgent probe (82) | 67.1 | 100.0 | 91.5 | 98.8 | 96.3 |
| Generic boilerplate % (non-urgent test) | 12.6 | 85.8 | 2.7 | 2.4 | 3.5 |
| False urgency: urgent 'go now' line % on factual test questions | 0.0 | 1.3 | 0.9 | 19.2 | 1.3 |
| Urgent 'go now' line % on non-urgent test questions | 0.2 | 1.6 | 2.0 | 20.6 | 2.2 |
| Urgent-worded referral % (held-out urgent probe) | 51.2 | 86.6 | 91.5 | 98.8 | 92.7 |
| Judge head-to-head v3 DPO vs v4 (DeepSeek-V4.1-Flash, 100 rows) correctness 1-5 | n/a | 3.28 | 2.68 | n/a | n/a |
| Judge head-to-head v3 DPO vs v4b (DeepSeek-V4.1-Flash, 100 rows) correctness 1-5 | n/a | 3.18 | n/a | 3.12 | n/a |
| Judge head-to-head v4b vs v4c (DeepSeek-V4.1-Flash, same 100 rows) correctness 1-5 | n/a | n/a | n/a | 3.13 | 3.01 |
| Judge v4b vs v4c, repeat with labels reversed (position-bias check) correctness 1-5 | n/a | n/a | n/a | 3.13 | 3.00 |
| Judge run-3 head-to-head v3 SFT vs v3 DPO (reference) correctness 1-5 | 2.65 | 3.29 | n/a | n/a | n/a |
| Judge head-to-head v3 DPO vs v4 (DeepSeek-V4.1-Flash, 100 rows) Amharic fluency 1-5 | n/a | 4.31 | 3.76 | n/a | n/a |
| Judge head-to-head v3 DPO vs v4b (DeepSeek-V4.1-Flash, 100 rows) Amharic fluency 1-5 | n/a | 4.34 | n/a | 4.47 | n/a |
| Judge head-to-head v4b vs v4c (DeepSeek-V4.1-Flash, same 100 rows) Amharic fluency 1-5 | n/a | n/a | n/a | 4.50 | 4.43 |
| Judge v4b vs v4c, repeat with labels reversed (position-bias check) Amharic fluency 1-5 | n/a | n/a | n/a | 4.50 | 4.39 |
| Judge run-3 head-to-head v3 SFT vs v3 DPO (reference) Amharic fluency 1-5 | 3.68 | 4.27 | n/a | n/a | n/a |
| Judge head-to-head v3 DPO vs v4 (DeepSeek-V4.1-Flash, 100 rows) safe referral when needed % | n/a | 100.00 | 19.30 | n/a | n/a |
| Judge head-to-head v3 DPO vs v4b (DeepSeek-V4.1-Flash, 100 rows) safe referral when needed % | n/a | 96.50 | n/a | 62.30 | n/a |
| Judge head-to-head v4b vs v4c (DeepSeek-V4.1-Flash, same 100 rows) safe referral when needed % | n/a | n/a | n/a | 76.90 | 55.10 |
| Judge v4b vs v4c, repeat with labels reversed (position-bias check) safe referral when needed % | n/a | n/a | n/a | 76.90 | 48.00 |
| Judge run-3 head-to-head v3 SFT vs v3 DPO (reference) safe referral when needed % | 26.30 | 100.00 | n/a | n/a | n/a |

Generation: greedy, max 256 new tokens, chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH; never trained on). MCQ zero-shot argmax over A-D. Referral metrics: scripts/referral_audit.py + scripts/referral_lib.py (regex, no LLM). 'Referral line' = a sentence that names a facility / health worker and tells the reader to go / consult / get care; 'generic boilerplate' = such a sentence with no condition (ካለ / ከሆነ / ከታዩ ...) and no urgency word. Test care labels (DeepSeek-V4.1-Flash, same triage prompt as the run-three train labels): {'urgent': 1, 'routine': 102, 'none': 447}. The test set has only one urgent question, so urgent-question referral is measured on a held-out probe of 82 new Amharic danger-sign questions (data/probe_urgent_v4.jsonl; generated with the run-three synthetic prompt, de-duplicated against all train/test questions, kept only if the same triage calls them urgent; never trained on).

Judge (v4): deepseek-ai/DeepSeek-V4.1-Flash:deepinfra, 100/100 rows, same fixed seeded sample, only v3 DPO and v4 in each call under blind shuffled labels. Raw: {"v3dpo": {"n": 100, "correctness_mean": 3.28, "fluency_mean": 4.31, "safe_referral_yes": 84, "safe_referral_no": 0, "safe_referral_NA": 16, "safe_referral_rate_when_needed": 1.0}, "v4dpo": {"n": 100, "correctness_mean": 2.68, "fluency_mean": 3.76, "safe_referral_yes": 11, "safe_referral_no": 46, "safe_referral_NA": 43, "safe_referral_rate_when_needed": 0.193}}.
Judge (v4b): deepseek-ai/DeepSeek-V4.1-Flash:deepinfra, 100/100 rows, same fixed seeded sample, only v3 DPO and v4b in each call under blind shuffled labels. Raw: {"v3dpo": {"n": 100, "correctness_mean": 3.18, "fluency_mean": 4.34, "safe_referral_yes": 83, "safe_referral_no": 3, "safe_referral_NA": 14, "safe_referral_rate_when_needed": 0.965}, "v4bdpo": {"n": 100, "correctness_mean": 3.12, "fluency_mean": 4.47, "safe_referral_yes": 38, "safe_referral_no": 23, "safe_referral_NA": 39, "safe_referral_rate_when_needed": 0.623}}.
Judge (v4c): deepseek-ai/DeepSeek-V4.1-Flash:deepinfra, 100/100 rows, same fixed seeded sample, only v4b and v4c in each call under blind shuffled labels. Raw: {"v4bdpo": {"n": 100, "correctness_mean": 3.13, "fluency_mean": 4.5, "safe_referral_yes": 40, "safe_referral_no": 12, "safe_referral_NA": 48, "safe_referral_rate_when_needed": 0.769}, "v4cdpo": {"n": 100, "correctness_mean": 3.01, "fluency_mean": 4.43, "safe_referral_yes": 27, "safe_referral_no": 22, "safe_referral_NA": 51, "safe_referral_rate_when_needed": 0.551}}.
Judge caveat: its 'safe referral when needed' counts routine questions where it thinks a referral is advisable, so it partly rewards the blanket line v4 is meant to remove; urgent safety is read from the held-out urgent probe instead.

## Data (CuratorKIT, data/curated_v4/curation_report.md)

- DPO v4 pairs: 425 train + 25 dev. By care: {'routine': 225, 'urgent': 225}. By kind: {'routine_rewrite': 113, 'minus_referral': 182, 'factual_answer': 112, 'original_mt_answer': 43}.
- Urgent (kept from v3, unchanged): chosen has the urgent referral, rejected lacks it (or is the original MT answer). Routine counter-pairs (new): chosen = fluent answer without the generic facility line; rejected = same answer + a boilerplate sentence v3 DPO actually emits. v3's 578 routine 'referral good' pairs dropped. No LLM calls.
- SFT targets with boilerplate stripped (data/curated_v4/train.jsonl, record only; SFT not re-run): {'stripped_routine': 307, 'stripped_none': 3}.
- DPO v4b pairs (attempt b, data/curated_v4b/curation_report.md): 915 train + 53 dev. By kind/care: {'routine_rewrite/routine': 113, 'minus_referral/urgent': 182, 'factual_answer/routine': 112, 'original_mt_answer/urgent': 43, 'fluency_vs_mt/routine': 278, 'fluency_vs_mt/urgent': 43, 'anti_repetition/urgent': 90, 'anti_repetition/routine': 107}. v4 pairs + v3's 323 'fluent rewrite vs original MT answer' pairs (routine chosen with the boilerplate stripped) + anti-repetition pairs (rejected = same answer + its last sentence looped 3x). Referral-signal pairs stay 225 urgent / 225 routine.
- DPO v4c pairs (attempt c, data/curated_v4c/curation_report.md; scripts/curate_v4c.py, no LLM): 1086 train + 62 dev. By kind/care/need: {'minus_referral/urgent/urgent': 182, 'anti_repetition/routine/none': 51, 'false_urgency/routine/routine': 70, 'false_urgency/routine/none': 200, 'fluency_vs_mt/routine/routine': 278, 'anti_repetition/routine/routine': 56, 'factual_answer/routine/none': 112, 'original_mt_answer/urgent/urgent': 43, 'routine_rewrite/routine/routine': 113, 'fluency_vs_mt/urgent/urgent': 43}. v4b pairs minus the 90 urgent anti-repetition pairs, plus 270 false-urgency counter-pairs (factual / routine question from the curated SFT pool; chosen = clean gold answer; rejected = same answer + one of 58 urgent 'go now' lines v4b emitted on non-urgent test questions). Chosen answers ending in an urgent line: 268 (v4b about 358).

## Training

- v4 DPO: aligntune (TRL backend) DPO, low-mem attempt 1; from the merged v3 SFT model (not on top of v3 DPO); hp {'epochs': 1, 'lr': 5e-05, 'beta': 0.1, 'loss': 'sigmoid', 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 768, 'max_prompt_len': None, 'truncation': 'keep_start', 'bs': 1, 'grad_acc': 16, 'precompute_ref_log_probs': True, 'gradient_checkpointing': True, 'max_memory': {'0': 3670016000, '1': 13958643712}, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; data {"train": {"in": 425, "kept": 425, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 1, "prompt_tokens_max": 440}, "val": {"in": 25, "kept": 25, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 1, "prompt_tokens_max": 414}}; result {'training_time': 333.9945, 'final_loss': 0.41435042961880014, 'model_path': '/kaggle/working/results/dpo_v4/ckpt', 'total_steps': 0}; final eval {"eval_loss": 0.2061, "eval_rewards/chosen": 0.5897, "eval_rewards/rejected": -1.2956, "eval_rewards/accuracies": 1.0, "eval_rewards/margins": 1.8853}; non-finite events 0; peak GPU GiB {'0': 7.86, '1': 4.02}; wall 531 s.
- v4b DPO: aligntune (TRL backend) DPO, low-mem attempt 1; same settings; data {"train": {"in": 915, "kept": 915, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 36, "prompt_tokens_max": 440}, "val": {"in": 53, "kept": 53, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 10, "prompt_tokens_max": 414}}; result {'training_time': 882.9637, 'final_loss': 0.2701920598488429, 'model_path': '/kaggle/working/results/dpo_v4b/ckpt', 'total_steps': 0}; final eval {"eval_loss": 0.1043, "eval_rewards/chosen": 0.8225, "eval_rewards/rejected": -4.9164, "eval_rewards/accuracies": 0.9623, "eval_rewards/margins": 5.7389}; non-finite events 0; peak GPU GiB {'0': 7.86, '1': 4.02}; wall 1284 s.
- v4c DPO: aligntune (TRL backend) DPO, low-mem attempt 1; same settings; data {"train": {"in": 1086, "kept": 1086, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 30, "prompt_tokens_max": 440}, "val": {"in": 62, "kept": 62, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 9, "prompt_tokens_max": 423}}; result {'training_time': 963.434, 'final_loss': 0.29953267096596603, 'model_path': '/kaggle/working/results/dpo_v4c/ckpt', 'total_steps': 0}; final eval {"eval_loss": 0.125, "eval_rewards/chosen": 0.223, "eval_rewards/rejected": -4.6314, "eval_rewards/accuracies": 0.9677, "eval_rewards/margins": 4.8544}; non-finite events 0; peak GPU GiB {'0': 7.86, '1': 4.02}; wall 1388 s.

## Decision: v4 / v4b DPO vs v3 DPO

Rule (Anu, run four): submit v4 over v3 DPO only if the routine-question referral rate drops substantially (target < 30%), urgent-question safe referral stays >= 85%, judge correctness and fluency drop no more than 0.1, and Belebele / AfriMMLU stay within about 1 point.

**v4 DPO**:
- non-urgent referral line < 30% (and well below v3 DPO): yes
- urgent referral >= 85% (held-out urgent probe): yes
- judge correctness drop <= 0.1: NO
- judge fluency drop <= 0.1: NO
- Belebele not down by more than ~1 pt: yes
- AfriMMLU not down by more than ~1 pt: yes
- passes: no (judge correctness drop <= 0.1, judge fluency drop <= 0.1)

**v4b DPO**:
- non-urgent referral line < 30% (and well below v3 DPO): yes
- urgent referral >= 85% (held-out urgent probe): yes
- judge correctness drop <= 0.1: yes
- judge fluency drop <= 0.1: yes
- Belebele not down by more than ~1 pt: yes
- AfriMMLU not down by more than ~1 pt: yes
- passes: yes

Result: **v4b DPO** for submission.

## Decision: v4c DPO vs v4b DPO (false-urgency fix)

Rule (Anu, run four c): pick v4c over v4b if false urgency on factual questions drops below about 5%, the urgent probe stays >= 90%, non-urgent referral stays <= 30%, judge correctness and fluency (head-to-head v4b vs v4c) drop by no more than 0.1, and Belebele / AfriMMLU stay within about 1 point.

**v4c DPO**:
- false urgency on factual questions < ~5%: yes
- urgent probe >= 90%: yes
- non-urgent referral line <= 30%: yes
- judge correctness drop vs v4b <= 0.1: NO
- judge fluency drop vs v4b <= 0.1: yes
- Belebele within ~1 pt of v4b: yes
- AfriMMLU within ~1 pt of v4b: yes
- passes: no (judge correctness drop vs v4b <= 0.1)

Final result: **v4b DPO** for submission.

## Caveats on the v4b pick (read before the write-up)

- v4b swapped the generic line for an urgent one on some factual questions. Generic boilerplate on non-urgent test answers fell 85.8% -> 2.4%, but 20.6% of non-urgent answers (19.2% of factual ones) now end in an urgent "go to a health facility now" line, versus 1.6% for v3 DPO. Example: "What are the treatments for Bell's palsy?" ends in "go to the nearest health facility now and get examined". Likely cause: in the v4b mix, urgent-ending chosen answers appear in about 358 pairs (225 urgent referral pairs + 43 urgent fluency pairs + 90 urgent anti-repetition pairs). v4 (425 pairs, no fluency or anti-repetition pairs) did not have this problem (0.9% on factual answers), but it fell back to v3 SFT quality.
- On the strict "routine" slice (the 102 test questions labelled routine, where a referral is often reasonable), v4b's referral-line rate is 30.4%, just over the 30% target. The rule was checked on all non-urgent questions (routine + factual, 549): 23.0%.
- Judge noise: v3 DPO scored 3.28 / 4.31 in one head-to-head and 3.18 / 4.34 in the other (same rows, same judge), so roughly +-0.1 is noise. v4b's correctness gap (-0.06) is inside that noise.
- 34.5% of v4b answers still hit the 256-token cap (v3 DPO 14.0%). Anti-repetition pairs helped only a little (v4 was at 40.7%). v4b answers are longer (321 vs 268 chars), which helps corpus chrF++ (12.3, the best of any run).
- v4c (run 2026-11-06 12:17-13:05 EAT) is that fix: v4b pairs - 90 urgent anti-repetition pairs + 270 false-urgency counter-pairs (scripts/curate_v4c.py, no LLM). See the v4c decision above.

## Caveats on v4c (attempt c, false-urgency fix)

- v4c fixes the false urgency: urgent "go now" line on factual test answers 19.2% -> 1.3% (v3 DPO level), on all non-urgent 20.6% -> 2.2%; routine-slice referral 30.4% -> 13.7%; cap hits 34.5% -> 29.5%; Belebele +0.7, AfriMMLU +0.6, chrF++ unchanged (12.3).
- It misses the rule on one condition: judge correctness 3.13 -> 3.01 (-0.12, limit 0.1). A second pass on the same 100 rows with the label order reversed gave 3.13 -> 3.00 and fluency 4.50 -> 4.39, so the drop is not position bias. It is spread across ordinary factual rows, not just the rows where v4b had an urgent line, so it reads as a small real content shift (inside the run-to-run noise seen earlier, but consistent across both passes).
- Urgent probe: referral line 98.8% -> 96.3% (3 of 82 probe answers lost the line), urgent-worded 98.8% -> 92.7%. Still above the 90% bar.
- The judge's "safe referral when needed" falls 76.9% -> 55.1% (48.0% reversed): it wants a referral on some routine questions that v4c now answers without one. Urgent safety is read from the probe instead, but it is a trade-off to state in the write-up.
