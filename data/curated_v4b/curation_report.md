# Curated DPO v4b (run four, attempt b): CuratorKIT report

Run four, attempt b: CuratorKIT DPO v4b data. No LLM calls.

Why: v4 (data/curated_v4) removed the boilerplate (non-urgent referral line 86.5% -> 4.7%) but the judge scored it
2.68 / 3.76 vs v3 DPO 3.28 / 4.31, and 40.7% of its answers hit the 256-token cap in repetition loops (v3 SFT 35.3%,
v3 DPO 14.0%). v4 dropped v3's 280 routine 'fluent rewrite vs original MT answer' pairs (their chosen ended in the
boilerplate) and has only 27 optimiser steps, so it lost v3 DPO's fluency / stopping signal.

v4b = v4 pairs (225 urgent + 225 routine counter-pairs, unchanged)
    + fluency pairs: all 323 v3 'original_mt_answer' pairs; for the 280 routine ones the chosen rewrite has the generic
      boilerplate stripped (BoilerplateStripper), rejected = the original MT answer with any boilerplate stripped too,
      so the pair no longer carries a referral signal; the 43 urgent ones are unchanged (chosen keeps the urgent line)
    + anti-repetition pairs: for a seeded half of the urgent and of the routine chosen answers, rejected = the same
      answer + its last body sentence repeated 3 times (the loop failure mode), so stopping after the answer is preferred.
Referral-signal pairs stay 50/50 urgent / routine; fluency and anti-repetition pairs are split the same way as far as
the data allows. Usage: python scripts/curate_v4b.py

```
{
 "created": "2026-11-06T08:13:18.472034+00:00",
 "pairs_train": 915,
 "pairs_dev": 53,
 "by_kind": {
  "routine_rewrite": 113,
  "minus_referral": 182,
  "factual_answer": 112,
  "original_mt_answer": 43,
  "fluency_vs_mt": 321,
  "anti_repetition": 197
 },
 "by_kind_care": {
  "routine_rewrite/routine": 113,
  "minus_referral/urgent": 182,
  "factual_answer/routine": 112,
  "original_mt_answer/urgent": 43,
  "fluency_vs_mt/routine": 278,
  "fluency_vs_mt/urgent": 43,
  "anti_repetition/urgent": 90,
  "anti_repetition/routine": 107
 },
 "referral_signal_pairs_by_care": {
  "routine": 225,
  "urgent": 225
 },
 "notes": {
  "fluency_routine_chosen_stripped": 280,
  "fluency_routine_rejected_stripped": 3,
  "fluency_skipped_identical": 1,
  "fluency_routine_skipped_ref_asym": 1,
  "anti_skipped_long": 6
 }
}
```
