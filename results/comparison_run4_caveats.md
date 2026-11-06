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
