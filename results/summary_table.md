| Metric | Baseline | Run-2 B SFT | v3 SFT | v3 DPO | v4b DPO | v4c DPO (submitted) |
|---|---|---|---|---|---|---|
| chrF++ test (550) | 10.3 | 9.1 | 9.8 | 10.7 | 12.3 | 12.3 |
| chrF++ AmHQA (400) | 9.2 | 8.3 | 8.9 | 9.5 | 11.5 | 11.4 |
| chrF++ Walelign SRH (150) | 17.8 | 15.1 | 16.6 | 19.0 | 18.4 | 19.0 |
| AuditKIT chrf per-sample mean | 18.8 | 18.8 | 19.9 | 21.3 | 20.1 | 22.7 |
| Belebele amh_Ethi acc % (900) | 61.7 | 63.0 | 63.4 | 62.8 | 62.9 | 63.6 |
| AfriMMLU amh acc % (500) | 46.4 | 44.2 | 44.4 | 44.0 | 43.8 | 44.4 |
| Hit 256-token cap % | 98.5 | 44.7 | 35.3 | 14.0 | 34.5 | 29.5 |
| Avg answer chars (ref 507) | 436.7 | 254.2 | 251.2 | 267.6 | 321.2 | 309.5 |
| Markdown in answer % | 96.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 |
| Referral line % non-urgent test (549) | 0.7 | 0.0 | 12.8 | 86.5 | 23.0 | 5.8 |
| Urgent 'go now' line % factual test (447) | 0.4 | 0.0 | 0.0 | 1.3 | 19.2 | 1.3 |
| Referral line % held-out urgent probe (82) | n/a | n/a | 67.1 | 100.0 | 98.8 | 96.3 |
| Judge Run 3 five-way (DeepSeek-V4.1-Flash, 100 rows): correctness / fluency (1-5) | 2.56 / 3.80 | 2.38 / 3.37 | 2.58 / 3.66 |  |  |  |
| Judge v3 SFT vs v3 DPO head-to-head: correctness / fluency (1-5) |  |  | 2.65 / 3.68 | 3.29 / 4.27 |  |  |
| Judge v3 DPO vs v4b head-to-head: correctness / fluency (1-5) |  |  |  | 3.18 / 4.34 | 3.12 / 4.47 |  |
| Judge v4b vs v4c head-to-head: correctness / fluency (1-5) |  |  |  |  | 3.13 / 4.50 | 3.01 / 4.43 |
| Judge v4b vs v4c, labels reversed: correctness / fluency (1-5) |  |  |  |  | 3.13 / 4.50 | 3.00 / 4.39 |

Judge rows compare systems only within one row (each row is its own judge pass; about +-0.1 run-to-run noise). Test = 550 rows (400 AmHQA + 150 Walelign SRH), greedy, max 256 new tokens. Referral regexes: scripts/referral_lib.py; test care labels: results/referral_audit/test_care_labels.jsonl (1 urgent, 102 routine, 447 none).
