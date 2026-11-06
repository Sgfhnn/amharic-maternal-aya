# Run three: baseline vs run-2 raw vs run-2 curated vs v3 SFT vs v3 DPO (Tiny Aya Earth, Amharic maternal/SRH health)

| Metric | Baseline | Run-2 A raw-SFT | Run-2 B curated-SFT | v3 SFT | v3 DPO |
|---|---|---|---|---|---|
| chrF++ test (corpus) | 10.3 | 8.9 | 9.1 | 9.8 | 10.7 |
| BLEU-13a test (corpus) | 0.9 | 0.9 | 1.4 | 1.2 | 1.4 |
| spBLEU flores200 test (corpus) | 5.2 | 3.6 | 3.9 | 3.9 | 4.0 |
| chrF++ AmHQA (400) | 9.2 | 8.0 | 8.3 | 8.9 | 9.5 |
| chrF++ Walelign SRH (150) | 17.8 | 15.6 | 15.1 | 16.6 | 19.0 |
| chrF++ maternal-topic rows (88) | 10.8 | 9.3 | 8.7 | 10.1 | 10.6 |
| AuditKIT chrf (per-sample mean) | 18.8 | 18.8 | 18.8 | 19.9 | 21.3 |
| Belebele amh_Ethi acc % (900) | 61.7 | 63.7 | 63.0 | 63.4 | 62.8 |
| AfriMMLU amh acc % (500) | 46.4 | 45.0 | 44.2 | 44.4 | 44.0 |
| Answers with Markdown % | 96.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Hit 256-token cap % | 98.5 | 44.2 | 44.7 | 35.3 | 14.0 |
| Referral keyword % (run-2 list, all test) | 4.2 | 2.2 | 1.6 | 16.5 | 85.3 |
| Avg answer chars (ref 507) | 436.7 | 243.5 | 254.2 | 251.2 | 267.6 |
| Ethiopic script share % | 84.8 | 98.3 | 98.4 | 98.8 | 99.3 |
| Referral keyword % (wide v3 list, all test) | 4.4 | 2.4 | 2.0 | 17.3 | 87.5 |
| LLM judge run 3 (DeepSeek-V4.1-Flash) correctness 1-5 | 2.56 | 2.46 | 2.38 | 2.58 | n/a |
| LLM judge run 3 (DeepSeek-V4.1-Flash) Amharic fluency 1-5 | 3.80 | 3.39 | 3.37 | 3.66 | n/a |
| LLM judge run 3 (DeepSeek-V4.1-Flash) safe referral when needed % | 24.00 | 25.00 | 30.40 | 59.30 | n/a |
| LLM judge run 2 (gemini-3-flash) correctness 1-5 | 2.62 | 2.86 | 2.87 | n/a | n/a |
| LLM judge run 2 (gemini-3-flash) Amharic fluency 1-5 | 3.99 | 3.44 | 3.46 | n/a | n/a |
| LLM judge run 2 (gemini-3-flash) safe referral when needed % | 27.30 | 24.20 | 29.40 | n/a | n/a |
| LLM judge run 3 head-to-head v3 SFT vs v3 DPO (DeepSeek-V4.1-Flash) correctness 1-5 | n/a | n/a | n/a | 2.65 | 3.29 |
| LLM judge run 3 head-to-head v3 SFT vs v3 DPO (DeepSeek-V4.1-Flash) Amharic fluency 1-5 | n/a | n/a | n/a | 3.68 | 4.27 |
| LLM judge run 3 head-to-head v3 SFT vs v3 DPO (DeepSeek-V4.1-Flash) safe referral when needed % | n/a | n/a | n/a | 26.30 | 100.00 |

Generation: greedy, max 256 new tokens, chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH; never trained on). MCQ: zero-shot, argmax next-token logit over A-D. AuditKIT `chrf` is char 6-gram F1, not chrF++. "Hit 256-token cap" = share of answers that did not stop on their own.

Run-three judge: deepseek-ai/DeepSeek-V4.1-Flash:deepinfra via HF Inference Providers, 100/100 rows of the same fixed seeded sample as run two (60 Walelign SRH + 40 AmHQA), all five answers in one call per row under shuffled blind labels. It is a different model from the Gemma model that rewrote the v3 training answers, to avoid self-preference. Raw counts: {"baseline": {"n": 100, "correctness_mean": 2.56, "fluency_mean": 3.8, "safe_referral_yes": 6, "safe_referral_no": 19, "safe_referral_NA": 75, "safe_referral_rate_when_needed": 0.24}, "A": {"n": 100, "correctness_mean": 2.46, "fluency_mean": 3.39, "safe_referral_yes": 6, "safe_referral_no": 18, "safe_referral_NA": 76, "safe_referral_rate_when_needed": 0.25}, "B": {"n": 100, "correctness_mean": 2.38, "fluency_mean": 3.37, "safe_referral_yes": 7, "safe_referral_no": 16, "safe_referral_NA": 77, "safe_referral_rate_when_needed": 0.304}, "v3sft": {"n": 100, "correctness_mean": 2.58, "fluency_mean": 3.66, "safe_referral_yes": 16, "safe_referral_no": 11, "safe_referral_NA": 73, "safe_referral_rate_when_needed": 0.593}}

Head-to-head judge (DPO retry): same judge model and same 100-row sample, but each call shows only the v3 SFT and v3 DPO answers (blind shuffled labels), 100/100 rows. The five-system row above was judged before DPO existed, so the DPO decision uses this head-to-head row; v3 SFT is re-scored in it so both sit on one scale. Raw counts: {"v3sft": {"n": 100, "correctness_mean": 2.65, "fluency_mean": 3.68, "safe_referral_yes": 15, "safe_referral_no": 42, "safe_referral_NA": 43, "safe_referral_rate_when_needed": 0.263}, "v3dpo": {"n": 100, "correctness_mean": 3.29, "fluency_mean": 4.27, "safe_referral_yes": 84, "safe_referral_no": 0, "safe_referral_NA": 16, "safe_referral_rate_when_needed": 1.0}}

Run-two judge (reference only, three systems): google/gemini-3-flash via Cline, 94 rows. Its scale is not comparable to the run-three judge; compare systems within one judge row.

## Training

- Run-2 B curated-SFT: framework aligntune (TRL backend); hp {'epochs': 1, 'lr': 0.0001, 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 1024, 'bs': 2, 'grad_acc': 8, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; data {"train": {"file": "chat_train_curated.jsonl", "in": 4806, "kept": 4806, "dropped_no_supervised_tokens": 0, "dropped_by_source": {}, "empty_assistant": 0, "answer_truncated_kept": 258}, "val": {"file": "chat_dev_curated.jsonl", "in": 249, "kept": 249, "dropped_no_supervised_tokens": 0, "dropped_by_s; result {'training_time': 5467.324557542801, 'final_loss': 1.5010688803916754, 'model_path': '/kaggle/working/results/sft_curated/ckpt', 'steps': 301, 'epochs': 0, 'metrics': {'train_runtime': 5466.6667, 'tra; final eval {"eval_loss": 1.4542}; non-finite events 0; wall 5666 s
- v3 SFT: framework aligntune (TRL backend); hp {'epochs': 1, 'lr': 0.0001, 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 1024, 'bs': 2, 'grad_acc': 8, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; data {"train": {"file": "chat_train_v3.jsonl", "in": 2691, "kept": 2691, "dropped_no_supervised_tokens": 0, "dropped_by_source": {}, "empty_assistant": 0, "answer_truncated_kept": 22}, "val": {"file": "chat_dev_v3.jsonl", "in": 249, "kept": 249, "dropped_no_supervised_tokens": 0, "dropped_by_source": {},; result {'training_time': 3136.910120010376, 'final_loss': 1.2283347541763938, 'model_path': '/kaggle/working/results/sft_v3/ckpt', 'steps': 169, 'epochs': 0, 'metrics': {'train_runtime': 3136.2395, 'train_sa; final eval {"eval_loss": 1.4607}; non-finite events 0; wall 3329 s
- v3 DPO first try (kernel amh-tinyaya-r3, OOM): framework FAILED; hp {'epochs': 1, 'lr': 5e-05, 'beta': 0.1, 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 1024, 'max_prompt_len': 384, 'bs': 2, 'grad_acc': 8, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; data {"train": 763, "val": 40}; result None; final eval {}; non-finite events 0; wall 245 s; attempts/errors ["in use. Of the allocated memory 11.22 GiB is allocated by PyTorch, and 54.15 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)\n", "in use. Of the allocated memory 12.57 GiB is allocated by PyTorch, and 61.47 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)\n"]
- v3 DPO retry (kernel amh-tinyaya-r3-dpo, low-memory): framework aligntune (TRL backend) DPO, low-mem attempt 1; hp {'epochs': 1, 'lr': 5e-05, 'beta': 0.1, 'loss': 'sigmoid', 'max_grad_norm': 1.0, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 768, 'max_prompt_len': None, 'truncation': 'keep_start', 'bs': 1, 'grad_acc': 16, 'precompute_ref_log_probs': True, 'gradient_checkpointing': True, 'max_memory': {'0': 3670016000, '1': 13958643712}, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}; data {"train": {"in": 763, "kept": 763, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 21, "prompt_tokens_max": 440}, "val": {"in": 40, "kept": 40, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 5, "prompt_tokens_max": 404}}; result {'training_time': 729.5016, 'final_loss': 0.2182405854885777, 'model_path': '/kaggle/working/results/dpo_v3/ckpt', 'total_steps': 0}; final eval {"eval_loss": 0.0368, "eval_rewards/accuracies": 1.0, "eval_rewards/margins": 5.3273}; non-finite events 0; wall 1070 s; used {'max_len': 768, 'precompute_ref_log_probs': True}; final DPOConfig {"per_device_train_batch_size": 1, "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 16, "gradient_checkpointing": true, "gradient_checkpointing_kwargs": {"use_reentrant": false}, "max_length": 768, "truncation_mode": "keep_start", "precompute_ref_log_probs": true, "fp16": true, "bf16": false, "beta": 0.1, "loss_type": ["sigmoid"], "learning_rate": 5e-05, "num_train_epochs": 1, "max_steps": -1, "lr_scheduler_type": "linear", "warmup_steps": 0.1, "max_grad_norm": 1.0}; peak GPU GiB {'0': 7.86, '1': 4.02}; device map {'0': 16, '1': 24}

## Pipeline

- v3 SFT: run-2 curated adapter merged into the official CohereLabs/tiny-aya-earth, then a fresh LoRA (AlignTune TRL backend, r16, alpha32, lr 1e-4, clip 1.0, max_len 1024, length guard, 1 epoch) on data/curated_v3 (2691 rows). Model = official + run-2 B adapter + v3 adapter (stacked).
- v3 DPO: v3 SFT merged, then a fresh LoRA trained with AlignTune DPO (TRL backend, beta 0.1, lr 5e-5, 1 epoch, sigmoid loss); reference = the v3 SFT model (initial zero-delta LoRA copy, i.e. adapter effectively off). Pairs: {'pairs_train': 763, 'pairs_dev': 40, 'by_kind': {'minus_referral': 480, 'original_mt_answer': 323}, 'by_care': {'routine': 578, 'urgent': 225}}. First try OOM'd on a T4 (bs 2, max_len 1024). Retry (scripts/train_dpo_lowmem.py): bs 1 x grad-acc 16, gradient checkpointing, reference log-probs precomputed once (no in-step reference forward, no second model copy), max_len 768 keep_start (512 would make 453/763 pairs identical because the stock chat template's system preamble makes prompts 376-440 tokens; no prompt cap for the same reason, and TRL 1.7.1 has no max_prompt_length), fp16 base + fp32 LoRA, model split over both T4s. Two TRL patches, still run through AlignTune: chunked entropy metric, and an in-memory replacement for DPOTrainer._precompute_ref_logps (the stock one fails on AlignTune's in-memory datasets).
- Curation: data/curated_v3/curation_report.md. Danger-sign matcher: {"regex_matches": 25, "regex_matches_llm_confirmed_urgent": 13, "llm_urgent_total": 70, "llm_routine_total": 1341}.

## Decision: v3 DPO vs v3 SFT

Rule: DPO replaces SFT only if safe referral rises, fluency drops no more than 0.05, correctness does not drop, and Belebele/AfriMMLU stay within about 1 point (judge numbers from the head-to-head row).

- safe referral rises: yes
- fluency drop <= 0.05: yes
- correctness does not drop: yes
- Belebele within ~1 pt: yes
- AfriMMLU within ~1 pt: yes
- Belebele 63.4 -> 62.8; AfriMMLU 44.4 -> 44.0

Result: **v3 DPO** for submission.

Caveat: v3 DPO appends a referral line to 87% of all test answers, mostly one near-fixed sentence ("for a full check-up and correct treatment, go to a health facility and consult a health professional"), including on definitional questions. The judge's safe-referral score cannot penalise over-referral, so 100% partly reflects blanket referral. Conservative for a public health assistant, but say so in the write-up.
