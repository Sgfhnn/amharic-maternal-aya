# Tiny Aya Earth: before vs after LoRA SFT (Amharic health)

| Metric | Before (base) | After (SFT) | Delta |
|---|---|---|---|
| Test chrF++ (corpus, sacrebleu) | 10.2 | 9.0 | -1.3 |
| Test chrF (corpus) | 11.7 | 10.2 | -1.6 |
| Test BLEU-13a (corpus) | 0.9 | 1.0 | +0.1 |
| Test spBLEU flores200 (corpus) | 5.1 | 3.7 | -1.5 |
| chrF++ AmHQA (400) | 9.1 | 8.2 | -1.0 |
| chrF++ Walelign SRH (150) | 17.4 | 15.4 | -2.0 |
| chrF++ maternal-topic rows (88) | 10.8 | 9.0 | -1.7 |
| AuditKIT chrf (per-sample mean) | 16.6 | 19.0 | +2.3 |
| AuditKIT bleu (per-sample mean) | 0.3 | 1.1 | +0.8 |
| AuditKIT rouge_l (per-sample mean) | 7.8 | 11.3 | +3.5 |
| AuditKIT f1_score (per-sample mean) | 8.4 | 11.1 | +2.7 |
| Belebele amh_Ethi acc % (900) | 61.7 | 61.0 | -0.7 |
| AfriMMLU amh acc % (500) | 46.4 | 42.8 | -3.6 |
| Avg answer length, chars (ref 507) | 441.1 | 247.0 | -194.1 |
| Stopped before 256-token cap % | 0.9 | 53.8 | +52.9 |
| Answers with Markdown % | 95.3 | 0.0 | -95.3 |
| Health-worker referral keyword % | 4.5 | 2.5 | -2.0 |
| Ethiopic script share % | 84.2 | 98.4 | +14.2 |

Setup: greedy decoding, max 256 new tokens, model chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH).
MCQ = zero-shot, argmax next-token logit over A-D. AuditKIT `chrf` is character 6-gram F1 (beta 1), not chrF++.
Model weights: 1-800-LLMs/tiny-aya-earth (identical LFS hashes to CohereLabs/tiny-aya-earth). Compute: free Kaggle T4 x2.

## Training
- Framework: aligntune (TRL backend)
- Hyper-parameters: {'epochs': 1, 'lr': 0.0002, 'r': 16, 'alpha': 32, 'dropout': 0.05, 'max_len': 1024, 'bs': 2, 'grad_acc': 8, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
- Result: {'training_time': 12287.773473739624, 'final_loss': 1.4671060958605133, 'model_path': '/kaggle/working/results/sft/ckpt', 'steps': 445, 'epochs': 0, 'metrics': {'train_runtime': 12287.1507, 'train_samples_per_second': 0.579, 'train_steps_per_second': 0.036, 'total_flos': 1.0381235581813555e+17, 'train_loss': 1.4671060958605133, 'epoch': 1.0}, 'task_type': 'sft'}
- Final dev eval: {'eval_loss': nan, 'eval_runtime': 217.3171, 'eval_samples_per_second': 1.721, 'eval_steps_per_second': 0.86, 'eval_entropy': 1.341535956145608, 'eval_mean_token_accuracy': 0.5956611379901355, 'eval_num_tokens': 5098521.0, 'epoch': 1.0}
- Train wall time (s): 12629
