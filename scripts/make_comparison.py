"""Write results/comparison.md from results/baseline/metrics.json and results/sft/eval/metrics.json."""
import json, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else "results"
b = json.load(open(f"{R}/baseline/metrics.json"))
a = json.load(open(f"{R}/sft/eval/metrics.json")) if os.path.exists(f"{R}/sft/eval/metrics.json") else None
ti = json.load(open(f"{R}/sft/train_info.json")) if os.path.exists(f"{R}/sft/train_info.json") else {}


def g(m, path):
    cur = m
    for p in path:
        if cur is None or p not in cur:
            return None
        cur = cur[p]
    return cur


rows = [
    ("Test chrF++ (corpus, sacrebleu)", ["overall", "chrf_pp"], 1, ""),
    ("Test chrF (corpus)", ["overall", "chrf"], 1, ""),
    ("Test BLEU-13a (corpus)", ["overall", "bleu_13a"], 1, ""),
    ("Test spBLEU flores200 (corpus)", ["overall", "spbleu_flores200"], 1, ""),
    ("chrF++ AmHQA (400)", ["by_source", "amhqa_mendeley", "chrf_pp"], 1, ""),
    ("chrF++ Walelign SRH (150)", ["by_source", "walelign_srh", "chrf_pp"], 1, ""),
    ("chrF++ maternal-topic rows (88)", ["by_topic", "maternal", "chrf_pp"], 1, ""),
    ("AuditKIT chrf (per-sample mean)", ["auditkit_per_sample_mean", "headline", "chrf"], 100, ""),
    ("AuditKIT bleu (per-sample mean)", ["auditkit_per_sample_mean", "headline", "bleu"], 100, ""),
    ("AuditKIT rouge_l (per-sample mean)", ["auditkit_per_sample_mean", "headline", "rouge_l"], 100, ""),
    ("AuditKIT f1_score (per-sample mean)", ["auditkit_per_sample_mean", "headline", "f1_score"], 100, ""),
    ("Belebele amh_Ethi acc % (900)", ["mcq", "belebele_amh_Ethi", "accuracy"], 100, ""),
    ("AfriMMLU amh acc % (500)", ["mcq", "afrimmlu_amh", "accuracy"], 100, ""),
    ("Avg answer length, chars (ref 507)", ["diagnostics", "avg_pred_chars"], 1, ""),
    ("Stopped before 256-token cap %", ["diagnostics", "stopped_before_max_tokens"], 100, ""),
    ("Answers with Markdown %", ["diagnostics", "markdown_rate_pred"], 100, ""),
    ("Health-worker referral keyword %", ["diagnostics", "referral_keyword_rate_pred"], 100, ""),
    ("Ethiopic script share %", ["diagnostics", "avg_ethiopic_share"], 100, ""),
]
L = ["# Tiny Aya Earth: before vs after LoRA SFT (Amharic health)", ""]
L.append("| Metric | Before (base) | After (SFT) | Delta |")
L.append("|---|---|---|---|")
for name, path, mul, _ in rows:
    x, y = g(b, path), g(a, path) if a else None
    fx = f"{x * mul:.1f}" if isinstance(x, (int, float)) else "n/a"
    fy = f"{y * mul:.1f}" if isinstance(y, (int, float)) else "n/a"
    fd = f"{(y - x) * mul:+.1f}" if isinstance(x, (int, float)) and isinstance(y, (int, float)) else ""
    L.append(f"| {name} | {fx} | {fy} | {fd} |")
L += ["", "Setup: greedy decoding, max 256 new tokens, model chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH).",
      "MCQ = zero-shot, argmax next-token logit over A-D. AuditKIT `chrf` is character 6-gram F1 (beta 1), not chrF++.",
      "Model weights: 1-800-LLMs/tiny-aya-earth (identical LFS hashes to CohereLabs/tiny-aya-earth). Compute: free Kaggle T4 x2."]
if ti:
    L += ["", "## Training", f"- Framework: {ti.get('framework')}", f"- Hyper-parameters: {ti.get('hp')}",
          f"- Result: {ti.get('result')}", f"- Final dev eval: {ti.get('final_eval')}",
          f"- Train wall time (s): {round(ti.get('train_seconds', 0))}"]
    if ti.get("aligntune_error"):
        L.append(f"- AlignTune error before fallback: `{ti['aligntune_error'].strip()[:400]}`")
open(f"{R}/comparison.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
