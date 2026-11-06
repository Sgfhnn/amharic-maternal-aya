"""Write results/comparison_run2.md: baseline vs raw-SFT (A) vs curated-SFT (B), plus run one for reference.

Reads (whatever exists):
  results/baseline_official/metrics.json  (baseline, official CohereLabs/tiny-aya-earth, run two)  else results/baseline/
  results/sft_raw/eval/metrics.json, results/sft_raw/train_info.json        (Model A)
  results/sft_curated/eval/metrics.json, results/sft_curated/train_info.json (Model B)
  results/sft/eval/metrics.json                                             (run one SFT, for reference)
  results/judge/judge_summary.json                                          (LLM judge)
Usage: python scripts/make_comparison_run2.py
"""
import json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"


def load(p):
    p = f"{R}/{p}"
    return json.load(open(p)) if os.path.exists(p) else None


base_dir = "baseline_official" if os.path.exists(f"{R}/baseline_official/metrics.json") else "baseline"
cols = [("Baseline", load(f"{base_dir}/metrics.json")), ("Run 1 SFT (raw v2, lr 2e-4)", load("sft/eval/metrics.json")),
        ("A: raw-SFT", load("sft_raw/eval/metrics.json")), ("B: curated-SFT", load("sft_curated/eval/metrics.json"))]
tinfo = {"A": load("sft_raw/train_info.json"), "B": load("sft_curated/train_info.json")}
judge = load("judge/judge_summary.json")


def g(m, path):
    cur = m
    for p in path:
        if cur is None or not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


rows = [
    ("chrF++ test (corpus)", ["overall", "chrf_pp"], 1),
    ("BLEU-13a test (corpus)", ["overall", "bleu_13a"], 1),
    ("spBLEU flores200 test (corpus)", ["overall", "spbleu_flores200"], 1),
    ("chrF++ AmHQA (400)", ["by_source", "amhqa_mendeley", "chrf_pp"], 1),
    ("chrF++ Walelign SRH (150)", ["by_source", "walelign_srh", "chrf_pp"], 1),
    ("chrF++ maternal-topic rows (88)", ["by_topic", "maternal", "chrf_pp"], 1),
    ("AuditKIT chrf (per-sample mean)", ["auditkit_per_sample_mean", "headline", "chrf"], 100),
    ("AuditKIT rouge_l (per-sample mean)", ["auditkit_per_sample_mean", "headline", "rouge_l"], 100),
    ("Belebele amh_Ethi acc % (900)", ["mcq", "belebele_amh_Ethi", "accuracy"], 100),
    ("AfriMMLU amh acc % (500)", ["mcq", "afrimmlu_amh", "accuracy"], 100),
    ("Answers with Markdown %", ["diagnostics", "markdown_rate_pred"], 100),
    ("Hit 256-token cap %", ["diagnostics", "stopped_before_max_tokens"], -100),
    ("Referral keyword % (all test)", ["diagnostics", "referral_keyword_rate_pred"], 100),
    ("Avg answer chars (ref 507)", ["diagnostics", "avg_pred_chars"], 1),
    ("Ethiopic script share %", ["diagnostics", "avg_ethiopic_share"], 100),
]


def fmt(v, mul):
    if not isinstance(v, (int, float)):
        return "pending"
    if mul == -100:
        return f"{(1 - v) * 100:.1f}"
    return f"{v * mul:.1f}"


L = ["# Run two: baseline vs raw-SFT vs curated-SFT (Tiny Aya Earth, Amharic maternal/SRH health)", ""]
L.append("| Metric | " + " | ".join(c for c, _ in cols) + " |")
L.append("|---|" + "---|" * len(cols))
for name, path, mul in rows:
    L.append(f"| {name} | " + " | ".join(fmt(g(m, path), mul) for _, m in cols) + " |")
if judge:
    agg = judge["aggregate"]
    keymap = {"Baseline": "baseline", "A: raw-SFT": "A", "B: curated-SFT": "B"}
    for lab, k in [("LLM judge correctness 1-5", "correctness_mean"), ("LLM judge Amharic fluency 1-5", "fluency_mean"),
                   ("LLM judge safe referral when needed %", "safe_referral_rate_when_needed")]:
        cells = []
        for c, _ in cols:
            v = g(agg, [keymap.get(c, "-"), k])
            cells.append("n/a" if c not in keymap else (f"{v * (100 if '%' in lab else 1):.2f}" if isinstance(v, (int, float)) else "pending"))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
L += ["",
      f"Baseline column: `results/{base_dir}` ({'official CohereLabs/tiny-aya-earth, re-run in run two' if base_dir == 'baseline_official' else 'run one baseline on the byte-identical mirror'}). "
      "Run 1 column is the earlier SFT (lr 2e-4, NaN dev loss) for reference.",
      "Generation: greedy, max 256 new tokens, chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH; never trained on). "
      "MCQ: zero-shot, argmax next-token logit over A-D. AuditKIT `chrf` is char 6-gram F1, not chrF++. "
      "\"Hit 256-token cap\" = share of answers that did not stop on their own.", ""]
if judge:
    L += [f"LLM judge: {judge['model']} via Cline API (free promo credit), {judge['judged_rows']}/{judge['sample_size']} rows of a fixed seeded sample "
          "(60 Walelign SRH + 40 AmHQA), three answers per call under shuffled blind labels. "
          f"Raw counts: {json.dumps(judge['aggregate'])}", ""]
L += ["## Training (identical settings for A and B)", ""]
for k, ti in tinfo.items():
    if not ti:
        L.append(f"- {k}: pending"); continue
    fe = (ti.get("final_eval") or {}).get("eval_loss")
    L.append(f"- {k}: framework {ti.get('framework')}; model {ti.get('model_id')}; hp {ti.get('hp')}; "
             f"train rows kept {g(ti, ['data', 'train', 'kept'])} (length guard dropped {g(ti, ['data', 'train', 'dropped_no_supervised_tokens'])}); "
             f"dev rows {g(ti, ['data', 'val', 'kept'])}; steps {g(ti, ['result', 'steps'])}; train loss {g(ti, ['result', 'final_loss'])}; "
             f"final dev eval_loss {fe}; non-finite events {len(ti.get('nan_events', []))}; wall {round(ti.get('train_seconds', 0))} s")
cur = json.load(open(f"{ROOT}/data/curated/curation_stats.json")) if os.path.exists(f"{ROOT}/data/curated/curation_stats.json") else None
L += ["", "## Notes", "",
      "- NaN dev loss (run one): 119/374 dev rows and 2,046/7,109 train rows (all ethionlp_general, long XL-Sum/MT prompts) had a prompt "
      "(Cohere preamble ~360 tokens + article) that already filled max_length 1024. TRL truncates from the right, so every assistant label "
      "was masked; an eval batch with zero supervised tokens gives 0/0 = NaN and poisons the mean. Train loss stayed finite because TRL "
      "normalises by tokens across the accumulated batch, but ~29% of train rows contributed nothing. Fix in run two: a length guard "
      "(drop rows with < 16 supervised tokens inside 1024) applied identically to A and B, LoRA params fp32 under fp16 autocast, "
      "lr 1e-4, max_grad_norm 1.0. Sanity kernel (50 steps): eval_loss 1.670 -> 1.653, no non-finite values.",
      "- Model: official CohereLabs/tiny-aya-earth (HF access granted to 5LeaderW). The kernel assembles it from the official "
      "config/tokenizer files plus weight shards whose SHA-256 matches the official LFS oids, so no token sits in the kernel."]
if cur:
    tr = {s["step"].split(" ")[0]: s for s in cur["train"]["steps"]}
    L += [f"- Curation (CuratorKIT): train 7,109 -> {cur['train']['final']}, dev 374 -> {cur['dev']['final']}; "
          f"dedup {tr['ExactDeduplicator']['removed'] + tr['MinHashDeduplicator']['removed']}, broken text {tr['BrokenTextGate']['removed'] + tr['URLSentenceStripper']['removed']}, "
          f"English-heavy {tr['EnglishHeavyGate']['removed']}, off-topic {tr['OffTopicGate']['removed']}, Q/A mismatch {tr['QAMismatchGate']['removed']}, "
          f"EthioNLP downsample {tr['EthioNLPDownsampler']['removed']} (kept {tr['EthioNLPDownsampler']['kept']}), "
          f"referral line added to {tr['DangerSignReferral']['referral_line_added']} of {tr['DangerSignReferral']['danger_sign_questions']} danger-sign answers. "
          "Details: data/curated/curation_report.md"]
open(f"{R}/comparison_run2.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
