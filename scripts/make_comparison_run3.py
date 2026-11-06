"""Write results/comparison_run3.md: baseline vs run-2 raw (A) vs run-2 curated (B) vs v3 SFT vs v3 DPO.

Reads whatever exists: results/baseline_official, results/sft_raw/eval, results/sft_curated/eval, results/sft_v3/eval,
results/dpo_v3/eval (metrics.json + predictions.jsonl), train_info.json files, results/judge_run3/judge_summary.json
(run-three judge, all five systems in one blind call per row) and results/judge/judge_summary.json (run-two judge).
Usage: python scripts/make_comparison_run3.py
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curate_v3 import REFERRAL as REF_WIDE  # wider referral word list used in run-three curation


def load(p):
    p = f"{R}/{p}"
    return json.load(open(p)) if os.path.exists(p) else None


SYS = [("Baseline", "baseline", "baseline_official"), ("Run-2 A raw-SFT", "A", "sft_raw/eval"),
       ("Run-2 B curated-SFT", "B", "sft_curated/eval"), ("v3 SFT", "v3sft", "sft_v3/eval"), ("v3 DPO", "v3dpo", "dpo_v3/eval")]
M = {k: load(f"{d}/metrics.json") for _, k, d in SYS}


def g(m, path):
    cur = m
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def wide_ref_rate(d):
    p = f"{R}/{d}/predictions.jsonl"
    if not os.path.exists(p):
        return None
    preds = [json.loads(l)["prediction"] for l in open(p, encoding="utf-8")]
    return sum(bool(REF_WIDE.search(x)) for x in preds) / len(preds)


rows = [
    ("chrF++ test (corpus)", ["overall", "chrf_pp"], 1),
    ("BLEU-13a test (corpus)", ["overall", "bleu_13a"], 1),
    ("spBLEU flores200 test (corpus)", ["overall", "spbleu_flores200"], 1),
    ("chrF++ AmHQA (400)", ["by_source", "amhqa_mendeley", "chrf_pp"], 1),
    ("chrF++ Walelign SRH (150)", ["by_source", "walelign_srh", "chrf_pp"], 1),
    ("chrF++ maternal-topic rows (88)", ["by_topic", "maternal", "chrf_pp"], 1),
    ("AuditKIT chrf (per-sample mean)", ["auditkit_per_sample_mean", "headline", "chrf"], 100),
    ("Belebele amh_Ethi acc % (900)", ["mcq", "belebele_amh_Ethi", "accuracy"], 100),
    ("AfriMMLU amh acc % (500)", ["mcq", "afrimmlu_amh", "accuracy"], 100),
    ("Answers with Markdown %", ["diagnostics", "markdown_rate_pred"], 100),
    ("Hit 256-token cap %", ["diagnostics", "stopped_before_max_tokens"], -100),
    ("Referral keyword % (run-2 list, all test)", ["diagnostics", "referral_keyword_rate_pred"], 100),
    ("Avg answer chars (ref 507)", ["diagnostics", "avg_pred_chars"], 1),
    ("Ethiopic script share %", ["diagnostics", "avg_ethiopic_share"], 100),
]


def fmt(v, mul):
    if not isinstance(v, (int, float)):
        return "pending"
    return f"{(1 - v) * 100:.1f}" if mul == -100 else f"{v * mul:.1f}"


L = ["# Run three: baseline vs run-2 raw vs run-2 curated vs v3 SFT vs v3 DPO (Tiny Aya Earth, Amharic maternal/SRH health)", "",
     "| Metric | " + " | ".join(n for n, _, _ in SYS) + " |", "|---|" + "---|" * len(SYS)]
for name, path, mul in rows:
    L.append(f"| {name} | " + " | ".join(fmt(g(M[k], path), mul) for _, k, _ in SYS) + " |")
L.append("| Referral keyword % (wide v3 list, all test) | " + " | ".join(fmt(wide_ref_rate(d), 100) for _, _, d in SYS) + " |")
j3, j2 = load("judge_run3/judge_summary.json"), load("judge/judge_summary.json")
for jname, j in [("judge run 3", j3), ("judge run 2", j2)]:
    if not j:
        continue
    for lab, key, mul in [("correctness 1-5", "correctness_mean", 1), ("Amharic fluency 1-5", "fluency_mean", 1),
                          ("safe referral when needed %", "safe_referral_rate_when_needed", 100)]:
        cells = []
        for _, k, _ in SYS:
            v = g(j["aggregate"], [k, key])
            cells.append(f"{v * mul:.2f}" if isinstance(v, (int, float)) else ("n/a" if k not in j["aggregate"] else "pending"))
        L.append(f"| LLM {jname} ({j['model'].split('/')[-1].split(':')[0]}) {lab} | " + " | ".join(cells) + " |")
jh = load("judge_run3_dpo/judge_summary.json")
if jh:
    for lab, key, mul in [("correctness 1-5", "correctness_mean", 1), ("Amharic fluency 1-5", "fluency_mean", 1),
                          ("safe referral when needed %", "safe_referral_rate_when_needed", 100)]:
        cells = []
        for _, k, _ in SYS:
            v = g(jh["aggregate"], [k, key])
            cells.append(f"{v * mul:.2f}" if isinstance(v, (int, float)) else "n/a")
        L.append(f"| LLM judge run 3 head-to-head v3 SFT vs v3 DPO ({jh['model'].split('/')[-1].split(':')[0]}) {lab} | " + " | ".join(cells) + " |")
L += ["", "Generation: greedy, max 256 new tokens, chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH; never trained on). "
      "MCQ: zero-shot, argmax next-token logit over A-D. AuditKIT `chrf` is char 6-gram F1, not chrF++. \"Hit 256-token cap\" = share of "
      "answers that did not stop on their own.", ""]
if j3:
    L += [f"Run-three judge: {j3['model']} via HF Inference Providers, {j3['judged_rows']}/{j3['sample_size']} rows of the same fixed seeded "
          "sample as run two (60 Walelign SRH + 40 AmHQA), all five answers in one call per row under shuffled blind labels. It is a "
          "different model from the Gemma model that rewrote the v3 training answers, to avoid self-preference. "
          f"Raw counts: {json.dumps(j3['aggregate'])}", ""]
if jh:
    L += [f"Head-to-head judge (DPO retry): same judge model and same 100-row sample, but each call shows only the v3 SFT and v3 DPO "
          f"answers (blind shuffled labels), {jh['judged_rows']}/{jh['sample_size']} rows. The five-system row above was judged before DPO "
          "existed, so the DPO decision uses this head-to-head row; v3 SFT is re-scored in it so both sit on one scale. "
          f"Raw counts: {json.dumps(jh['aggregate'])}", ""]
if j2:
    L += [f"Run-two judge (reference only, three systems): {j2['model']} via Cline, {j2['judged_rows']} rows. Its scale is not "
          "comparable to the run-three judge; compare systems within one judge row.", ""]
L += ["## Training", ""]
for lab, p in [("Run-2 B curated-SFT", "sft_curated/train_info.json"), ("v3 SFT", "sft_v3/train_info.json"),
               ("v3 DPO first try (kernel amh-tinyaya-r3, OOM)", "dpo_v3_attempt1_oom/train_info.json"),
               ("v3 DPO retry (kernel amh-tinyaya-r3-dpo, low-memory)", "dpo_v3/train_info.json")]:
    ti = load(p)
    if not ti:
        L.append(f"- {lab}: pending"); continue
    fe = (ti.get("final_eval") or {})
    L.append(f"- {lab}: framework {ti.get('framework', 'FAILED')}; hp {ti.get('hp')}; data {json.dumps(g(ti, ['data']))[:300]}; "
             f"result {str(ti.get('result'))[:200]}; final eval {json.dumps({k: round(v, 4) for k, v in fe.items() if isinstance(v, float) and k in ('eval_loss', 'eval_rewards/accuracies', 'eval_rewards/margins')})}; "
             f"non-finite events {len(ti.get('nan_events', []))}; wall {round(ti.get('train_seconds', 0))} s"
             + (f"; used {ti.get('used')}; final DPOConfig {json.dumps(ti.get('final_dpo_args'), default=str)}; peak GPU GiB {ti.get('peak_mem_gib')}; device map {ti.get('device_map_counts')}" if ti.get("final_dpo_args") else "")
             + (f"; attempts/errors {json.dumps([a['error'][-400:] for a in ti.get('attempts', [])])}" if ti.get("attempts") else ""))
cs = json.load(open(f"{ROOT}/data/curated_v3/curation_stats.json"))
L += ["", "## Pipeline", "",
      "- v3 SFT: run-2 curated adapter merged into the official CohereLabs/tiny-aya-earth, then a fresh LoRA (AlignTune TRL backend, "
      "r16, alpha32, lr 1e-4, clip 1.0, max_len 1024, length guard, 1 epoch) on data/curated_v3 "
      f"({cs['final_train']} rows). Model = official + run-2 B adapter + v3 adapter (stacked).",
      "- v3 DPO: v3 SFT merged, then a fresh LoRA trained with AlignTune DPO (TRL backend, beta 0.1, lr 5e-5, 1 epoch, sigmoid loss); "
      f"reference = the v3 SFT model (initial zero-delta LoRA copy, i.e. adapter effectively off). Pairs: {cs['dpo']}. "
      "First try OOM'd on a T4 (bs 2, max_len 1024). Retry (scripts/train_dpo_lowmem.py): bs 1 x grad-acc 16, gradient checkpointing, "
      "reference log-probs precomputed once (no in-step reference forward, no second model copy), max_len 768 keep_start "
      "(512 would make 453/763 pairs identical because the stock chat template's system preamble makes prompts 376-440 tokens; "
      "no prompt cap for the same reason, and TRL 1.7.1 has no max_prompt_length), fp16 base + fp32 LoRA, model split over both "
      "T4s. Two TRL patches, still run through AlignTune: chunked entropy metric, and an in-memory replacement for "
      "DPOTrainer._precompute_ref_logps (the stock one fails on AlignTune's in-memory datasets).",
      f"- Curation: data/curated_v3/curation_report.md. Danger-sign matcher: {json.dumps({k: cs['danger_matcher'][k] for k in ['regex_matches', 'regex_matches_llm_confirmed_urgent', 'llm_urgent_total', 'llm_routine_total']})}."]
if jh and M.get("v3dpo") and M.get("v3sft"):
    a, b = jh["aggregate"]["v3sft"], jh["aggregate"]["v3dpo"]
    bel = lambda k: g(M[k], ["mcq", "belebele_amh_Ethi", "accuracy"]) * 100
    mm = lambda k: g(M[k], ["mcq", "afrimmlu_amh", "accuracy"]) * 100
    checks = {"safe referral rises": b["safe_referral_rate_when_needed"] > a["safe_referral_rate_when_needed"],
              "fluency drop <= 0.05": b["fluency_mean"] >= a["fluency_mean"] - 0.05,
              "correctness does not drop": b["correctness_mean"] >= a["correctness_mean"],
              "Belebele within ~1 pt": abs(bel("v3dpo") - bel("v3sft")) <= 1.0,
              "AfriMMLU within ~1 pt": abs(mm("v3dpo") - mm("v3sft")) <= 1.0}
    win = all(checks.values())
    wide = wide_ref_rate("dpo_v3/eval")
    L += ["", "## Decision: v3 DPO vs v3 SFT", "",
          "Rule: DPO replaces SFT only if safe referral rises, fluency drops no more than 0.05, correctness does not drop, and "
          "Belebele/AfriMMLU stay within about 1 point (judge numbers from the head-to-head row).", ""]
    L += [f"- {k}: {'yes' if v else 'NO'}" for k, v in checks.items()]
    L += [f"- Belebele {bel('v3sft'):.1f} -> {bel('v3dpo'):.1f}; AfriMMLU {mm('v3sft'):.1f} -> {mm('v3dpo'):.1f}",
          "", f"Result: **{'v3 DPO' if win else 'v3 SFT'}** for submission.", "",
          f"Caveat: v3 DPO appends a referral line to {wide * 100:.0f}% of all test answers, mostly one near-fixed sentence "
          "(\"for a full check-up and correct treatment, go to a health facility and consult a health professional\"), "
          "including on definitional questions. The judge's safe-referral score cannot penalise over-referral, so 100% "
          "partly reflects blanket referral. Conservative for a public health assistant, but say so in the write-up."]
open(f"{R}/comparison_run3.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
