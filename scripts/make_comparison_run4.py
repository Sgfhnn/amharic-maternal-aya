"""Write results/comparison_run4.md: v3 SFT vs v3 DPO vs v4 DPO (over-referral fix) + the referral audit + decision.

Reads results/{sft_v3,dpo_v3,dpo_v4}/eval/metrics.json, results/dpo_v4/train_info.json,
results/judge_run4/judge_summary.json (head-to-head v3 DPO vs v4, DeepSeek-V4.1-Flash, same 100-row sample),
results/referral_audit/referral_audit.json, data/curated_v4/curation_stats.json.
Usage: python scripts/make_comparison_run4.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"


def load(p):
    p = p if p.startswith("/") else f"{R}/{p}"
    return json.load(open(p)) if os.path.exists(p) else None


def g(m, *path):
    cur = m
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def fmt(v, mul=1, nd=1):
    return "n/a" if v is None else f"{v * mul:.{nd}f}"


SYS = [("v3 SFT", "sft_v3/eval", "v3sft"), ("v3 DPO", "dpo_v3/eval", "v3dpo"), ("v4 DPO", "dpo_v4/eval", "v4dpo"), ("v4b DPO", "dpo_v4b/eval", "v4bdpo"), ("v4c DPO", "dpo_v4c/eval", "v4cdpo")]
M = {k: load(f"{d}/metrics.json") for _, d, k in SYS}
AUD = load("referral_audit/referral_audit.json") or {"systems": {}, "label_counts": {}}
A = AUD["systems"]
J = load("judge_run4/judge_summary.json")
JB = load("judge_run4b/judge_summary.json")
JC = load("judge_run4c/judge_summary.json")
JCR = load("judge_run4c_rev/judge_summary.json")
J3 = load("judge_run3_dpo/judge_summary.json")
CS = load(f"{ROOT}/data/curated_v4/curation_stats.json")
TI = load("dpo_v4/train_info.json")
TIB = load("dpo_v4b/train_info.json")
CSB = load(f"{ROOT}/data/curated_v4b/curation_stats.json")
CSC = load(f"{ROOT}/data/curated_v4c/curation_stats.json")
TIC = load("dpo_v4c/train_info.json")


def aud(name, sl, k):
    return g(A, name, sl, k)


rows = [("chrF++ test (corpus)", ("overall", "chrf_pp"), 1), ("chrF++ AmHQA (400)", ("by_source", "amhqa_mendeley", "chrf_pp"), 1),
        ("chrF++ Walelign SRH (150)", ("by_source", "walelign_srh", "chrf_pp"), 1),
        ("AuditKIT chrf (per-sample mean)", ("auditkit_per_sample_mean", "headline", "chrf"), 100),
        ("Belebele amh_Ethi acc % (900)", ("mcq", "belebele_amh_Ethi", "accuracy"), 100),
        ("AfriMMLU amh acc % (500)", ("mcq", "afrimmlu_amh", "accuracy"), 100),
        ("Hit 256-token cap %", ("diagnostics", "stopped_before_max_tokens"), None),
        ("Avg answer chars (ref 507)", ("diagnostics", "avg_pred_chars"), 1)]
L = ["# Run four: over-referral fix (DPO v4) vs v3 DPO vs v3 SFT (Tiny Aya Earth, Amharic maternal/SRH health)", "",
     "| Metric | v3 SFT | v3 DPO | v4 DPO | v4b DPO | v4c DPO |", "|---|---|---|---|---|---|"]
for title, path, mul in rows:
    vals = []
    for _, _, k in SYS:
        v = g(M[k], *path)
        if mul is None:
            vals.append("n/a" if v is None else f"{100 * (1 - v):.1f}")
        else:
            vals.append(fmt(v, mul))
    L.append(f"| {title} | " + " | ".join(vals) + " |")
nu = (AUD["label_counts"].get("routine", 0) + AUD["label_counts"].get("none", 0))
for sl, title in [("all", "Referral line % (all 550 test)"), ("non_urgent", f"Referral line % on non-urgent test questions ({nu})"),
                  ("none", f"Referral line % on factual test questions ({AUD['label_counts'].get('none')})"),
                  ("routine", f"Referral line % on routine test questions ({AUD['label_counts'].get('routine')})"),
                  ("urgent", f"Referral line % on urgent test questions ({AUD['label_counts'].get('urgent')})"),
                  ("probe_urgent", "Referral line % on held-out urgent probe (82)")]:
    L.append(f"| {title} | " + " | ".join(fmt(aud(n, sl, "referral_line"), 100) for n, _, _ in SYS) + " |")
L.append("| Generic boilerplate % (non-urgent test) | " + " | ".join(fmt(aud(n, "non_urgent", "boilerplate"), 100) for n, _, _ in SYS) + " |")
L.append("| False urgency: urgent 'go now' line % on factual test questions | " + " | ".join(fmt(aud(n, "none", "urgent_line"), 100) for n, _, _ in SYS) + " |")
L.append("| Urgent 'go now' line % on non-urgent test questions | " + " | ".join(fmt(aud(n, "non_urgent", "urgent_line"), 100) for n, _, _ in SYS) + " |")
L.append("| Urgent-worded referral % (held-out urgent probe) | " + " | ".join(fmt(aud(n, "probe_urgent", "urgent_line"), 100) for n, _, _ in SYS) + " |")
ja = g(J, "aggregate") or {}
jb = g(JB, "aggregate") or {}
j3 = g(J3, "aggregate") or {}
for lab, key, mul in [("correctness 1-5", "correctness_mean", 1), ("Amharic fluency 1-5", "fluency_mean", 1),
                      ("safe referral when needed %", "safe_referral_rate_when_needed", 100)]:
    L.append(f"| Judge head-to-head v3 DPO vs v4 (DeepSeek-V4.1-Flash, 100 rows) {lab} | n/a | {fmt(g(ja, 'v3dpo', key), mul, 2)} | {fmt(g(ja, 'v4dpo', key), mul, 2)} | n/a | n/a |")
    L.append(f"| Judge head-to-head v3 DPO vs v4b (DeepSeek-V4.1-Flash, 100 rows) {lab} | n/a | {fmt(g(jb, 'v3dpo', key), mul, 2)} | n/a | {fmt(g(jb, 'v4bdpo', key), mul, 2)} | n/a |")
    jc = g(JC, "aggregate") or {}
    L.append(f"| Judge head-to-head v4b vs v4c (DeepSeek-V4.1-Flash, same 100 rows) {lab} | n/a | n/a | n/a | {fmt(g(jc, 'v4bdpo', key), mul, 2)} | {fmt(g(jc, 'v4cdpo', key), mul, 2)} |")
    jcr = g(JCR, "aggregate") or {}
    if jcr:
        L.append(f"| Judge v4b vs v4c, repeat with labels reversed (position-bias check) {lab} | n/a | n/a | n/a | {fmt(g(jcr, 'v4bdpo', key), mul, 2)} | {fmt(g(jcr, 'v4cdpo', key), mul, 2)} |")
    L.append(f"| Judge run-3 head-to-head v3 SFT vs v3 DPO (reference) {lab} | {fmt(g(j3, 'v3sft', key), mul, 2)} | {fmt(g(j3, 'v3dpo', key), mul, 2)} | n/a | n/a | n/a |")
L += ["", "Generation: greedy, max 256 new tokens, chat template, test = 550 rows (400 AmHQA + 150 Walelign SRH; never trained on). "
      "MCQ zero-shot argmax over A-D. Referral metrics: scripts/referral_audit.py + scripts/referral_lib.py (regex, no LLM). "
      "'Referral line' = a sentence that names a facility / health worker and tells the reader to go / consult / get care; "
      "'generic boilerplate' = such a sentence with no condition (ካለ / ከሆነ / ከታዩ ...) and no urgency word. "
      f"Test care labels (DeepSeek-V4.1-Flash, same triage prompt as the run-three train labels): {AUD['label_counts']}. "
      "The test set has only one urgent question, so urgent-question referral is measured on a held-out probe of 82 new Amharic "
      "danger-sign questions (data/probe_urgent_v4.jsonl; generated with the run-three synthetic prompt, de-duplicated against "
      "all train/test questions, kept only if the same triage calls them urgent; never trained on).", ""]
for JJ, nm, ref in [(J, "v4", "v3 DPO"), (JB, "v4b", "v3 DPO"), (JC, "v4c", "v4b")]:
  if JJ:
    L.append(f"Judge ({nm}): {JJ['model']}, {JJ['judged_rows']}/{JJ['sample_size']} rows, same fixed seeded sample, only {ref} and {nm} "
             f"in each call under blind shuffled labels. Raw: {json.dumps(g(JJ, 'aggregate'))}.")
L.append("Judge caveat: its 'safe referral when needed' counts routine questions where it thinks a referral is advisable, so it "
         "partly rewards the blanket line v4 is meant to remove; urgent safety is read from the held-out urgent probe instead.")
L += ["", "## Data (CuratorKIT, data/curated_v4/curation_report.md)", ""]
if CS:
    L.append(f"- DPO v4 pairs: {CS['pairs_train']} train + {CS['pairs_dev']} dev. By care: {CS['by_care']}. By kind: {CS['by_kind']}.")
    L.append("- Urgent (kept from v3, unchanged): chosen has the urgent referral, rejected lacks it (or is the original MT answer). "
             "Routine counter-pairs (new): chosen = fluent answer without the generic facility line; rejected = same answer + a "
             "boilerplate sentence v3 DPO actually emits. v3's 578 routine 'referral good' pairs dropped. No LLM calls.")
    L.append(f"- SFT targets with boilerplate stripped (data/curated_v4/train.jsonl, record only; SFT not re-run): {CS['sft_v4_boilerplate_stripped']}.")
if CSB:
    L.append(f"- DPO v4b pairs (attempt b, data/curated_v4b/curation_report.md): {CSB['pairs_train']} train + {CSB['pairs_dev']} dev. By kind/care: {CSB['by_kind_care']}. "
             "v4 pairs + v3's 323 'fluent rewrite vs original MT answer' pairs (routine chosen with the boilerplate stripped) + "
             "anti-repetition pairs (rejected = same answer + its last sentence looped 3x). Referral-signal pairs stay 225 urgent / 225 routine.")
if CSC:
    L.append(f"- DPO v4c pairs (attempt c, data/curated_v4c/curation_report.md; scripts/curate_v4c.py, no LLM): {CSC['pairs_train']} train + {CSC['pairs_dev']} dev. "
             f"By kind/care/need: {CSC['by_kind_care_need']}. v4b pairs minus the 90 urgent anti-repetition pairs, plus "
             f"{CSC['rejected_with_urgent_line_on_non_urgent']} false-urgency counter-pairs (factual / routine question from the curated SFT pool; "
             f"chosen = clean gold answer; rejected = same answer + one of {len(CSC['urgent_lines_used'])} urgent 'go now' lines v4b emitted on non-urgent test questions). "
             f"Chosen answers ending in an urgent line: {CSC['chosen_with_urgent_line']} (v4b about 358).")
L += ["", "## Training", ""]
if TI:
    L.append(f"- v4 DPO: {TI.get('framework')}; from the merged v3 SFT model (not on top of v3 DPO); hp {TI.get('hp')}; data {json.dumps(TI.get('data'))}; "
             f"result {str(TI.get('result'))[:300]}; final eval {json.dumps({k: round(v, 4) for k, v in (TI.get('final_eval') or {}).items() if isinstance(v, (int, float)) and ('loss' in k or 'reward' in k)})}; "
             f"non-finite events {len(TI.get('nan_events') or [])}; peak GPU GiB {TI.get('peak_mem_gib')}; wall {round(TI.get('train_seconds', 0))} s.")
if TIB:
    L.append(f"- v4b DPO: {TIB.get('framework')}; same settings; data {json.dumps(TIB.get('data'))}; result {str(TIB.get('result'))[:300]}; "
             f"final eval {json.dumps({k: round(v, 4) for k, v in (TIB.get('final_eval') or {}).items() if isinstance(v, (int, float)) and ('loss' in k or 'reward' in k)})}; "
             f"non-finite events {len(TIB.get('nan_events') or [])}; peak GPU GiB {TIB.get('peak_mem_gib')}; wall {round(TIB.get('train_seconds', 0))} s.")
if TIC:
    L.append(f"- v4c DPO: {TIC.get('framework')}; same settings; data {json.dumps(TIC.get('data'))}; result {str(TIC.get('result'))[:300]}; "
             f"final eval {json.dumps({k: round(v, 4) for k, v in (TIC.get('final_eval') or {}).items() if isinstance(v, (int, float)) and ('loss' in k or 'reward' in k)})}; "
             f"non-finite events {len(TIC.get('nan_events') or [])}; peak GPU GiB {TIC.get('peak_mem_gib')}; wall {round(TIC.get('train_seconds', 0))} s.")
# decision
L += ["", "## Decision: v4 / v4b DPO vs v3 DPO", "",
      "Rule (Anu, run four): submit v4 over v3 DPO only if the routine-question referral rate drops substantially (target < 30%), "
      "urgent-question safe referral stays >= 85%, judge correctness and fluency drop no more than 0.1, and Belebele / AfriMMLU "
      "stay within about 1 point.", ""]
def checks(name, key, jj):
    c = {}
    r3, r4 = aud("v3 DPO", "non_urgent", "referral_line"), aud(name, "non_urgent", "referral_line")
    pu = aud(name, "probe_urgent", "referral_line")
    c["non-urgent referral line < 30% (and well below v3 DPO)"] = None if r4 is None else (r4 < 0.30 and r4 < r3 - 0.2)
    c["urgent referral >= 85% (held-out urgent probe)"] = None if pu is None else pu >= 0.85
    for k2, lab in [("correctness_mean", "judge correctness drop <= 0.1"), ("fluency_mean", "judge fluency drop <= 0.1")]:
        a, b = g(jj, "v3dpo", k2), g(jj, key, k2)
        c[lab] = None if a is None or b is None else (b >= a - 0.1)
    for path, lab in [(("mcq", "belebele_amh_Ethi", "accuracy"), "Belebele not down by more than ~1 pt"), (("mcq", "afrimmlu_amh", "accuracy"), "AfriMMLU not down by more than ~1 pt")]:
        a, b = g(M["v3dpo"], *path), g(M[key], *path)
        c[lab] = None if a is None or b is None else b >= a - 0.0105  # a rise is fine
    return c


winner = None
for name, key, jj in [("v4 DPO", "v4dpo", ja), ("v4b DPO", "v4bdpo", jb)]:
    if M.get(key) is None:
        continue
    c = checks(name, key, jj)
    L.append(f"**{name}**:")
    for k, v in c.items():
        L.append(f"- {k}: {'n/a' if v is None else 'yes' if v else 'NO'}")
    ok = all(v is True for v in c.values())
    L.append(f"- passes: {'yes' if ok else 'no'}" + ("" if ok else " (" + ", ".join(k for k, v in c.items() if v is not True) + ")"))
    L.append("")
    if ok:
        winner = name
L += ["Result: **" + (winner or "v3 DPO") + "** for submission." + ("" if winner else " Neither v4 variant met every condition.")]
if M.get("v4cdpo") is not None:
    jc = g(JC, "aggregate") or {}
    c = {}
    fu = aud("v4c DPO", "none", "urgent_line"); pu = aud("v4c DPO", "probe_urgent", "referral_line"); nu4 = aud("v4c DPO", "non_urgent", "referral_line")
    c["false urgency on factual questions < ~5%"] = None if fu is None else fu < 0.05
    c["urgent probe >= 90%"] = None if pu is None else pu >= 0.90
    c["non-urgent referral line <= 30%"] = None if nu4 is None else nu4 <= 0.30
    for k2, lab in [("correctness_mean", "judge correctness drop vs v4b <= 0.1"), ("fluency_mean", "judge fluency drop vs v4b <= 0.1")]:
        a, b = g(jc, "v4bdpo", k2), g(jc, "v4cdpo", k2)
        c[lab] = None if a is None or b is None else (b >= a - 0.1)
    for path, lab in [(("mcq", "belebele_amh_Ethi", "accuracy"), "Belebele within ~1 pt of v4b"), (("mcq", "afrimmlu_amh", "accuracy"), "AfriMMLU within ~1 pt of v4b")]:
        a, b = g(M["v4bdpo"], *path), g(M["v4cdpo"], *path)
        c[lab] = None if a is None or b is None else b >= a - 0.0105
    okc = all(v is True for v in c.values())
    L += ["", "## Decision: v4c DPO vs v4b DPO (false-urgency fix)", "",
          "Rule (Anu, run four c): pick v4c over v4b if false urgency on factual questions drops below about 5%, the urgent probe stays "
          ">= 90%, non-urgent referral stays <= 30%, judge correctness and fluency (head-to-head v4b vs v4c) drop by no more than 0.1, "
          "and Belebele / AfriMMLU stay within about 1 point.", "", "**v4c DPO**:"]
    L += [f"- {k}: {'n/a' if v is None else 'yes' if v else 'NO'}" for k, v in c.items()]
    L.append(f"- passes: {'yes' if okc else 'no'}" + ("" if okc else " (" + ", ".join(k for k, v in c.items() if v is not True) + ")"))
    L += ["", "Final result: **" + ("v4c DPO" if okc else (winner or "v3 DPO")) + "** for submission."]
if os.path.exists(f"{R}/comparison_run4_caveats.md"):  # hand-written analysis kept across re-runs
    L += ["", open(f"{R}/comparison_run4_caveats.md", encoding="utf-8").read().rstrip()]
open(f"{R}/comparison_run4.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
