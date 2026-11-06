"""Packaging: one metrics table for the submission chain baseline -> run-2 B -> v3 SFT -> v3 DPO -> v4b DPO -> v4c DPO.

Reads only files already in results/ (metrics.json, mcq_metrics.json, predictions.jsonl, probe_v4/*.jsonl,
referral_audit/test_care_labels.jsonl, judge_*/judge_summary.json). Referral rates are recomputed with
scripts/referral_lib.py (same regex as scripts/referral_audit.py), which also fills run-2 B (not in the run-four audit).
Writes results/summary_table.md and results/summary_table.json. No model or LLM calls.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referral_lib as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"
SYS = [("Baseline", "baseline_official", None), ("Run-2 B SFT", "sft_curated/eval", None),
       ("v3 SFT", "sft_v3/eval", "sft_v3"), ("v3 DPO", "dpo_v3/eval", "dpo_v3"),
       ("v4b DPO", "dpo_v4b/eval", "dpo_v4b"), ("v4c DPO (submitted)", "dpo_v4c/eval", "dpo_v4c")]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def urgent_line(t):
    return any(L.is_ref_sentence(s) and L.URGENT.search(s) for s in L.sentences(t))


def pct(xs, f):
    return 100 * sum(map(f, xs)) / len(xs) if xs else None


def main():
    lab = {r["id"]: r["need"] for r in jl(f"{R}/referral_audit/test_care_labels.jsonl")}
    table = {}
    for name, d, probe in SYS:
        m = json.load(open(f"{R}/{d}/metrics.json"))
        q = json.load(open(f"{R}/{d}/mcq_metrics.json"))
        preds = jl(f"{R}/{d}/predictions.jsonl")
        nonurg = [r["prediction"] for r in preds if lab.get(r["id"]) in ("routine", "none")]
        fact = [r["prediction"] for r in preds if lab.get(r["id"]) == "none"]
        dg = m["diagnostics"]
        row = {
            "chrF++ test (550)": m["overall"]["chrf_pp"],
            "chrF++ AmHQA (400)": m["by_source"]["amhqa_mendeley"]["chrf_pp"],
            "chrF++ Walelign SRH (150)": m["by_source"]["walelign_srh"]["chrf_pp"],
            "AuditKIT chrf per-sample mean": 100 * m["auditkit_per_sample_mean"]["headline"]["chrf"],
            "Belebele amh_Ethi acc % (900)": 100 * q["belebele_amh_Ethi"]["accuracy"],
            "AfriMMLU amh acc % (500)": 100 * q["afrimmlu_amh"]["accuracy"],
            "Hit 256-token cap %": 100 * (1 - dg["stopped_before_max_tokens"]),
            "Avg answer chars (ref 507)": dg["avg_pred_chars"],
            "Markdown in answer %": 100 * dg["markdown_rate_pred"],
            "Referral line % non-urgent test (549)": pct(nonurg, L.referral_line),
            "Urgent 'go now' line % factual test (447)": pct(fact, urgent_line),
        }
        pp = f"{R}/probe_v4/{probe}.jsonl" if probe else None
        row["Referral line % held-out urgent probe (82)"] = pct([r["prediction"] for r in jl(pp)], L.referral_line) if pp and os.path.exists(pp) else None
        table[name] = row
    names = [n for n, _, _ in SYS]
    metrics = list(next(iter(table.values())))
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in metrics:
        lines.append(f"| {k} | " + " | ".join("n/a" if table[n][k] is None else f"{table[n][k]:.1f}" for n in names) + " |")
    J = lambda p: json.load(open(f"{R}/{p}/judge_summary.json"))
    judge = [("Run 3 five-way (DeepSeek-V4.1-Flash, 100 rows)", "judge_run3", {"Baseline": "baseline", "Run-2 B SFT": "B", "v3 SFT": "v3sft"}),
             ("v3 SFT vs v3 DPO head-to-head", "judge_run3_dpo", {"v3 SFT": "v3sft", "v3 DPO": "v3dpo"}),
             ("v3 DPO vs v4b head-to-head", "judge_run4b", {"v3 DPO": "v3dpo", "v4b DPO": "v4bdpo"}),
             ("v4b vs v4c head-to-head", "judge_run4c", {"v4b DPO": "v4bdpo", "v4c DPO (submitted)": "v4cdpo"}),
             ("v4b vs v4c, labels reversed", "judge_run4c_rev", {"v4b DPO": "v4bdpo", "v4c DPO (submitted)": "v4cdpo"})]
    jt = {}
    for title, p, mp in judge:
        s = J(p)
        s = s["aggregate"]
        for col, key in mp.items():
            v = s[key]
            jt.setdefault(title, {})[col] = (v["correctness_mean"], v["fluency_mean"])
    for title in jt:
        lines.append(f"| Judge {title}: correctness / fluency (1-5) | " + " | ".join(
            f"{jt[title][n][0]:.2f} / {jt[title][n][1]:.2f}" if n in jt[title] else "" for n in names) + " |")
    md = "\n".join(lines)
    md += ("\n\nJudge rows compare systems only within one row (each row is its own judge pass; about +-0.1 run-to-run noise). "
           "Test = 550 rows (400 AmHQA + 150 Walelign SRH), greedy, max 256 new tokens. Referral regexes: scripts/referral_lib.py; "
           "test care labels: results/referral_audit/test_care_labels.jsonl (1 urgent, 102 routine, 447 none).\n")
    open(f"{R}/summary_table.md", "w").write(md)
    json.dump({"automatic": table, "judge": jt}, open(f"{R}/summary_table.json", "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
