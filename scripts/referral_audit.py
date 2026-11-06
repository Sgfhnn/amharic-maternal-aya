"""Run four: AuditKIT-style over-referral audit (per-sample regex metrics, sliced by question care label; no LLM).

Per answer (scripts/referral_lib.py):
  referral_line  - a sentence names a facility / health worker AND tells the reader to go / consult / get care
  boilerplate    - a referral sentence with no condition and no urgency word (the generic v3-DPO closing line)
  urgent_line    - a referral sentence with an urgency word (now / immediately / urgently)
Slices: test questions by LLM care label (results/referral_audit/test_care_labels.jsonl, DeepSeek-V4.1-Flash, same
triage prompt as the run-three train labels): urgent / routine (should see a health worker, not an emergency) /
none (facts, definitions); "non-urgent" = routine + none. Plus the held-out urgent probe (data/probe_urgent_v4.jsonl,
82 new danger-sign questions, never trained on) when its generations exist in results/probe_v4/.
Writes results/referral_audit/referral_audit.json and prints a markdown table.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referral_lib as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"
SYSTEMS = [("Baseline", "baseline_official/predictions.jsonl", None), ("v3 SFT", "sft_v3/eval/predictions.jsonl", "probe_v4/sft_v3.jsonl"),
           ("v3 DPO", "dpo_v3/eval/predictions.jsonl", "probe_v4/dpo_v3.jsonl"), ("v4 DPO", "dpo_v4/eval/predictions.jsonl", "probe_v4/dpo_v4.jsonl"),
           ("v4b DPO", "dpo_v4b/eval/predictions.jsonl", "probe_v4/dpo_v4b.jsonl"),
           ("v4c DPO", "dpo_v4c/eval/predictions.jsonl", "probe_v4/dpo_v4c.jsonl")]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def rates(preds):
    n = len(preds)
    if not n:
        return None
    return {"n": n, "referral_line": sum(map(L.referral_line, preds)) / n, "boilerplate": sum(map(L.boilerplate_line, preds)) / n,
            "urgent_line": sum(any(L.is_ref_sentence(s) and L.URGENT.search(s) for s in L.sentences(p)) for p in preds) / n,
            "wide_keyword": sum(map(L.wide_keyword, preds)) / n}


def main():
    lab = {r["id"]: r["need"] for r in jl(f"{R}/referral_audit/test_care_labels.jsonl")}
    out = {}
    for name, p, probe in SYSTEMS:
        if not os.path.exists(f"{R}/{p}"):
            continue
        rows = jl(f"{R}/{p}")
        sl = {"all": [r["prediction"] for r in rows]}
        for k in ("urgent", "routine", "none"):
            sl[k] = [r["prediction"] for r in rows if lab.get(r["id"]) == k]
        sl["non_urgent"] = sl["routine"] + sl["none"]
        out[name] = {k: rates(v) for k, v in sl.items()}
        if probe and os.path.exists(f"{R}/{probe}"):
            out[name]["probe_urgent"] = rates([r["prediction"] for r in jl(f"{R}/{probe}")])
    counts = {k: sum(v == k for v in lab.values()) for k in ("urgent", "routine", "none")}
    os.makedirs(f"{R}/referral_audit", exist_ok=True)
    json.dump({"label_counts": counts, "systems": out}, open(f"{R}/referral_audit/referral_audit.json", "w"), indent=1)
    names = list(out)
    f = lambda x, k: "n/a" if not x else f"{100 * x[k]:.1f}"
    lines = ["| Referral metric (regex, % of answers) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for sl, title in [("all", f"All test (550)"), ("non_urgent", f"Non-urgent test (routine+none, {counts['routine'] + counts['none']})"),
                      ("none", f"Factual / none test ({counts['none']})"), ("routine", f"Routine test ({counts['routine']})"),
                      ("urgent", f"Urgent test ({counts['urgent']})"), ("probe_urgent", "Held-out urgent probe (82)")]:
        for k, kt in [("referral_line", "referral line"), ("boilerplate", "generic boilerplate"), ("urgent_line", "urgent referral line")]:
            if sl in ("all",) and k == "urgent_line":
                continue
            lines.append(f"| {title}: {kt} | " + " | ".join(f(out[n].get(sl), k) for n in names) + " |")
    md = "\n".join(lines)
    open(f"{R}/referral_audit/referral_audit.md", "w").write(md + "\n")
    print(md)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
