"""Run four, attempt c: CuratorKIT DPO v4c data. No LLM calls (regex + scripts only).

Why: v4b removed the generic boilerplate (non-urgent referral line 86.5% -> 23.0%) and kept quality, but 19.2% of factual
test answers now end in an urgent "go to a health facility now" line (v3 DPO 1.3%), e.g. on Bell's palsy treatment.
Likely cause: ~358 v4b pairs had an urgent-ending chosen answer (225 urgent referral + 43 urgent fluency + 90 urgent
anti-repetition) and no pair ever showed "urgent line on a non-urgent question = bad".

v4c = v4b pairs
    - the 90 urgent anti-repetition pairs (their chosen answers all end in an urgent line)
    + false-urgency counter-pairs (new): prompt = a factual ('none') or routine health question from the curated v4 SFT
      pool (CuratorKIT quality-gated gold answers; ethionlp_general excluded); chosen = that clean answer (no urgent
      referral line; factual ones have no referral line at all); rejected = the same answer + one urgent "go now" line
      that v4b actually emitted on a non-urgent test question (mined from results/dpo_v4b/eval/predictions.jsonl;
      only the referral phrasing is reused, no test question or answer text). Generic, complete lines only.
Urgent referral pairs (225) and urgent fluency pairs (43) stay, so danger-sign recall should hold.
Usage: python scripts/curate_v4c.py
"""
import collections, json, os, random, sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referral_lib as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4, V4B, OUT = f"{ROOT}/data/curated_v4", f"{ROOT}/data/curated_v4b", f"{ROOT}/data/curated_v4c"
SEED = 11
MAX_TOK = 760
N_NONE, N_ROUTINE = 200, 70
DEV_FRAC = 0.05
SOURCES = {"afrihealth_qa", "walelign_srh", "amhqa_mendeley", "danilior_maternal"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def has_urgent_line(t):
    return any(L.is_ref_sentence(s) and L.URGENT.search(s) for s in L.sentences(t))


def mine_urgent_lines():
    lab = {r["id"]: r["need"] for r in jl(f"{ROOT}/results/referral_audit/test_care_labels.jsonl")}
    c = collections.Counter()
    for r in jl(f"{ROOT}/results/dpo_v4b/eval/predictions.jsonl"):
        if lab.get(r["id"]) == "urgent":
            continue
        for s in L.sentences(r["prediction"]):
            if L.is_ref_sentence(s) and L.URGENT.search(s):
                c[s] += 1
    keep = []
    for s, n in c.most_common():
        # generic, complete, short; no doubled words; no topic words (condition clauses tie a line to one disease)
        if not s.endswith("።") or len(s) > 110 or "አሁኑኑ አሁኑኑ" in s or L.COND.search(s):
            continue
        if any(w in s for w in ("ቅማል", "ልጅ", "ሕፃን", "ልጁ", "ሽንት", "ሆድ", "ደም")):
            continue
        keep.append((s, n))
    return keep, c


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/work/official_tinyaya")

    def ntok(p, a):
        return len(tok.apply_chat_template(p + [{"role": "assistant", "content": a}], tokenize=True, return_dict=True)["input_ids"])

    lines, allc = mine_urgent_lines()
    stats = collections.Counter({"mined_urgent_sentences_distinct": len(allc), "mined_urgent_sentences_total": sum(allc.values()),
                                 "urgent_lines_kept": len(lines)})
    v4b = {"train": jl(f"{V4B}/dpo/train.jsonl"), "dev": jl(f"{V4B}/dpo/dev.jsonl")}
    base = {}
    for part in ("train", "dev"):
        kept = [p for p in v4b[part] if not (p["kind"] == "anti_repetition" and p["care"] == "urgent")]
        stats[f"dropped_urgent_antirep_{part}"] = len(v4b[part]) - len(kept)
        base[part] = kept
    dev_base = {p["id"].split("#")[0] for p in v4b["dev"]}
    used = {p["id"].split("#")[0] for p in v4b["train"] + v4b["dev"]}
    test_q = {r["question"].strip() for r in jl(f"{ROOT}/data/test.jsonl")} if os.path.exists(f"{ROOT}/data/test.jsonl") else set()
    pool = {"none": [], "routine": []}
    for r in jl(f"{V4}/train.jsonl"):
        cv = r["curation_v4"] if isinstance(r["curation_v4"], dict) else eval(r["curation_v4"])
        care = cv.get("care")
        if care not in pool or r["source"] not in SOURCES:
            continue
        a, q = r["answer"].strip(), r["question"].strip()
        if r["id"] in dev_base or q in test_q:
            stats["pool_skip_dev_or_test"] += 1; continue
        if has_urgent_line(a) or (care == "none" and L.referral_line(a)):
            stats[f"pool_skip_ref_{care}"] += 1; continue
        if not a.endswith(("።", "?", "፧", ".")) or len(a) < 60:
            stats["pool_skip_unclean_end"] += 1; continue
        pool[care].append(r)
    rng = random.Random(SEED)
    new = []
    for care, n in (("none", N_NONE), ("routine", N_ROUTINE)):
        cand = pool[care]
        rng.shuffle(cand)
        cand.sort(key=lambda r: r["id"] in used)  # prefer prompts not already in v4b pairs (stable sort keeps shuffle)
        stats[f"pool_{care}"] = len(cand)
        k = 0
        for r in cand:
            if k >= n:
                break
            prompt = [{"role": "user", "content": r["question"].strip()}]
            ch = r["answer"].strip()
            line = lines[k % len(lines)][0] if k < len(lines) else rng.choice(lines)[0]
            rj = ch + " " + line
            if ntok(prompt, rj) > MAX_TOK:
                stats["skip_long"] += 1; continue
            new.append({"id": r["id"] + "#falseurg", "kind": "false_urgency", "care": "routine", "need": care, "prompt": prompt,
                        "chosen": [{"role": "assistant", "content": ch}], "rejected": [{"role": "assistant", "content": rj}],
                        "provenance": ["curated_v4 SFT answer", "UrgentLineAppender(v4b test emissions)", "SchemaGate"]})
            k += 1
    rng.shuffle(new)
    ndev = round(len(new) * DEV_FRAC)
    tr, dv = base["train"] + new[ndev:], base["dev"] + new[:ndev]
    random.Random(42).shuffle(tr)
    os.makedirs(f"{OUT}/dpo", exist_ok=True)
    for name, part in [("train", tr), ("dev", dv)]:
        with open(f"{OUT}/dpo/{name}.jsonl", "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    allp = tr + dv
    urg_end = sum(1 for p in allp if has_urgent_line(p["chosen"][0]["content"]))
    st = {"created": datetime.now(UTC).isoformat(), "pairs_train": len(tr), "pairs_dev": len(dv),
          "by_kind_care_need": dict(collections.Counter(f"{p['kind']}/{p['care']}/{p['need']}" for p in allp)),
          "chosen_with_urgent_line": urg_end,
          "rejected_with_urgent_line_on_non_urgent": sum(1 for p in allp if p["kind"] == "false_urgency"),
          "urgent_lines_used": [s for s, _ in lines], "notes": dict(stats)}
    json.dump(st, open(f"{OUT}/curation_stats.json", "w"), indent=1, ensure_ascii=False)
    md = ["# Curated DPO v4c (run four, attempt c): CuratorKIT report", "", __doc__.strip(), "", "```",
          json.dumps(st, indent=1, ensure_ascii=False), "```"]
    open(f"{OUT}/curation_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in st.items() if k != "urgent_lines_used"}, indent=1, ensure_ascii=False))
    print("lines:", len(lines))


if __name__ == "__main__":
    main()
