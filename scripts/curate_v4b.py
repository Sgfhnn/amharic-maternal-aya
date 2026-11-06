"""Run four, attempt b: CuratorKIT DPO v4b data. No LLM calls.

Why: v4 (data/curated_v4) removed the boilerplate (non-urgent referral line 86.5% -> 4.7%) but the judge scored it
2.68 / 3.76 vs v3 DPO 3.28 / 4.31, and 40.7% of its answers hit the 256-token cap in repetition loops (v3 SFT 35.3%,
v3 DPO 14.0%). v4 dropped v3's 280 routine 'fluent rewrite vs original MT answer' pairs (their chosen ended in the
boilerplate) and has only 27 optimiser steps, so it lost v3 DPO's fluency / stopping signal.

v4b = v4 pairs (225 urgent + 225 routine counter-pairs, unchanged)
    + fluency pairs: all 323 v3 'original_mt_answer' pairs; for the 280 routine ones the chosen rewrite has the generic
      boilerplate stripped (BoilerplateStripper), rejected = the original MT answer with any boilerplate stripped too,
      so the pair no longer carries a referral signal; the 43 urgent ones are unchanged (chosen keeps the urgent line)
    + anti-repetition pairs: for a seeded half of the urgent and of the routine chosen answers, rejected = the same
      answer + its last body sentence repeated 3 times (the loop failure mode), so stopping after the answer is preferred.
Referral-signal pairs stay 50/50 urgent / routine; fluency and anti-repetition pairs are split the same way as far as
the data allows. Usage: python scripts/curate_v4b.py
"""
import collections, json, os, random, sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referral_lib as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3, V4, OUT = f"{ROOT}/data/curated_v3", f"{ROOT}/data/curated_v4", f"{ROOT}/data/curated_v4b"
SEED = 7
MAX_TOK = 760


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def txt(m):
    return m[0]["content"]


def mk(id_, kind, care, need, prompt, chosen, rejected, prov):
    return {"id": id_, "kind": kind, "care": care, "need": need, "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}], "rejected": [{"role": "assistant", "content": rejected}],
            "provenance": prov}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/work/official_tinyaya")

    def ntok(p, a):
        return len(tok.apply_chat_template(p + [{"role": "assistant", "content": a}], tokenize=True, return_dict=True)["input_ids"])

    v4 = {"train": jl(f"{V4}/dpo/train.jsonl"), "dev": jl(f"{V4}/dpo/dev.jsonl")}
    dev_base = {p["id"].split("#")[0] for p in v4["dev"]}
    v3 = jl(f"{V3}/dpo/train.jsonl") + jl(f"{V3}/dpo/dev.jsonl")
    v3_dev_base = {p["id"].split("#")[0] for p in jl(f"{V3}/dpo/dev.jsonl")}
    rng = random.Random(SEED)
    stats = collections.Counter()
    fluency = []
    for p in v3:
        if p["kind"] != "original_mt_answer":
            continue
        ch, rj = txt(p["chosen"]), txt(p["rejected"])
        steps = ["v3 original_mt_answer pair"]
        if p["care"] == "routine":
            ch, n1 = L.strip_boilerplate(ch)
            rj, n2 = L.strip_boilerplate(rj)
            stats["fluency_routine_chosen_stripped"] += bool(n1); stats["fluency_routine_rejected_stripped"] += bool(n2)
            steps.append("BoilerplateStripper")
            if L.referral_line(ch) and not L.referral_line(rj):
                stats["fluency_routine_skipped_ref_asym"] += 1
                continue
        if ch.strip() == rj.strip() or len(ch) < 40:
            stats["fluency_skipped_identical"] += 1
            continue
        fluency.append(mk(p["id"].replace("#mt", "#mt4b"), "fluency_vs_mt", p["care"], p["care"], p["prompt"], ch, rj, steps))
    # anti-repetition pairs from v4 chosen answers (half urgent, half routine)
    anti = []
    for care in ("urgent", "routine"):
        pool = [p for p in v4["train"] + v4["dev"] if p["care"] == care and p["kind"] in ("minus_referral", "routine_rewrite", "factual_answer")]
        rng.shuffle(pool)
        for p in pool[:len(pool) // 2]:
            ch = txt(p["chosen"])
            body = [s for s in L.sentences(ch) if not L.is_ref_sentence(s)]
            if not body:
                continue
            loop = " ".join([body[-1]] * 3)
            rj = ch.rstrip() + " " + loop
            if ntok(p["prompt"], rj) > MAX_TOK + 60 or ntok(p["prompt"], ch) > MAX_TOK - 8:
                stats["anti_skipped_long"] += 1
                continue
            anti.append(mk(p["id"].split("#")[0] + "#antirep", "anti_repetition", care, p.get("need", care), p["prompt"], ch, rj,
                           ["v4 chosen", "AntiRepetitionAppender"]))
    allp = v4["train"] + v4["dev"] + fluency + anti
    tr, dv = [], []
    for p in allp:
        b = p["id"].split("#")[0]
        isdev = b in dev_base or (p["kind"] == "fluency_vs_mt" and b in v3_dev_base)
        (dv if isdev else tr).append(p)
    random.Random(42).shuffle(tr)
    os.makedirs(f"{OUT}/dpo", exist_ok=True)
    for name, part in [("train", tr), ("dev", dv)]:
        with open(f"{OUT}/dpo/{name}.jsonl", "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    st = {"created": datetime.now(UTC).isoformat(), "pairs_train": len(tr), "pairs_dev": len(dv),
          "by_kind": dict(collections.Counter(p["kind"] for p in allp)),
          "by_kind_care": dict(collections.Counter(f"{p['kind']}/{p['care']}" for p in allp)),
          "referral_signal_pairs_by_care": dict(collections.Counter(p["care"] for p in allp if p["kind"] in ("minus_referral", "original_mt_answer", "routine_rewrite", "factual_answer"))),
          "notes": dict(stats)}
    json.dump(st, open(f"{OUT}/curation_stats.json", "w"), indent=1)
    md = ["# Curated DPO v4b (run four, attempt b): CuratorKIT report", "", __doc__.strip(), "", "```", json.dumps(st, indent=1), "```"]
    open(f"{OUT}/curation_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
