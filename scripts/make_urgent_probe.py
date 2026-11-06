"""Run four: held-out URGENT probe set for the over-referral audit.

Why: the 550-row test set has only 1 question the LLM labels urgent (346+101 none, 102 routine), so urgent-question
referral cannot be measured on it. This writes ~60-70 new Amharic danger-sign questions (same 14 categories and prompt
as the run-three synthetic set, curate_v3.SYN_SYS, Gemma 4 31B at temperature 0.9), drops any with Latin letters or
that duplicate / near-duplicate a training question, and keeps those the DeepSeek-V4.1-Flash triage (same CLS_SYS as
the test labels) calls urgent. Never trained on. Output: data/probe_urgent_v4.jsonl (eval_lib test.jsonl format).
Usage: HF_CAP_USD=1.00 python scripts/make_urgent_probe.py
"""
import difflib, json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_client as H
from curate_v3 import CLS_SYS, SYN_CATS, SYN_SYS, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = "google/gemma-4-31B-it:deepinfra"
JUDGE = "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra"
OUT = f"{ROOT}/data/probe_urgent_v4.jsonl"
RAW = f"{ROOT}/work/run4_probe_raw.jsonl"


def main():
    import re
    latin = re.compile(r"[A-Za-z]")
    train_q = [json.loads(l)["question"] for f in ["data/curated_v3/train.jsonl", "data/train.jsonl", "data/test.jsonl"]
               for l in open(f"{ROOT}/{f}", encoding="utf-8")]
    train_q += [json.loads(l)["question"] for l in open(f"{ROOT}/work/run3/synth_questions.jsonl", encoding="utf-8")]
    train_n = [norm(q) for q in train_q]
    s0 = H.spent_usd()
    if not os.path.exists(RAW):
        def fn(item):
            cat, topic = item
            txt, _ = H.chat(GEN, [{"role": "user", "content": SYN_SYS.format(n=6, topic=topic) +
                                   "\nMake them different from typical textbook questions; vary who is asking."}],
                            max_tokens=900, temperature=0.9, tag="run4_probe_gen")
            return [{"cat": cat, "question": q.strip()} for q in H.parse_json(txt) if isinstance(q, str)]
        with ThreadPoolExecutor(7) as ex:
            qs = [q for out in ex.map(fn, SYN_CATS.items()) for q in out]
        with open(RAW, "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
    qs = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    def grams(t):
        return {t[i:i + 3] for i in range(max(1, len(t) - 2))}
    G = {}

    def near(a, b):  # 3-gram Jaccard prefilter, then difflib ratio > 0.8
        ga = G.setdefault(a, grams(a)); gb = G.setdefault(b, grams(b))
        if len(ga & gb) / max(1, len(ga | gb)) < 0.4:
            return False
        return difflib.SequenceMatcher(None, a, b).ratio() > 0.8
    kept, seen = [], []
    drop = {"latin": 0, "dup": 0}
    for q in qs:
        n = norm(q["question"])
        if latin.search(q["question"]):
            drop["latin"] += 1; continue
        if n in train_n or any(near(n, t) for t in train_n + seen):
            drop["dup"] += 1; continue
        seen.append(n); kept.append(q)
    # triage with the same prompt as the test labels
    lab = {}
    for i in range(0, len(kept), 20):
        b = kept[i:i + 20]
        items = [{"id": f"q{k}", "QUESTION": r["question"]} for k, r in enumerate(b)]
        txt, _ = H.chat(JUDGE, [{"role": "system", "content": CLS_SYS}, {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                        max_tokens=3000, tag="run4_probe_label")
        res = H.parse_json(txt)
        for k, r in enumerate(b):
            lab[i + k] = (res.get(f"q{k}") or {}).get("need")
    rows = []
    for i, q in enumerate(kept):
        if lab.get(i) == "urgent":
            rows.append({"id": f"probe-urgent-{len(rows):03d}-{q['cat']}", "source": "probe_urgent", "topic": "maternal",
                         "question": q["question"], "answer": "", "context": "", "cat": q["cat"], "need": "urgent"})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"generated": len(qs), "dropped": drop, "after_dedup": len(kept),
                      "labels": {k: list(lab.values()).count(k) for k in ("urgent", "routine", "none", None)},
                      "probe_urgent": len(rows), "logged_usd_this_run": round(H.spent_usd() - s0, 4)}))


if __name__ == "__main__":
    main()
