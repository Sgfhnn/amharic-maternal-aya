"""Run three, CuratorKIT step 1: LLM quality scores for every curated train (and dev) answer.

Scores each row 1-5 for Amharic naturalness (fluency) and correctness (medical correctness for health rows;
task faithfulness for the ethionlp_general rows), batched 8 rows per call on HF Inference Providers.
Cached per row id in work/run3/quality_scores.jsonl, so reruns cost nothing.

Usage: python scripts/llm_quality_score.py --split train [--model google/gemma-4-31B-it:deepinfra] [--limit N]
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_client as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = """You are a strict native Amharic editor and an Ethiopian public-health physician. You review training data for an Amharic maternal, sexual and reproductive health assistant.
For each item you get an id, a QUESTION (or task instruction) and an ANSWER. Score the ANSWER:
- fluency (1-5): does it read like natural Amharic written by a native speaker? 5 = natural, idiomatic, grammatical; 4 = natural with small slips; 3 = understandable but stiff or literal (word-by-word machine translation, odd word choices, calques from English); 2 = awkward/garbled in places; 1 = broken, mostly another language, or repetitive.
- correctness (1-5): for health questions, is the medical content accurate, safe and does it answer the question? For non-health tasks (translation, title, summary, story), is it a faithful and complete response to the task? 5 = fully correct, 3 = partly right or incomplete, 1 = wrong, harmful or unrelated.
Be critical: machine-translated text usually deserves fluency 3 or less.
Reply with JSON only, one entry per id: {"<id>": {"fluency": int, "correctness": int}, ...}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="google/gemma-4-31B-it:deepinfra")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cache", default=f"{ROOT}/work/run3/quality_scores.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{ROOT}/data/curated/{a.split}.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    cache = {}
    if os.path.exists(a.cache):
        for l in open(a.cache, encoding="utf-8"):
            j = json.loads(l)
            if j["model"] == a.model:
                cache[j["id"]] = j
    todo = [r for r in rows if r["id"] not in cache]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print("rows", len(rows), "cached", len(rows) - len(todo), "batches", len(batches), "spent so far $%.4f" % H.spent_usd(), flush=True)
    stop = {"why": None}

    def one(b):
        if stop["why"]:
            return []
        items = []
        for k, r in enumerate(b):
            q = r["question"] + (("\n[context] " + r["context"][:600]) if r.get("context") else "")
            items.append({"id": f"r{k}", "QUESTION": q[:1500], "ANSWER": r["answer"][:2500]})
        try:
            txt, _ = H.chat(a.model, [{"role": "system", "content": SYS},
                                      {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                            max_tokens=60 * len(b) + 200, tag=f"quality_{a.split}")
            res = H.parse_json(txt)
        except H.BillingError as e:
            stop["why"] = str(e); return []
        except Exception as e:
            print("batch failed", repr(e)[:200], flush=True); return []
        out = []
        for k, r in enumerate(b):
            s = res.get(f"r{k}")
            try:
                out.append({"id": r["id"], "model": a.model, "fluency": int(s["fluency"]), "correctness": int(s["correctness"])})
            except Exception:
                pass
        return out

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(a.cache, "a", encoding="utf-8") as f:
        for out in ex.map(one, batches):
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n"); n += 1
            f.flush()
    print("scored", n, "stop", stop["why"], "spent total $%.4f" % H.spent_usd(), flush=True)


if __name__ == "__main__":
    main()
