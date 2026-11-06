"""Run four: label the 550 test questions urgent / routine / none for the over-referral audit.

Same triage prompt as the run-three train labels (curate_v3.CLS_SYS); model deepseek-ai/DeepSeek-V4.1-Flash:deepinfra
(HF Inference Providers, the run-three judge path). Only the QUESTION is shown (no reference, no model answer).
Cached in results/referral_audit/test_care_labels.jsonl; reruns cost nothing.
Usage: HF_CAP_USD=1.00 python scripts/label_test_care.py
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_client as H
from curate_v3 import CLS_SYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra"
OUT = f"{ROOT}/results/referral_audit/test_care_labels.jsonl"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = [json.loads(l) for l in open(f"{ROOT}/data/test.jsonl", encoding="utf-8")]
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            j = json.loads(l); done[j["id"]] = j
    todo = [r for r in rows if r["id"] not in done]
    s0 = H.spent_usd()
    print("test rows", len(rows), "todo", len(todo), "logged HF spend before $%.4f" % s0, flush=True)

    def fn(b):
        items = [{"id": f"q{k}", "QUESTION": r["question"][:600]} for k, r in enumerate(b)]
        try:
            txt, _ = H.chat(MODEL, [{"role": "system", "content": CLS_SYS},
                                    {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                            max_tokens=3000, tag="run4_label_test")
            res = H.parse_json(txt)
        except H.BillingError:
            raise
        except Exception as e:
            print("batch failed", repr(e)[:200], flush=True); return []
        out = []
        for k, r in enumerate(b):
            s = res.get(f"q{k}") or {}
            if s.get("need") in ("urgent", "routine", "none"):
                out.append({"id": r["id"], "source": r["source"], "need": s["need"], "sign": s.get("sign", ""), "model": MODEL})
        return out
    batches = [todo[i:i + 20] for i in range(0, len(todo), 20)]
    with ThreadPoolExecutor(8) as ex, open(OUT, "a", encoding="utf-8") as f:
        for out in ex.map(fn, batches):
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n = sum(1 for _ in open(OUT))
    print("labelled", n, "/", len(rows), "this run logged $%.4f" % (H.spent_usd() - s0), flush=True)


if __name__ == "__main__":
    main()
