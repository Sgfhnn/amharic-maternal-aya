"""Run four: greedy generation (same settings as eval_lib gen: chat template, max 256 new tokens) on a custom question
file, e.g. the held-out urgent probe. Usage: python probe_gen.py --questions probe_urgent_v4.jsonl --model DIR [--adapter DIR] --out FILE"""
import argparse, json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_h = glob.glob("/kaggle/input/**/eval_lib.py", recursive=True)
if _h:
    sys.path.insert(0, os.path.dirname(_h[0]))
import eval_lib as E

ap = argparse.ArgumentParser()
ap.add_argument("--questions", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(E.find(a.questions), encoding="utf-8")]
model, tok, dt = E.load(a.model, a.adapter)
outs = E.generate(model, tok, [r["question"] for r in rows], 256, 16)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"id": r["id"], "source": r["source"], "topic": r["topic"], "prompt": r["question"], "reference": "",
                            "prediction": o["text"], "n_tokens": o["n_tokens"], "stopped": o["stopped"], "adapter": a.adapter,
                            "dtype": dt}, ensure_ascii=False) + "\n")
E.log("probe done", a.out, len(rows))
