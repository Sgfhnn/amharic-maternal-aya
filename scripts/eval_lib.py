"""Generation + multiple-choice evaluation for Tiny Aya on the Amharic health test set.

Runs on GPU (Kaggle T4). Produces raw predictions only; text metrics (chrF++, BLEU,
AuditKIT per-sample metrics) are computed afterwards by scripts/score.py so that
scoring is identical for "before" and "after".

Usage (inside a kernel):
  python eval_lib.py --task gen  --model <hf id or dir> [--adapter DIR] --out OUTDIR
  python eval_lib.py --task mcq  --model <hf id or dir> [--adapter DIR] --out OUTDIR
"""
import argparse, ast, csv, glob, json, os, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "1-800-LLMs/tiny-aya-earth"  # ungated mirror; weight files' git-LFS oids == CohereLabs/tiny-aya-earth


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def stop_ids(tok):
    ids = {tok.eos_token_id, 3}
    for t in ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"]:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def probe_ok(model, tok):
    msgs = [{"role": "user", "content": "ነፍሰ ጡር ሴት ምን አይነት ምግብ መመገብ አለባት?"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()
    finite = bool(torch.isfinite(logits).all())
    out = model.generate(**enc, max_new_tokens=24, do_sample=False, eos_token_id=stop_ids(tok), pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    log("probe finite:", finite, "sample:", repr(txt[:120]))
    return finite and (len(txt.strip()) > 0 or os.environ.get("EVAL_TEST_MODE") == "1")


def load(model_id, adapter=None, prefer="fp16", multi_gpu=False):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    dtypes = [torch.float16, torch.float32] if prefer == "fp16" else [torch.float32]
    for dt in dtypes:
        dm = "auto" if (multi_gpu or dt == torch.float32) else ("cpu" if not torch.cuda.is_available() else {"": 0})
        log("loading", model_id, dt, "device_map", dm)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, device_map=dm)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
            log("merged adapter", adapter)
        model.eval()
        if probe_ok(model, tok):
            return model, tok, str(dt)
        log("probe failed for", dt, "- trying next dtype")
        del model
        torch.cuda.empty_cache()
    raise RuntimeError("model produced non-finite / empty output in all dtypes")


def generate(model, tok, prompts, max_new=256, bs=16):
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out = [None] * len(prompts)
    sids = stop_ids(tok)
    t0 = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        convs = [[{"role": "user", "content": prompts[i]}] for i in idx]
        enc = tok.apply_chat_template(convs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=sids,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            ids = new[j].tolist()
            hit_stop = any(t in sids for t in ids)
            out[i] = {"text": tok.decode(ids, skip_special_tokens=True).strip(),
                      "n_tokens": int(sum(1 for t in ids if t != tok.pad_token_id)), "stopped": hit_stop}
        log(f"gen {min(b + bs, len(order))}/{len(order)} {time.time() - t0:.0f}s")
    return out


def letter_ids(tok, letters):
    res = []
    for L in letters:
        c = set()
        for v in (L, " " + L):
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) >= 1:
                c.add(ids[-1] if v.startswith(" ") and len(ids) > 1 else ids[0])
        res.append(sorted(c))
    return res


def mcq_scores(model, tok, prompts, n_opt=4, bs=8):
    letters = "ABCD"[:n_opt]
    lid = letter_ids(tok, letters)
    preds = []
    for b in range(0, len(prompts), bs):
        convs = [[{"role": "user", "content": p}] for p in prompts[b:b + bs]]
        enc = tok.apply_chat_template(convs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      padding=True).to(model.device)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1].float()
        for row in lg:
            sc = [max(row[i].item() for i in ids) for ids in lid]
            preds.append(letters[max(range(n_opt), key=lambda k: sc[k])])
    return preds


def belebele_items():
    items = [json.loads(l) for l in open(find("belebele_amh.jsonl"), encoding="utf-8")]
    prompts, gold = [], []
    for r in items:
        p = (f"{r['flores_passage']}\n\nQuestion: {r['question']}\n"
             f"A. {r['mc_answer1']}\nB. {r['mc_answer2']}\nC. {r['mc_answer3']}\nD. {r['mc_answer4']}\n\n"
             "Answer with only the letter (A, B, C, or D) of the correct answer.")
        prompts.append(p)
        gold.append("ABCD"[int(r["correct_answer_num"]) - 1])
    return prompts, gold


def afrimmlu_items():
    rows = list(csv.DictReader(open(find("afrimmlu_amh_test.tsv"), encoding="utf-8"), delimiter="\t"))
    prompts, gold, subj = [], [], []
    for r in rows:
        ch = ast.literal_eval(r["choices"])
        p = (f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n{r['question']}\n"
             + "".join(f"{'ABCD'[k]}. {c}\n" for k, c in enumerate(ch[:4]))
             + "\nAnswer with only the letter (A, B, C, or D) of the correct answer.")
        prompts.append(p); gold.append(r["answer"].strip()); subj.append(r["subject"])
    return prompts, gold, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gen", "mcq", "all"], default="all")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--multi_gpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tok, dt = load(a.model, a.adapter, multi_gpu=a.multi_gpu)
    meta = {"model": a.model, "adapter": a.adapter, "dtype": dt, "decoding": "greedy", "max_new_tokens": a.max_new,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    import transformers; meta["transformers"] = transformers.__version__
    if a.task in ("gen", "all"):
        rows = [json.loads(l) for l in open(find("test.jsonl"), encoding="utf-8")]
        if a.limit: rows = rows[:a.limit]
        prompts = [(r["context"] + "\n\n" + r["question"]) if r.get("context") else r["question"] for r in rows]
        t = time.time()
        outs = generate(model, tok, prompts, a.max_new, a.bs)
        meta["gen_seconds"] = round(time.time() - t, 1)
        with open(f"{a.out}/predictions.jsonl", "w", encoding="utf-8") as f:
            for r, p, o in zip(rows, prompts, outs):
                f.write(json.dumps({"id": r["id"], "source": r["source"], "topic": r["topic"], "prompt": p,
                                    "reference": r["answer"], "prediction": o["text"], "n_tokens": o["n_tokens"],
                                    "stopped": o["stopped"]}, ensure_ascii=False) + "\n")
    if a.task in ("mcq", "all"):
        res = {}
        t = time.time()
        p, g = belebele_items()
        if a.limit: p, g = p[:a.limit], g[:a.limit]
        pr = mcq_scores(model, tok, p, bs=max(1, a.bs // 4))
        res["belebele_amh_Ethi"] = {"n": len(g), "accuracy": sum(x == y for x, y in zip(pr, g)) / len(g),
                                    "pred_dist": {L: pr.count(L) for L in "ABCD"}}
        json.dump({"pred": pr, "gold": g}, open(f"{a.out}/belebele_preds.json", "w"))
        log("belebele", res["belebele_amh_Ethi"])
        p, g, s = afrimmlu_items()
        if a.limit: p, g, s = p[:a.limit], g[:a.limit], s[:a.limit]
        pr = mcq_scores(model, tok, p, bs=max(1, a.bs // 2))
        res["afrimmlu_amh"] = {"n": len(g), "accuracy": sum(x == y for x, y in zip(pr, g)) / len(g),
                               "pred_dist": {L: pr.count(L) for L in "ABCD"}}
        json.dump({"pred": pr, "gold": g, "subject": s}, open(f"{a.out}/afrimmlu_preds.json", "w"))
        log("afrimmlu", res["afrimmlu_amh"])
        meta["mcq_seconds"] = round(time.time() - t, 1)
        meta["mcq_method"] = ("zero-shot, chat template, English instruction; argmax over next-token logits of "
                              "letters A-D (with/without leading space) after <|START_RESPONSE|>")
        json.dump(res, open(f"{a.out}/mcq_metrics.json", "w"), indent=2)
    json.dump(meta, open(f"{a.out}/run_meta_{a.task}.json", "w"), indent=2)
    log("done", meta)


if __name__ == "__main__":
    main()
