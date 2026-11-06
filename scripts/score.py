"""Score predictions.jsonl (from eval_lib.py) -> metrics.json.

Corpus-level (sacrebleu 2.x): chrF++ (word_order=2), chrF, BLEU (13a), spBLEU (flores200 SPM).
Per-sample (AuditKIT 1.1.3, ak.evaluate with a callable that replays the saved predictions):
  chrf (char 6-gram, beta=1, i.e. NOT chrF++), bleu (whitespace, smoothed sentence BLEU), rouge_l, f1_score.
Plus output diagnostics (length, Ethiopic-script share, stop rate, health-worker referral keyword rate).

Usage: python score.py RESULTS_DIR   (reads RESULTS_DIR/predictions.jsonl [+ mcq_metrics.json])
"""
import json, re, sys, unicodedata
from collections import defaultdict

import sacrebleu

REFERRAL = re.compile(r"(ሐኪም|ሀኪም|ሃኪም|የጤና ባለሙያ|ጤና ባለሙያ|የጤና ተቋም|ጤና ጣቢያ|ሆስፒታል|ክሊኒክ|ዶክተር)")


def eth_share(t):
    L = [c for c in t if unicodedata.category(c).startswith("L")]
    return sum(0x1200 <= ord(c) <= 0x137F for c in L) / len(L) if L else 0.0


def corpus(preds, refs):
    out = {"n": len(preds),
           "chrf_pp": sacrebleu.corpus_chrf(preds, [refs], word_order=2).score,
           "chrf": sacrebleu.corpus_chrf(preds, [refs]).score,
           "bleu_13a": sacrebleu.corpus_bleu(preds, [refs]).score}
    try:
        out["spbleu_flores200"] = sacrebleu.corpus_bleu(preds, [refs], tokenize="flores200").score
    except Exception as e:  # SPM model download can fail offline
        out["spbleu_flores200_error"] = str(e)[:200]
    return out


def auditkit_scores(rows):
    import auditkit as ak
    pred = {r["prompt"]: r["prediction"] for r in rows}
    samples = [ak.Sample(input=r["prompt"], target=r["reference"]) for r in rows]
    res = ak.evaluate(samples, lambda prompts: [pred[p] for p in prompts], ["chrf", "bleu", "rouge_l", "f1_score"])
    return {"headline": {k: float(v) for k, v in dict(res.headline).items()}, "fingerprint": getattr(res, "fingerprint", None),
            "errors": [str(e)[:200] for e in (res.errors or [])][:5]}


def main(d):
    rows = [json.loads(l) for l in open(f"{d}/predictions.jsonl", encoding="utf-8")]
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    m = {"overall": corpus(preds, refs)}
    for key in ("source", "topic"):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        m[f"by_{key}"] = {k: corpus([r["prediction"] for r in v], [r["reference"] for r in v]) for k, v in sorted(g.items())}
    m["diagnostics"] = {
        "avg_pred_chars": sum(map(len, preds)) / len(preds),
        "avg_ref_chars": sum(map(len, refs)) / len(refs),
        "avg_pred_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
        "stopped_before_max_tokens": sum(r["stopped"] for r in rows) / len(rows),
        "empty_predictions": sum(1 for p in preds if not p.strip()),
        "avg_ethiopic_share": sum(map(eth_share, preds)) / len(preds),
        "referral_keyword_rate_pred": sum(bool(REFERRAL.search(p)) for p in preds) / len(preds),
        "referral_keyword_rate_ref": sum(bool(REFERRAL.search(p)) for p in refs) / len(refs),
        "markdown_rate_pred": sum(bool(re.search(r"(\*\*|^#+ |^\s*[-*] )", p, re.M)) for p in preds) / len(preds),
    }
    try:
        m["auditkit_per_sample_mean"] = auditkit_scores(rows)
    except Exception as e:
        m["auditkit_error"] = repr(e)[:500]
    try:
        m["mcq"] = json.load(open(f"{d}/mcq_metrics.json"))
    except FileNotFoundError:
        pass
    json.dump(m, open(f"{d}/metrics.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"overall": m["overall"], "auditkit": m.get("auditkit_per_sample_mean", m.get("auditkit_error")),
                      "mcq": {k: v["accuracy"] for k, v in m.get("mcq", {}).items()}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
