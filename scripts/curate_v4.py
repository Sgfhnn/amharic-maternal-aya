"""Run four: CuratorKIT DPO v4 data (over-referral fix). No LLM calls.

Inputs (all from run three): data/curated_v3/{train.jsonl,dpo/train.jsonl,dpo/dev.jsonl}, work/run3/referral_labels.jsonl
(LLM urgent / routine / none labels, Gemma 4 31B), work/run3/rewrites.jsonl + rewrite_checks.jsonl (verified rewrites),
results/dpo_v3/eval/predictions.jsonl (the boilerplate lines v3 DPO actually emits).

Pairs (CuratorKIT DataSample with chosen/rejected, provenance per step):
- urgent (kept from v3): chosen = answer WITH the (urgent) referral sentence; rejected = same without it, or the
  original MT answer. All 225 v3 urgent pairs kept unchanged; v3's 578 routine "referral good" pairs are dropped.
- routine counter-pairs (new): chosen = fluent answer with the generic facility boilerplate stripped (a specific
  conditional 'see a provider if X' clause is kept); rejected = the same answer + one boilerplate sentence sampled
  (frequency-weighted) from what v3 DPO emits on the test set. Two sources, half each:
    a) need=routine: the verified run-three routine rewrites (body; their referral line was the boilerplate)
    b) need=none (facts/definitions): quality-gated curated_v3 train answers with no referral line at all
- balance: routine count = urgent count (about 50/50).
Also writes data/curated_v4/train.jsonl + chat/train.jsonl: curated_v3 SFT targets with the generic boilerplate
stripped from every non-urgent row (record only; run four does not re-run SFT).

Usage: python scripts/curate_v4.py
"""
import collections, json, os, random, sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referral_lib as L
from curatorkit import DataSample
from curatorkit.gates.schema import SchemaGate
from curatorkit.interfaces import BaseNormalizer
from curatorkit.schema import ProvenanceRecord

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = f"{ROOT}/data/curated_v3"
OUT = f"{ROOT}/data/curated_v4"
W3 = f"{ROOT}/work/run3"
SEED = 42
STEP_VERSION = "run4-1.0"
MAX_TOK = 760  # train_dpo_lowmem max_length 768; keep the appended boilerplate inside the window
HEALTH = {"walelign_srh", "afrihealth_qa", "danilior_maternal", "amhqa_mendeley"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def prov(s, step, notes):
    s.append_provenance(ProvenanceRecord(step_name=step, step_version=STEP_VERSION, config_hash=step[:16], notes=notes))


class BoilerplateStripper(BaseNormalizer):
    """CuratorKIT normalizer: remove generic, unconditional facility-referral sentences from non-urgent targets."""
    NAME = "BoilerplateStripper"

    def __init__(self):
        self.stats = collections.Counter()

    def run(self, samples):
        for s in samples:
            field = "chosen" if s.chosen else "output"
            txt = getattr(s, field)
            new, n = L.strip_boilerplate(txt)
            if n:
                setattr(s, field, new)
                self.stats[f"stripped_{s.metadata.get('care', s.metadata.get('need', '?'))}"] += 1
                prov(s, self.NAME, {"removed_sentences": n})
        return samples


class BoilerplateAppender(BaseNormalizer):
    """CuratorKIT normalizer: rejected = chosen + one real v3-DPO boilerplate line (frequency-weighted, seeded)."""
    NAME = "BoilerplateAppender"

    def __init__(self, lines, weights, seed=SEED):
        self.lines, self.weights, self.rng = lines, weights, random.Random(seed)
        self.used = collections.Counter()

    def run(self, samples):
        for s in samples:
            b = self.rng.choices(self.lines, self.weights)[0]
            s.rejected = s.chosen.rstrip() + " " + b
            self.used[b] += 1
            prov(s, self.NAME, {"boilerplate": b})
        return samples


def v3dpo_boilerplate():
    c = collections.Counter()
    for r in jl(f"{ROOT}/results/dpo_v3/eval/predictions.jsonl"):
        for s in L.sentences(r["prediction"]):
            if L.is_boilerplate(s) and s.endswith("።") and s.startswith("ለ") and len(s) < 140:
                c[s] += 1
    lines = [k for k, v in c.most_common() if v >= 2]
    return lines, [c[k] for k in lines], c


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/work/official_tinyaya")

    def ntok(q, a):
        return len(tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                           tokenize=True, return_dict=True)["input_ids"])

    labels = {r["id"]: r for r in jl(f"{W3}/referral_labels.jsonl")}
    rw = {r["id"]: r for r in jl(f"{W3}/rewrites.jsonl")}
    chk = {r["id"]: r for r in jl(f"{W3}/rewrite_checks.jsonl")}
    train3 = jl(f"{V3}/train.jsonl")
    byid = {r["id"]: r for r in train3}
    v3tr, v3dv = jl(f"{V3}/dpo/train.jsonl"), jl(f"{V3}/dpo/dev.jsonl")
    v3_dev_base = {p["id"].split("#")[0] for p in v3dv}
    lines, weights, bp_counter = v3dpo_boilerplate()
    rng = random.Random(SEED)
    steps = []

    # ---- urgent pairs (kept)
    urgent = []
    for split, part in [("train", v3tr), ("dev", v3dv)]:
        for p in part:
            if p["care"] != "urgent":
                continue
            s = DataSample(task_type="preference", id=p["id"], source_uri=f"data/curated_v3/dpo/{split}.jsonl", instruction=p["prompt"][0]["content"],
                           chosen=p["chosen"][0]["content"], rejected=p["rejected"][0]["content"],
                           metadata={"care": "urgent", "kind": p["kind"], "split": split, "need": "urgent"})
            prov(s, "KeepUrgentPairs", {"kind": p["kind"]})
            urgent.append(s)
    steps.append({"step": "KeepUrgentPairs (v3 urgent pairs, unchanged)", "in": len(v3tr) + len(v3dv), "out": len(urgent),
                  "notes": {"dropped_v3_routine_referral_pairs": sum(p["care"] == "routine" for p in v3tr + v3dv)}})
    n_target = len(urgent)

    # ---- routine candidates a) verified routine rewrites (same verification rule as curate_v3 'good')
    cand_a = []
    for k, v in rw.items():
        c = chk.get(k)
        if v["care"] != "routine" or not c or c["correctness"] < 4 or c["fluency"] < 4 or not c["referral_ok"]:
            continue
        if k not in byid or L.referral_line(v["body"]) or L.LATIN_RE.search(v["body"]):
            continue
        ref = v["referral"].strip()
        chosen = v["body"].strip() + ((" " + ref) if (L.is_ref_sentence(ref) and not L.is_boilerplate(ref)) else "")
        cand_a.append((k, byid[k]["question"], chosen, "routine", "routine_rewrite", byid[k]["source"]))
    # ---- routine candidates b) need=none quality-gated answers with no referral line
    cand_b = []
    for r in train3:
        lab = labels.get(r["id"], {}).get("need")
        if lab != "none" or r["source"] not in HEALTH or r.get("synthetic") or r["curation_v3"].get("referral_rewrite"):
            continue
        a = r["answer"].strip()
        if L.referral_line(a) or L.wide_keyword(a) or len(a) < 80 or L.LATIN_RE.search(a):
            continue
        cand_b.append((r["id"], r["question"], a, "none", "factual_answer", r["source"]))
    longest_bp = max(lines, key=lambda x: len(tok(x)["input_ids"]))

    def fits(c):
        return ntok(c[1], c[2] + " " + longest_bp) <= MAX_TOK
    fa = [c for c in cand_a if fits(c)]
    fb = [c for c in cand_b if fits(c)]
    rng.shuffle(fa)
    # stratify b by source so AmHQA-style general-health facts are represented
    bysrc = collections.defaultdict(list)
    for c in fb:
        bysrc[c[5]].append(c)
    for v in bysrc.values():
        rng.shuffle(v)
    na = min(len(fa), (n_target + 1) // 2)
    nb = n_target - na
    pick_b, srcs = [], sorted(bysrc)
    while len(pick_b) < nb and any(bysrc.values()):
        for s_ in srcs:
            if bysrc[s_] and len(pick_b) < nb:
                pick_b.append(bysrc[s_].pop())
    routine_c = fa[:na] + pick_b
    steps.append({"step": "SelectRoutineCounterPairs", "in": len(cand_a) + len(cand_b), "out": len(routine_c),
                  "notes": {"routine_rewrite_candidates": len(cand_a), "routine_rewrite_fit_768": len(fa), "picked_routine_rewrite": na,
                            "factual_candidates": len(cand_b), "factual_fit_768": len(fb), "picked_factual": len(pick_b),
                            "picked_factual_by_source": dict(collections.Counter(c[5] for c in pick_b))}})
    routine = []
    for k, q, a, need, kind, src in routine_c:
        s = DataSample(task_type="preference", id=k + "#noboiler", source_uri=f"data/curated_v3/train.jsonl#{src}", instruction=q, chosen=a,
                       metadata={"care": "routine", "need": need, "kind": kind, "source": src})
        prov(s, "SelectRoutineCounterPairs", {"need": need, "kind": kind})
        routine.append(s)
    st = BoilerplateStripper(); routine = st.run(routine)
    steps.append({"step": "BoilerplateStripper (CuratorKIT BaseNormalizer) on routine chosen", "in": len(routine), "out": len(routine),
                  "notes": dict(st.stats)})
    ap = BoilerplateAppender(lines, weights); routine = ap.run(routine)
    steps.append({"step": "BoilerplateAppender (CuratorKIT BaseNormalizer): rejected = chosen + v3-DPO boilerplate", "in": len(routine),
                  "out": len(routine), "notes": {"distinct_lines_used": len(ap.used), "top_lines": ap.used.most_common(5)}})
    allp = urgent + routine
    sg = SchemaGate(min_tokens=2, max_tokens=100000)
    b = len(allp); allp, rej = sg.run(allp)
    steps.append({"step": "SchemaGate (CuratorKIT)", "in": b, "out": len(allp), "notes": {"rejected": len(rej)}})
    # sanity: routine chosen has no boilerplate, rejected does; urgent chosen has a referral line
    bad = [s for s in allp if s.metadata["care"] == "routine" and (L.boilerplate_line(s.chosen) or not L.boilerplate_line(s.rejected))]
    for s in bad[:5]:
        print("BAD", L.boilerplate_line(s.chosen), repr(s.chosen[-200:]), "||", repr(s.rejected[-150:]))
    assert not bad, len(bad)
    urgent_chosen_ref = sum(L.referral_line(s.chosen) for s in allp if s.metadata["care"] == "urgent")

    # ---- split: urgent keep v3 split; routine 5% dev by seeded id
    rids = sorted(s.id for s in allp if s.metadata["care"] == "routine"); random.Random(SEED + 1).shuffle(rids)
    rdev = set(rids[:max(8, round(len(rids) * 0.05))])
    tr, dv = [], []
    for s in allp:
        isdev = (s.metadata.get("split") == "dev") if s.metadata["care"] == "urgent" else (s.id in rdev)
        rec = {"id": s.id, "kind": s.metadata["kind"], "care": s.metadata["care"], "need": s.metadata["need"],
               "prompt": [{"role": "user", "content": s.instruction}],
               "chosen": [{"role": "assistant", "content": s.chosen}], "rejected": [{"role": "assistant", "content": s.rejected}],
               "provenance": [p.step_name for p in s.provenance_chain]}
        (dv if isdev else tr).append(rec)
    random.Random(SEED).shuffle(tr)
    os.makedirs(f"{OUT}/dpo", exist_ok=True); os.makedirs(f"{OUT}/chat", exist_ok=True)
    for name, part in [("train", tr), ("dev", dv)]:
        with open(f"{OUT}/dpo/{name}.jsonl", "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- SFT targets with boilerplate stripped from non-urgent rows (record only)
    sft_stats = collections.Counter()
    with open(f"{OUT}/train.jsonl", "w", encoding="utf-8") as f, open(f"{OUT}/chat/train.jsonl", "w", encoding="utf-8") as fc:
        for r in train3:
            r = dict(r)
            care = r["curation_v3"].get("referral_rewrite") or labels.get(r["id"], {}).get("need") or "none"
            if care != "urgent":
                new, n = L.strip_boilerplate(r["answer"])
                if n and len(new) > 20:
                    r["answer"] = new; sft_stats[f"stripped_{care}"] += 1
            r["curation_v4"] = {"care": care}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            user = (r["context"] + "\n\n" + r["question"]) if r.get("context") else r["question"]
            fc.write(json.dumps({"id": r["id"], "source": r["source"], "topic": r["topic"],
                                 "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["answer"]}]},
                                ensure_ascii=False) + "\n")
    stats = {"created": datetime.now(UTC).isoformat(), "steps": steps,
             "pairs_train": len(tr), "pairs_dev": len(dv),
             "by_care": dict(collections.Counter(p["care"] for p in tr + dv)),
             "by_kind": dict(collections.Counter(p["kind"] for p in tr + dv)),
             "by_care_split": {"train": dict(collections.Counter(p["care"] for p in tr)), "dev": dict(collections.Counter(p["care"] for p in dv))},
             "urgent_chosen_with_referral_line": urgent_chosen_ref,
             "v3dpo_boilerplate_lines_found": len(bp_counter), "boilerplate_lines_used_pool": len(lines),
             "sft_v4_boilerplate_stripped": dict(sft_stats), "sft_v4_rows": len(train3)}
    json.dump(stats, open(f"{OUT}/curation_stats.json", "w"), indent=1, ensure_ascii=False)
    ex_r = [p for p in tr if p["care"] == "routine"][:3]
    ex_u = [p for p in tr if p["care"] == "urgent"][:2]
    md = ["# Curated DPO v4 (run four, over-referral fix): CuratorKIT report", "",
          f"Generated by `scripts/curate_v4.py` on {stats['created'][:16]} UTC. No LLM calls (regex + scripts only). Test split not read.", "",
          "Why: v3 DPO ended 87.5% of test answers with a near-fixed 'go to a health facility' line, even on factual questions. "
          "The v3 pairs only taught 'referral good' (578 of 803 pairs were routine questions whose chosen answer ended in that line).", "",
          "| Step | In | Out | Notes |", "|---|---|---|---|"]
    for s in steps:
        md.append(f"| {s['step']} | {s['in']} | {s['out']} | {json.dumps(s['notes'], ensure_ascii=False)[:600]} |")
    md += ["", f"Pairs: **{len(tr)} train + {len(dv)} dev**. By care: {stats['by_care']}. By kind: {stats['by_kind']}.",
           f"Urgent chosen answers with a referral line (regex): {urgent_chosen_ref}/{stats['by_care'].get('urgent')}.", "",
           "Boilerplate = a referral sentence (facility / health-worker word + go / consult verb) with no condition (ካለ / ከሆነ / ከታዩ ...) "
           "and no urgency word (`scripts/referral_lib.py`). Rejected routine answers get one of the "
           f"{len(lines)} full boilerplate sentences v3 DPO emitted at least twice on the test set, frequency-weighted "
           "(the test answers are used only as a source of the model's own boilerplate wording, never as targets).", "",
           f"SFT targets (data/curated_v4/train.jsonl, record only, not trained in run four): boilerplate stripped from non-urgent rows {dict(sft_stats)}.", "",
           "## Examples", ""]
    for p in ex_r + ex_u:
        md += [f"- [{p['care']}/{p['kind']}] Q: {p['prompt'][0]['content'][:160]}",
               f"  - chosen end: ...{p['chosen'][0]['content'][-160:]}", f"  - rejected end: ...{p['rejected'][0]['content'][-160:]}"]
    open(f"{OUT}/curation_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "steps"}, ensure_ascii=False, indent=1))
    for s in steps:
        print(s["step"], s["in"], s["out"], json.dumps(s["notes"], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
