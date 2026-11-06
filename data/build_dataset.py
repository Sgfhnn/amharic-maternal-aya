"""Build merged, cleaned Amharic health instruction dataset (local only).
Run: python build_dataset.py   (from the data/ directory; raw files in raw/)
"""
import json, re, hashlib, unicodedata, random, os
from collections import Counter, defaultdict
import pandas as pd

SEED = 42
RAW = "raw"
OUT = "."
rng = random.Random(SEED)

# v2 (2026-11-05): kaggle_afri_enhanced (LLM-synthetic, Markdown-heavy, CC BY-SA) is EXCLUDED from
# train/dev. Its rows are still loaded and pass through dedup exactly as in v1 so that the random
# stream (and therefore the 550-row test split and the EthioNLP slice) stays byte-identical to v1.
EXCLUDE_SOURCES = {"kaggle_afri_enhanced"}

LIC = {
    "walelign_srh": "CC BY-NC 4.0 (Walelign/Amharic_SRH_Dataset)",
    "afrihealth_qa": "CC BY 4.0 (ImhotepSystems/AfriHealth-QA)",
    "danilior_maternal": "unspecified (danilior/amharic_maternal_health; no license on card)",
    "amhqa_mendeley": "CC BY 4.0 (Bogale et al. 2026, Mendeley Data 10.17632/8cks7m5f8s.1)",
    "kaggle_afri_enhanced": "CC BY-SA 4.0 (Kaggle dokster/health-qa-in-low-resource-african-languages)",
    "ethionlp_general": "unspecified on card (EthioNLP/Amharic_Instruction_dataset, Walia-LLM SFT mix of upstream datasets)",
}

MATERNAL_KW = ["እርግዝና", "እርጉዝ", "ነፍሰ ጡር", "ነብሰ ጡር", "ወሊድ", "መውለድ", "ምጥ", "አራስ", "ጨቅላ",
               "ጡት ማጥባት", "ጡት ማጥባ", "ፅንስ", "ጽንስ", "ድህረ ወሊድ", "ቅድመ ወሊድ", "እንግዴ", "ማህፀን ውስጥ ያለ",
               "የሚያጠቡ", "ህፃን", "ሕፃን", "ቄሳራዊ", "እናቶች"]


def norm_ws(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"(?<=[\u1200-\u137F]) ?(::|\u1361\u1361)", "\u1362", s)  # '::' / '፡፡' typed for '።'
    s = re.sub(r"[ \t\r\f\v\u00a0]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def ethiopic_share(text):
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    eth = sum(1 for c in letters if 0x1200 <= ord(c) <= 0x137F)
    return eth / len(letters)


# Fold common Amharic homophone letters so spelling variants hash the same (near-dup key)
FOLD = {}
def _fold_series(src, dst):
    for i in range(8):
        FOLD[chr(ord(src) + i)] = chr(ord(dst) + i)
for a, b in [("ሐ", "ሀ"), ("ኀ", "ሀ"), ("ሠ", "ሰ"), ("ዐ", "አ"), ("ፀ", "ጸ")]:
    _fold_series(a, b)


def key(text):
    t = unicodedata.normalize("NFC", text).lower()
    t = "".join(FOLD.get(c, c) for c in t)
    t = "".join(c for c in t if unicodedata.category(c)[0] in "LN")  # drop punct/space/symbols
    return hashlib.md5(t.encode()).hexdigest()


def topic_for(src, q, a):
    text = q + " " + a
    if src == "ethionlp_general":
        return "general"
    if src == "danilior_maternal":
        return "maternal"
    is_mat = any(k in text for k in MATERNAL_KW)
    if src in ("walelign_srh", "afrihealth_qa", "kaggle_afri_enhanced"):
        return "maternal" if is_mat else "srh"
    return "maternal" if is_mat else "general_health"


rows = []
raw_counts = {}

def add(src, sid, q, a, ctx="", split=None):
    rows.append(dict(src_id=str(sid), source=src, question=q, answer=a, context=ctx, split_hint=split))

# 1 Walelign SRH
d = pd.read_csv(f"{RAW}/srh.csv"); raw_counts["walelign_srh"] = len(d)
for r in d.itertuples():
    add("walelign_srh", r.Qes_ID, r.Question_Amh, r.Answer_Amh)

# 2 AfriHealth (Amh_Eth only, labelled splits only; test_holdout has no answers)
n = 0
for sp in ["train", "validation"]:
    d = pd.read_parquet(f"{RAW}/afri_{sp}.parquet")
    d = d[d.language_country == "Amh_Eth"]; n += len(d)
    for r in d.itertuples():
        add("afrihealth_qa", r.id, r.question, r.answer)
raw_counts["afrihealth_qa"] = n

# 4 danilior
d = pd.read_csv(f"{RAW}/danilior.csv"); raw_counts["danilior_maternal"] = len(d)
for i, r in enumerate(d.itertuples(index=False)):
    add("danilior_maternal", i, r[0], r[1])

# 5 AmHQA (single xlsx, 2051 rows, no split column)
d = pd.read_excel(f"{RAW}/amhqa.xlsx")
d.columns = [c.strip() for c in d.columns]
raw_counts["amhqa_mendeley"] = len(d)
for i, r in d.iterrows():
    add("amhqa_mendeley", i, r["ጥያቄ"], r["መልስ"], ctx="")

# 6 Kaggle: input/output == AfriHealth Amh rows (same IDs) -> only the LLM-enhanced pairs are new
d = pd.read_csv(f"{RAW}/kaggle_amh.csv"); raw_counts["kaggle_afri_enhanced"] = len(d)
for r in d.itertuples():
    add("kaggle_afri_enhanced", r.ID + "_enh", r.enhanced_prompt, r.enhanced_completion)

# Optional general slice
d = pd.read_parquet(f"{RAW}/ethionlp_train.parquet")
# spellcheck (near copy of input) and reverse summarisation (invent an article from a summary) excluded:
# low value / hallucination-prone for a health assistant
KEEP_DS = ["amharicqa", "xlsum_summerization", "amharic_title_generation",
           "amharic_story_generation", "amharic_poem_generation", "amharic_mt eng-amh"]
d = d[d.datasource.isin(KEEP_DS)]
HEALTH_KW = ["ጤና", "ህክምና", "ሕክምና", "በሽታ", "ሐኪም", "ሀኪም", "መድሃኒት", "መድኃኒት", "ክትባት", "ሆስፒታል", "ወረርሽኝ"]
txt = (d.instruction.fillna("") + " " + d.input.fillna("") + " " + d.output.fillna(""))
health = d[txt.apply(lambda t: any(k in t for k in HEALTH_KW))]
other = d.drop(health.index)
cand = pd.concat([health.sample(min(len(health), 2500), random_state=SEED),
                  other.sample(2500, random_state=SEED)])
raw_counts["ethionlp_general"] = f"{len(cand)} candidates sampled from {len(d)} rows ({len(health)} health-keyword rows)"
eth_rows = []
for r in cand.itertuples():
    eth_rows.append(dict(src_id=f"{r.datasource}:{r.Index}", source="ethionlp_general", question=r.instruction,
                         answer=r.output, context=r.input if isinstance(r.input, str) else "", split_hint=None,
                         ethio_ds=r.datasource, is_health=r.Index in health.index))

# ---------------- cleaning ----------------
drops = defaultdict(Counter)

def clean(rowlist):
    out = []
    for r in rowlist:
        for f in ("question", "answer", "context"):
            r[f] = norm_ws(r[f])
        s = r["source"]
        if not r["question"] or not r["answer"]:
            drops[s]["empty"] += 1; continue
        if len(r["answer"]) < 15:
            drops[s]["short_answer_lt15"] += 1; continue
        if ethiopic_share(r["question"] + " " + r["answer"]) < 0.5:
            drops[s]["not_mostly_geez"] += 1; continue
        out.append(r)
    return out

health_rows = clean(rows)
eth_clean = clean(eth_rows)

PRIORITY = ["amhqa_mendeley", "walelign_srh", "afrihealth_qa", "danilior_maternal", "kaggle_afri_enhanced", "ethionlp_general"]

def dedup(rowlist, seen_pair, seen_q):
    out = []
    for r in sorted(rowlist, key=lambda r: PRIORITY.index(r["source"])):
        pk = key(r["context"] + "|" + r["question"] + "|" + r["answer"])
        qk = key(r["context"] + "|" + r["question"])
        if pk in seen_pair:
            drops[r["source"]]["dup_exact_or_near"] += 1; continue
        if qk in seen_q:
            drops[r["source"]]["dup_question"] += 1; continue
        seen_pair.add(pk); seen_q.add(qk)
        r["qkey"] = key(r["question"])
        out.append(r)
    return out

seen_pair, seen_q = set(), set()
health_rows = dedup(health_rows, seen_pair, seen_q)

# general slice: prefer health-keyword rows, cap 3000
eth_clean = dedup(eth_clean, seen_pair, seen_q)
rng.shuffle(eth_clean)
eth_clean.sort(key=lambda r: not r["is_health"])
eth_final = eth_clean[:3000]

# ---------------- splits ----------------
# Test: 400 AmHQA (seeded; the Mendeley file ships as one sheet, no train/test marker)
#     + 150 Walelign SRH rows as in-domain substitute for the unavailable HanaMekonen set.
by_src = defaultdict(list)
for r in health_rows:
    by_src[r["source"]].append(r)
for s in by_src:
    rng.shuffle(by_src[s])
test = by_src["amhqa_mendeley"][:400] + by_src["walelign_srh"][:150]
train_pool = by_src["amhqa_mendeley"][400:] + by_src["walelign_srh"][150:]
for s in ["afrihealth_qa", "danilior_maternal", "kaggle_afri_enhanced"]:
    if s in EXCLUDE_SOURCES:
        drops[s]["excluded_source_v2"] += len(by_src[s]); continue
    train_pool += by_src[s]
train_pool += eth_final

test_q = {r["qkey"] for r in test}
leak = [r for r in train_pool if r["qkey"] in test_q]
for r in leak:
    drops[r["source"]]["leak_question_in_test"] += 1
train_pool = [r for r in train_pool if r["qkey"] not in test_q]

rng.shuffle(train_pool)
n_dev = round(0.05 * len(train_pool))
dev, train = train_pool[:n_dev], train_pool[n_dev:]

def finalize(rs, split):
    out = []
    for i, r in enumerate(rs):
        out.append(dict(
            id=f"{split}-{i:05d}-{r['source']}-{r['src_id']}",
            source=r["source"], license=LIC[r["source"]],
            question=r["question"], answer=r["answer"], context=r["context"],
            topic=topic_for(r["source"], r["question"], r["answer"]),
            split=split,
            synthetic=r["source"] == "kaggle_afri_enhanced",
        ))
    return out

splits = {"train": finalize(train, "train"), "dev": finalize(dev, "dev"), "test": finalize(test, "test")}

def to_chat(r):
    user = (r["context"] + "\n\n" + r["question"]) if r["context"] else r["question"]
    return {"id": r["id"], "source": r["source"], "topic": r["topic"],
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["answer"]}]}

os.makedirs(f"{OUT}/chat", exist_ok=True)
for sp, rs in splits.items():
    with open(f"{OUT}/{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/chat/{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")

# ---------------- stats ----------------
stats = {"seed": SEED, "raw_rows_per_source": raw_counts,
         "after_cleaning_dedup_per_source": Counter(r["source"] for rs in splits.values() for r in rs),
         "drops_per_source": {k: dict(v) for k, v in drops.items()},
         "per_split": {}, "licenses": {k: v for k, v in LIC.items() if k not in EXCLUDE_SOURCES},
         "excluded_sources": {k: LIC[k] for k in EXCLUDE_SOURCES}}
for sp, rs in splits.items():
    st = {"rows": len(rs), "by_source": Counter(r["source"] for r in rs), "by_topic": Counter(r["topic"] for r in rs),
          "avg_question_chars": round(sum(len(r["question"]) for r in rs) / max(1, len(rs)), 1),
          "avg_answer_chars": round(sum(len(r["answer"]) for r in rs) / max(1, len(rs)), 1),
          "avg_answer_chars_by_source": {s: round(sum(len(r["answer"]) for r in rs if r["source"] == s) /
                                              max(1, sum(1 for r in rs if r["source"] == s)), 1)
                                         for s in set(r["source"] for r in rs)}}
    stats["per_split"][sp] = st
stats["ethionlp_general_by_datasource"] = Counter(r["ethio_ds"] for r in eth_final)
stats["ethionlp_general_health_keyword_rows"] = sum(r["is_health"] for r in eth_final)
with open(f"{OUT}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=dict)
print(json.dumps(stats, ensure_ascii=False, indent=1, default=dict))
