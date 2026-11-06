"""Run-two data curation with CuratorKIT (Lexsi Labs) -> data/curated/.

Input: data/train.jsonl, data/dev.jsonl (dataset v2). The test split is never read or written.
Every row becomes a curatorkit.DataSample and flows through CuratorKIT steps:

  1  TextCleaner            (CuratorKIT)  NFC, control chars, whitespace (newlines kept)
  2  ExactDeduplicator      (CuratorKIT)  exact duplicate rows
  3  MinHashDeduplicator    (CuratorKIT)  near duplicates, Jaccard >= 0.9 on char 3-grams
  4  BrokenTextGate         (custom BaseGate) U+FFFD/mojibake, long repeated-char runs, "not in the given context"
                                           answers, e-mail/URL-only answers, very long unspaced Ethiopic runs
  5  EnglishHeavyGate       (custom BaseGate) Latin letters > 20% of answer letters (or > 30% of question letters
                                           for non-translation rows)
  6  OffTopicGate           (custom BaseGate) SRH/health rows about clinical-trial reporting, study statistics,
                                           insurance/billing, bullying, etc.: keyword candidates + LLM confirm
  7  QAMismatchGate         (custom BaseGate) question/answer mismatch: char-3gram TF-IDF cosine + acronym
                                           conflict -> candidates, confirmed by an LLM (gemini-2.5-flash-lite via
                                           the Cline API, free promo credit). Without the LLM, only the
                                           strongest lexical signal is used.
  8  EthioNLPDownsampler    (custom BaseNormalizer) ethionlp_general -> ~1,000 train rows (dev scaled the same),
                                           stratified by sub-task, preferring rows whose prompt fits max_len 1024
  9  DangerSignReferral     (custom BaseNormalizer) maternal danger-sign questions: make sure the answer ends with
                                           an Amharic line telling the reader to go to a health worker / facility now
 10  SchemaGate             (CuratorKIT)  required fields, min 2 whitespace tokens

Outputs: data/curated/{train,dev}.jsonl, data/curated/chat/{train,dev}.jsonl, data/curated/rejected.jsonl,
         data/curated/curation_report.md, data/curated/curation_stats.json, data/curated/llm_checks.jsonl
Usage: python scripts/curate.py [--no-llm]
"""
import argparse, collections, hashlib, json, math, os, random, re, sys, time
from datetime import UTC, datetime

from curatorkit import DataSample, RejectedSample
from curatorkit.gates.schema import SchemaGate
from curatorkit.interfaces import BaseGate, BaseNormalizer
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import ProvenanceRecord

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data"
OUT = f"{DATA}/curated"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REFERRAL = re.compile(r"(ሐኪም|ሀኪም|ሃኪም|የጤና ባለሙያ|ጤና ባለሙያ|የጤና ተቋም|ጤና ተቋም|ጤና ጣቢያ|ሆስፒታል|ክሊኒክ|ዶክተር)")
URGENT = re.compile(r"(ወዲያውኑ|አሁኑኑ|በአስቸኳይ|አስቸኳይ|በፍጥነት|ቶሎ|አፋጣኝ|ሳይዘገዩ|ሳትዘገይ|ሳይዘገይ|ድንገተኛ)")
REFERRAL_LINE = "ይህ የአደጋ ምልክት ሊሆን ስለሚችል አሁኑኑ ወደ ጤና ባለሙያ ወይም በአቅራቢያዎ ወዳለ ጤና ተቋም ይሂዱ።"
STEP_VERSION = "run2-1.0"


def prov(sample, step, notes):
    sample.append_provenance(ProvenanceRecord(step_name=step, step_version=STEP_VERSION, config_hash=step[:16], notes=notes))


def letters(t):
    return [c for c in t if c.isalpha()]


def latin_share(t):
    L = letters(t)
    return sum(c.isascii() for c in L) / len(L) if L else 0.0


def reject(sample, reason, step):
    r = RejectedSample(**sample.model_dump(), rejection_reason=reason, rejecting_step=step)
    return r


# ----------------------------------------------------------------------------- gates
class BrokenTextGate(BaseGate):
    NAME = "BrokenTextGate"
    PATS = [
        ("replacement_char", re.compile("\ufffd|Ã.|â€")),
        ("repeated_char_run", re.compile(r"(.)\1{7,}")),
        ("context_not_given", re.compile(r"በተሰጠው (አውድ|ጽሑፍ|ፅሁፍ|ጽሁፍ|መረጃ|ጽሑፍ)\S* ?(ውስጥ )?(ላይ )?(አልተ|አይገኝ|አልተገለጸ)")),
        ("unspaced_run", re.compile(r"[\u1200-\u137F]{40,}")),
        ("unfilled_template", re.compile(r"\{(title|text|input|question|topic)\}")),
        ("answer_label_only", re.compile(r"^\s*(መልስ|Answer)\s*[:፡]?\s*$")),
    ]

    def run(self, samples):
        ok, bad = [], []
        for s in samples:
            reason = None
            for name, p in self.PATS:
                field = s.output if name in ("context_not_given", "answer_label_only") else s.input + " " + s.instruction + " " + s.output
                if p.search(field):
                    reason = f"broken_text:{name}"; break
            if reason is None and len(s.output.strip()) < 15:
                reason = "broken_text:too_short"
            (bad.append(reject(s, reason, self.NAME)) if reason else (prov(s, self.NAME, {"passed": True}), ok.append(s)))
        return ok, bad


URL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+|www\.\S+")


class URLSentenceStripper(BaseGate):
    """Remove sentences that carry an e-mail address or URL (mostly US agency links in AmHQA answers);
    reject the row if little is left."""
    NAME = "URLSentenceStripper"

    def run(self, samples):
        ok, bad = [], []
        self.stats = {"rows_with_url_or_email": 0, "rows_rejected_after_strip": 0}
        for s in samples:
            if not URL_RE.search(s.output):
                ok.append(s); continue
            self.stats["rows_with_url_or_email"] += 1
            parts = re.split(r"(?<=[።!?])\s*", s.output)
            kept = " ".join(p for p in parts if p.strip() and not URL_RE.search(p)).strip()
            if len(kept) < 30:
                self.stats["rows_rejected_after_strip"] += 1
                bad.append(reject(s, "broken_text:email_or_url_only", self.NAME)); continue
            s.output = kept
            prov(s, self.NAME, {"url_sentences_removed": True}); ok.append(s)
        return ok, bad


class EnglishHeavyGate(BaseGate):
    NAME = "EnglishHeavyGate"

    def run(self, samples):
        ok, bad = [], []
        for s in samples:
            sub = s.metadata.get("subtask", "")
            a, qa = latin_share(s.output), latin_share(s.instruction + " " + s.output)
            reason = None
            if a > 0.30:
                reason = f"english_heavy:answer_latin={a:.2f}"
            elif qa > 0.25 and "eng-amh" not in sub and "mt" not in sub:
                reason = f"english_heavy:row_latin={qa:.2f}"
            (bad.append(reject(s, reason, self.NAME)) if reason else (prov(s, self.NAME, {"passed": True}), ok.append(s)))
        return ok, bad


OFFTOPIC_KW = re.compile(
    r"ክሊኒካዊ ሙከራ|ክሊኒካዊ ጥናት|አሉታዊ ክስተ|ተሳታፊዎች|በጥናቱ|ጥናቱ|የዳሰሳ ጥናት|ስፖንሰር|መርማሪ|ሜዲኬር|ሜዲኬይድ|ኢንሹራንስ|"
    r"ጉልበተኝነት|ሳይበር|አርቴፊሻል ኢንተለጀንስ|ማሽን ትምህርት|ቴክኖሎጂ|ፖሊሲ|በመቶኛ|መቶኛ|ስታቲስቲክ|ሪፖርት|ስንት ሴቶች|ስንት ሰዎች|"
    r"ወጪ|ያስከፍላል|ክፍያ|ካንሰር ካውንስል|ምርምር")
HEALTH_SOURCES = {"walelign_srh", "afrihealth_qa", "danilior_maternal", "amhqa_mendeley"}


def _ngr(t, n=3):
    t = re.sub(r"\s+", " ", t)
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]


ACR = re.compile(r"\b[A-Z][A-Za-z0-9-]{1,9}\b")


class LLMChecker:
    """Batched LLM confirmation through the Cline API (OpenAI-compatible)."""

    def __init__(self, enabled, model="google/gemini-2.5-flash-lite", log_path=None, budget_micro_usd=45000):
        self.enabled, self.model, self.log_path, self.budget = enabled, model, log_path, budget_micro_usd
        self.spent, self.calls, self.error = 0.0, 0, None
        if enabled:
            import llm_client
            self.L = llm_client

    def ask(self, kind, items):
        """items: list of (id, question, answer). Returns {id: verdict dict} or None per id if unavailable."""
        if not self.enabled or self.error:
            return {}
        out = {}
        sysmsg = {
            "mismatch": ("You audit an Amharic health QA dataset. For each item decide whether the ANSWER actually "
                         "answers the QUESTION (same disease/topic and responsive to what is asked). Reply with JSON "
                         "list: [{\"id\":..., \"match\": true|false, \"why\": \"<=12 English words\"}]. "
                         "Mark false only when the answer is about a different subject, a different question, or "
                         "is unrelated; partial or imperfect answers on the same topic are true."),
            "offtopic": ("You audit an Amharic maternal / sexual & reproductive health (SRH) QA dataset meant to "
                         "train a health assistant for the public in Ethiopia. For each item decide if it is "
                         "OFF-TOPIC for that purpose: clinical-trial or research reporting, statistics about a "
                         "specific study, insurance/billing/Medicare, non-health topics (bullying, technology, "
                         "politics), or questions that only make sense with a missing document. Health advice, "
                         "SRH, maternal, newborn, STI, contraception, gender-based violence support and myths "
                         "are ON-topic. Reply with JSON list: [{\"id\":..., \"off_topic\": true|false, "
                         "\"why\": \"<=12 English words\"}]."),
        }[kind]
        cache_p = f"{ROOT}/work/llm_cache_{kind}.json"
        cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
        todo = [it for it in items if it[0] not in cache]
        chunks = [todo[i:i + 20] for i in range(0, len(todo), 20)]

        def one(chunk):
            payload = [{"id": k, "question": q[:400], "answer": a[:500]} for k, q, a in chunk]
            try:
                txt, usage = self.L.chat(self.model, [{"role": "system", "content": sysmsg},
                                                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                         max_tokens=2000)
                return {str(r.get("id")): r for r in self.L.parse_json(txt)}, float((usage or {}).get("cost", 0)) * 1e6, None
            except self.L.BillingError as e:
                return {}, 0.0, f"billing: {e}"
            except Exception as e:
                return {}, 0.0, f"chunk failed: {e!r}"[:200]

        from concurrent.futures import ThreadPoolExecutor
        est = len(todo) * 400 * 0.1 + len(chunks) * 600 * 0.4  # rough micro-USD
        if self.spent + est > self.budget:
            self.error = f"estimated cost {est:.0f} micro-USD over budget"; chunks = []
        with ThreadPoolExecutor(8) as ex:
            for res, cost, err in ex.map(one, chunks):
                self.calls += 1; self.spent += cost; cache.update(res)
                if err:
                    print("llm:", err, flush=True)
                    if err.startswith("billing"):
                        self.error = err
        json.dump(cache, open(cache_p, "w"), ensure_ascii=False)
        for k, q, a in items:
            if k in cache:
                out[k] = cache[k]
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for k, q, a in items:
                    f.write(json.dumps({"kind": kind, "id": k, "question": q[:200], "answer": a[:200],
                                        "verdict": out.get(k)}, ensure_ascii=False) + "\n")
        return out


class OffTopicGate(BaseGate):
    NAME = "OffTopicGate"

    def __init__(self, llm):
        self.llm = llm
        self.stats = {}

    def run(self, samples):
        cands = [s for s in samples if s.metadata["source"] in HEALTH_SOURCES and s.metadata["source"] != "amhqa_mendeley"
                 and OFFTOPIC_KW.search(s.instruction + " " + s.output)]
        verdict = self.llm.ask("offtopic", [(s.id, s.instruction, s.output) for s in cands])
        drop = {}
        for s in cands:
            v = verdict.get(s.id)
            if v is not None and v.get("off_topic") is True:
                drop[s.id] = f"off_topic:llm:{str(v.get('why', ''))[:60]}"
        self.stats = {"keyword_candidates": len(cands), "llm_judged": len(verdict), "dropped": len(drop),
                      "llm_used": bool(verdict)}
        ok, bad = [], []
        for s in samples:
            if s.id in drop:
                bad.append(reject(s, drop[s.id], self.NAME))
            else:
                prov(s, self.NAME, {"passed": True}); ok.append(s)
        return ok, bad


class QAMismatchGate(BaseGate):
    NAME = "QAMismatchGate"

    def __init__(self, llm, quantile=0.12):
        self.llm, self.quantile, self.stats = llm, quantile, {}

    def run(self, samples):
        from sklearn.feature_extraction.text import TfidfVectorizer
        hs = [s for s in samples if s.metadata["source"] in HEALTH_SOURCES]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        vec.fit([s.instruction for s in hs] + [s.output for s in hs])
        Q = vec.transform([s.instruction for s in hs]); A = vec.transform([s.output for s in hs])
        sims = (Q.multiply(A)).sum(axis=1).A1
        by_src = collections.defaultdict(list)
        for s, v in zip(hs, sims):
            s.metadata["qa_cos"] = round(float(v), 4); by_src[s.metadata["source"]].append(v)
        thr = {k: sorted(v)[int(len(v) * self.quantile)] for k, v in by_src.items()}
        cands, strong = [], set()
        for s in hs:
            qa = set(ACR.findall(s.instruction)) - {"A", "I"}
            aa = set(ACR.findall(s.output))
            acr_conflict = bool(qa) and bool(aa) and not (qa & aa)
            low = s.metadata["qa_cos"] <= thr[s.metadata["source"]]
            if low or acr_conflict:
                cands.append(s)
                if acr_conflict and s.metadata["qa_cos"] <= thr[s.metadata["source"]] * 1.5:
                    strong.add(s.id)
        verdict = self.llm.ask("mismatch", [(s.id, s.instruction, s.output) for s in cands])
        drop = {}
        for s in cands:
            v = verdict.get(s.id)
            if v is not None:
                if v.get("match") is False:
                    drop[s.id] = f"qa_mismatch:llm:{str(v.get('why', ''))[:60]}"
            elif not verdict and s.id in strong:  # no LLM at all: lexical-only fallback
                drop[s.id] = "qa_mismatch:lexical(acronym conflict + low cosine)"
        self.stats = {"health_rows": len(hs), "cosine_thresholds_by_source": {k: round(float(v), 4) for k, v in thr.items()},
                      "candidates": len(cands), "acronym_low_cos_candidates": len(strong), "llm_judged": len(verdict),
                      "dropped": len(drop), "llm_used": bool(verdict)}
        ok, bad = [], []
        for s in samples:
            if s.id in drop:
                bad.append(reject(s, drop[s.id], self.NAME))
            else:
                prov(s, self.NAME, {"passed": True, "qa_cos": s.metadata.get("qa_cos")}); ok.append(s)
        return ok, bad


# ----------------------------------------------------------------------------- normalizers
class EthioNLPDownsampler(BaseNormalizer):
    NAME = "EthioNLPDownsampler"

    def __init__(self, target, prompt_budget_tokens, tok=None, seed=42):
        self.target, self.budget, self.tok, self.seed = target, prompt_budget_tokens, tok, seed
        self.stats = {}

    def _plen(self, s):
        txt = (s.input + "\n\n" + s.instruction) if s.input else s.instruction
        return len(self.tok.encode(txt, add_special_tokens=False).ids) if self.tok else len(txt) // 2

    def run(self, samples):
        rng = random.Random(self.seed)
        eth = [s for s in samples if s.metadata["source"] == "ethionlp_general"]
        rest = [s for s in samples if s.metadata["source"] != "ethionlp_general"]
        fits = [s for s in eth if self._plen(s) <= self.budget]
        groups = collections.defaultdict(list)
        for s in fits:
            groups[s.metadata.get("subtask", "?")].append(s)
        total = sum(len(v) for v in groups.values())
        kept = []
        for k, v in sorted(groups.items()):
            n = round(self.target * len(v) / total) if total else 0
            rng.shuffle(v); kept.extend(v[:n])
        for s in kept:
            prov(s, self.NAME, {"target": self.target})
        self.stats = {"ethionlp_in": len(eth), "fit_prompt_budget": len(fits), "kept": len(kept),
                      "by_subtask_kept": dict(collections.Counter(s.metadata.get("subtask", "?") for s in kept)),
                      "dropped_too_long_prompt": len(eth) - len(fits), "dropped_by_sampling": len(fits) - len(kept)}
        return rest + kept


PREG = re.compile(r"(እርግዝና|እርጉዝ|ነፍሰ ጡር|ነፍሰጡር|ጽንስ|ፅንስ|(?<![\u1200-\u137F])ሽል|(?<![\u1200-\u137F])[በከየ]?ምጥ|ወሊድ|መውለድ|ወለድ|አራስ|እንግዴ|ማርገዝ|አርግዤ|ጸንስ)")
POSTPARTUM = re.compile(r"(ድህረ ወሊድ|ከወሊድ በኋላ|ከወለድኩ|ከወለደች|አራስ እናት)")
LABOUR_DANGER = re.compile(r"(?<![\u1200-\u137F])[በከየ]?ምጥ.{0,25}(እያጋጠመ|ላይ መሆኔን|ተራዘመ|ረጅም|ችግር|ከባድ|ቆየ)|(ቅድመ ወሊድ ምጥ|ያለጊዜው ምጥ|ከባድ ምጥ|የተራዘመ ምጥ)")
DANGER = re.compile(
    r"(ደም (መፍሰስ|ይፈስ|መፍሰሱ|ፈሰሰ|ቢፈስ)|የደም መፍሰስ|ትኩሳት|ራስ ምታት|ኤክላምፕሲያ|ኤክላምሲያ|ኤክላምፕሲ|ፕሪኤክላምፕሲያ|ፕሪ-ኤክላምፕሲያ|"
    r"ፕሬ-ኤክላምፕሲያ|የሚጥል|መንቀጥቀጥ|ማንቀጥቀጥ|እንቅስቃሴ.{0,20}(ቀነሰ|መቀነስ|አቆመ|ማቆም|የለም|አልተሰማ)|(ቀነሰ|መቀነስ).{0,12}እንቅስቃሴ|"
    r"የእንሽርት ውሃ|ውሃ(ው)? (መፍሰስ|ፈሰሰ)|የዓይን ብዥታ|ብዥ ያለ|ከባድ (ህመም|ሕመም)|የሆድ ህመም|የሆድ ሕመም|የፊት እብጠት|እብጠት|"
    r"ችግር ካጋጠመ|መተንፈስ)")
PP_PROBLEM = re.compile(r"(ደም|ትኩሳት|ህመም|ሕመም|ኢንፌክሽን|ጭንቀት|ድብርት|መንፈስ|ችግር|ፈሳሽ|ሽታ|እብጠት|ማዞር)")


def is_danger_sign(q):
    if LABOUR_DANGER.search(q):
        return True
    if PREG.search(q) and DANGER.search(q):
        return True
    return bool(POSTPARTUM.search(q) and PP_PROBLEM.search(q))


def ends_with_referral(a):
    sents = [x for x in re.split(r"(?<=[።!?])\s*|\n+", a.strip()) if x.strip()]
    last = sents[-1] if sents else ""
    return bool(REFERRAL.search(last) and URGENT.search(last))


class DangerSignReferral(BaseNormalizer):
    NAME = "DangerSignReferral"

    def __init__(self):
        self.stats, self.examples = {}, []

    def run(self, samples):
        n_d = n_ok = n_add = 0
        bysrc = collections.Counter()
        for s in samples:
            if s.metadata["source"] == "ethionlp_general" or not is_danger_sign(s.instruction):
                continue
            n_d += 1
            if ends_with_referral(s.output):
                n_ok += 1; continue
            a = s.output.rstrip()
            if a and a[-1] not in "።!?.":
                a += "።"
            s.output = a + " " + REFERRAL_LINE
            n_add += 1; bysrc[s.metadata["source"]] += 1
            s.metadata["referral_added"] = True
            prov(s, self.NAME, {"referral_added": True})
            if len(self.examples) < 400:
                self.examples.append(s.instruction[:140])
        self.stats = {"danger_sign_questions": n_d, "already_ended_with_urgent_referral": n_ok,
                      "referral_line_added": n_add, "added_by_source": dict(bysrc), "line": REFERRAL_LINE}
        return samples


# ----------------------------------------------------------------------------- main
def load(split):
    out = []
    for l in open(f"{DATA}/{split}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        sub = ""
        if r["source"] == "ethionlp_general":
            m = re.match(r".*?ethionlp_general-(.+?):", r["id"])
            sub = m.group(1) if m else "?"
        out.append(DataSample(id=r["id"], source_uri=f"data/{split}.jsonl#{r['source']}", instruction=r["question"],
                              input=r.get("context") or "", output=r["answer"],
                              metadata={**r, "source": r["source"], "subtask": sub}))
    return out


def to_row(s):
    r = dict(s.metadata)
    r = {k: r[k] for k in ["id", "source", "license", "question", "answer", "context", "topic", "split", "synthetic"] if k in r}
    r["question"], r["context"], r["answer"] = s.instruction, s.input, s.output
    r["curation"] = {"qa_cos": s.metadata.get("qa_cos"), "referral_added": bool(s.metadata.get("referral_added")),
                     "steps": [p.step_name for p in s.provenance_chain]}
    return r


def to_chat(r):
    user = (r["context"] + "\n\n" + r["question"]) if r.get("context") else r["question"]
    return {"id": r["id"], "source": r["source"], "topic": r["topic"],
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["answer"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--eth_target", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/chat", exist_ok=True)
    tok = None
    try:
        from tokenizers import Tokenizer
        for p in [f"{ROOT}/work/official_tinyaya/tokenizer.json", "/tmp/tok/tokenizer.json"]:
            if os.path.exists(p):
                tok = Tokenizer.from_file(p); break
    except Exception:
        pass
    llm_log = f"{OUT}/llm_checks.jsonl"
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm = LLMChecker(enabled=not a.no_llm, log_path=llm_log)
    bal0 = None
    if llm.enabled:
        try:
            bal0 = llm.L.balance()
        except Exception as e:
            llm.error = f"balance check failed: {e!r}"[:200]
    report, all_rej = {}, []
    eth_total_train = None
    for split in ["train", "dev"]:
        S = load(split)
        steps = []
        def rec(name, before, after, extra=None, rej=None):
            d = {"step": name, "in": before, "out": after, "removed": before - after}
            if rej is not None:
                d["reasons"] = dict(collections.Counter(":".join(r.rejection_reason.split(":")[:2]).split("=")[0] for r in rej))
            if extra: d.update(extra)
            steps.append(d); print(split, d, flush=True)
        n0 = len(S)
        steps.append({"step": "load", "out": n0, "by_source": dict(collections.Counter(s.metadata["source"] for s in S))})
        b = len(S); S = TextCleaner().run(S); rec("TextCleaner (CuratorKIT)", b, len(S))
        b = len(S); S = ExactDeduplicator().run(S); rec("ExactDeduplicator (CuratorKIT)", b, len(S))
        b = len(S); S = MinHashDeduplicator(threshold=0.9).run(S); rec("MinHashDeduplicator 0.9 (CuratorKIT)", b, len(S))
        for g in [BrokenTextGate(), URLSentenceStripper(), EnglishHeavyGate(), OffTopicGate(llm), QAMismatchGate(llm)]:
            b = len(S); S, rj = g.run(S); all_rej += [(split, r) for r in rj]
            rec(g.NAME + " (CuratorKIT BaseGate)", b, len(S), getattr(g, "stats", None), rj)
        tgt = a.eth_target if split == "train" else max(1, round(a.eth_target * 157 / 2843))
        ds = EthioNLPDownsampler(tgt, prompt_budget_tokens=1024 - 363 - 64, tok=tok)
        b = len(S); S = ds.run(S); rec(ds.NAME + " (CuratorKIT BaseNormalizer)", b, len(S), ds.stats)
        dr = DangerSignReferral(); S = dr.run(S); rec(dr.NAME + " (CuratorKIT BaseNormalizer)", len(S), len(S), dr.stats)
        sg = SchemaGate(min_tokens=2, max_tokens=100000)
        b = len(S); S, rj = sg.run(S); all_rej += [(split, r) for r in rj]; rec("SchemaGate (CuratorKIT)", b, len(S), None, rj)
        rows = [to_row(s) for s in S]
        rng = random.Random(42); rng.shuffle(rows)
        with open(f"{OUT}/{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(f"{OUT}/chat/{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        report[split] = {"steps": steps, "final": len(rows),
                         "final_by_source": dict(collections.Counter(r["source"] for r in rows)),
                         "final_by_topic": dict(collections.Counter(r["topic"] for r in rows)),
                         "danger_examples": dr.examples[:60]}
    with open(f"{OUT}/rejected.jsonl", "w", encoding="utf-8") as f:
        for sp, r in all_rej:
            f.write(json.dumps({"split": sp, "id": r.id, "source": r.metadata.get("source"), "reason": r.rejection_reason,
                                "step": r.rejecting_step, "question": r.instruction[:300], "answer": r.output[:300]},
                               ensure_ascii=False) + "\n")
    llm_info = {"enabled": llm.enabled, "model": llm.model, "calls": llm.calls, "cost_micro_usd_reported": round(llm.spent),
                "error": llm.error}
    if llm.enabled and bal0 is not None:
        try:
            llm_info["cline_balance_before_micro_usd"] = bal0; llm_info["cline_balance_after_micro_usd"] = llm.L.balance()
        except Exception:
            pass
    report["llm"] = llm_info
    report["test_split"] = "not read, not modified"
    report["created"] = datetime.now(UTC).isoformat()
    json.dump(report, open(f"{OUT}/curation_stats.json", "w"), indent=1, ensure_ascii=False)
    write_md(report)
    print("LLM", llm_info)


def write_md(rep):
    L = ["# Curated dataset (run two): CuratorKIT curation report", "",
         f"Generated by `scripts/curate.py` on {rep['created'][:16]} UTC. Input: dataset v2 `data/train.jsonl`, `data/dev.jsonl`. "
         "The test split was not read or changed.", "",
         "Each row is a `curatorkit.DataSample`; built-in CuratorKIT steps (TextCleaner, ExactDeduplicator, MinHashDeduplicator, "
         "SchemaGate) run alongside project-specific gates written against CuratorKIT's `BaseGate` / `BaseNormalizer` "
         "interfaces, so every kept row carries a provenance chain and every drop has a structured reason in `rejected.jsonl`.", ""]
    for sp in ["train", "dev"]:
        r = rep[sp]
        L += [f"## {sp}", "", "| Step | In | Out | Removed | Notes |", "|---|---|---|---|---|"]
        for s in r["steps"]:
            if s["step"] == "load":
                L.append(f"| load | | {s['out']} | | {s['by_source']} |"); continue
            notes = []
            if s.get("reasons"): notes.append("reasons " + json.dumps(s["reasons"], ensure_ascii=False))
            for k in ("rows_with_url_or_email", "rows_rejected_after_strip", "keyword_candidates", "candidates", "llm_judged", "llm_used", "fit_prompt_budget", "dropped_too_long_prompt",
                      "danger_sign_questions", "already_ended_with_urgent_referral", "referral_line_added", "by_subtask_kept"):
                if k in s: notes.append(f"{k}={s[k]}")
            L.append(f"| {s['step']} | {s.get('in','')} | {s['out']} | {s.get('removed','')} | {'; '.join(str(n) for n in notes)} |")
        L += ["", f"Final {sp}: **{r['final']}** rows. By source: {r['final_by_source']}. By topic: {r['final_by_topic']}.", ""]
    L += ["## Referral line", "", f"Added to danger-sign answers that did not already end with an urgent referral: `{REFERRAL_LINE}` "
          "(\"This may be a danger sign, so go now to a health worker or the nearest health facility.\")", "",
          "Danger-sign questions (pregnancy/birth context plus bleeding, fever, headache, eclampsia/convulsions, labour, "
          "waters breaking, reduced fetal movement, blurred vision, severe pain, swelling, breathing; or postpartum plus a "
          "problem word). Sample of questions that got the line:", ""]
    for q in rep["train"]["danger_examples"][:25]:
        L.append(f"- {q}")
    L += ["", "## LLM use", "", f"{rep['llm']}", "",
          "The LLM only confirms candidates picked by cheap filters (keyword hits for off-topic; lowest-12% char-n-gram "
          "question/answer cosine per source or an acronym conflict for mismatch). Its verdicts are in `llm_checks.jsonl`.", ""]
    open(f"{OUT}/curation_report.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
