"""Run three curation (CuratorKIT): data/curated (run two) -> data/curated_v3.

Stages (each cached under work/run3/, so a rerun costs nothing; run them in order):
  classify  LLM labels every health question as urgent (danger sign) / routine (should see a health worker) / none.
  rewrite   For urgent rows and a sample of routine rows, the LLM rewrites the answer into natural Amharic, keeps or
            corrects the medical content, and ends with a clear referral sentence (returned separately).
  verify    A second model (DeepSeek) checks each rewrite: medically correct (1-5), fluent (1-5), clear referral.
  build     Applies the CuratorKIT gates/normalizers below and writes data/curated_v3/ (+ DPO pairs + report).

CuratorKIT steps in build (all on CuratorKIT DataSample, BaseGate / BaseNormalizer, with provenance):
  LLMQualityGate      keep rows with LLM fluency >= 4 and correctness >= 4 (scores from scripts/llm_quality_score.py)
  DangerSignMatcherV3 widened Amharic danger-sign regex (homophone-letter normalised), verified by the LLM label
  ReferralRewriter    replace urgent/routine answers with the verified fluent rewrite + referral sentence
                      (re-admits a rewritten row even when the quality gate dropped the original)
  SchemaGate          CuratorKIT built-in
The test split is never read.

Usage: python scripts/curate_v3.py classify|rewrite|verify|build
"""
import collections, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_client as H
from curatorkit import DataSample, RejectedSample
from curatorkit.gates.schema import SchemaGate
from curatorkit.interfaces import BaseGate, BaseNormalizer
from curatorkit.schema import ProvenanceRecord

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = f"{ROOT}/data/curated"
OUT = f"{ROOT}/data/curated_v3"
W = f"{ROOT}/work/run3"
GEN_MODEL = "google/gemma-4-31B-it:deepinfra"
CHECK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra"
HEALTH = {"walelign_srh", "afrihealth_qa", "danilior_maternal", "amhqa_mendeley"}
STEP_VERSION = "run3-1.0"
ROUTINE_CAP = 320
SEED = 42

# ------------------------------------------------------------------ Amharic homophone normalisation
# ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ (whole families), then the 4th (ā) form of ሀ/አ -> 1st form.
_FAM = {0x1210: 0x1200, 0x1280: 0x1200, 0x1220: 0x1230, 0x12D0: 0x12A0, 0x1340: 0x1338}


def norm(t):
    out = []
    for ch in t:
        c = ord(ch)
        for src, dst in _FAM.items():
            if src <= c < src + 8:
                c = dst + (c - src); break
        if c in (0x1203, 0x12A3):  # ሃ -> ሀ, ኣ -> አ
            c -= 3
        out.append(chr(c))
    return "".join(out)


def R(p):
    return re.compile(norm(p))


E = r"[\u1200-\u137F]"
PREG = R(r"(እርግዝና|እርጉ[ዝስ]|ነፍሰ ?ጡር|ነፍሰ በነፍስ|ጽንስ|ፅንስ|(?<!" + E + r")ሽል|(?<!" + E + r")[በከየ]?ምጥ|ወሊድ|መውለድ|ወለድ|አራስ|እንግዴ|ማርገዝ|አርግዤ|"
         r"ያረገዘች|ያረገዝኩ|አርግዣ|ፀንሻ|ጸንሻ|\d+ ?(ኛው|ኛ|ተኛው)? ?(ወር|ሳምንት)|(ሰባተኛው|ስምንተኛው|ዘጠነኛው|ስድስተኛው|አራተኛው|መጨረሻው|መጀመሪያው) ወር|ሆዴ ውስጥ|ሆዷ ውስጥ)")
POSTPARTUM = R(r"(ድህረ ?ወሊድ|ከወሊድ በ(ኋ|ሁዋ)ላ|ከወለድኩ|ከወለደች|ከወለዱ|እንደወለድኩ|እንደወለደች|ከተወለደች|ከተወለደ|ወልዳ|ተውልጃ|አውጥቼ|አውጥተን|አራስ እናት|ከተገላገል|ከመውለድ በ(ኋ|ሁዋ)ላ)")
NEWBORN = R(r"(አራስ (ልጅ|ህጻን|ህፃን|ሕፃን)|አዲስ የተወለደ|አዲስ በተወለዱ|ጨቅላ|የተወለደ (ልጅ|ህጻን|ህፃን)|ባለ \d+ ቀን|ህጻኑ|ህፃኑ|ሕፃኑ|ልጄ (ጡት|ሲተነፍስ|ትኩ)|እምብርት)")
SIGNS = {
    "bleeding": R(r"(ደም (መፍሰስ|ይፈስ|መፍሰሱ|ፈሰሰ|ቢፈስ|እየፈሰሰ|የሚፈስ|መጣ|እየመጣ|መምጣ|ማየት|አየሁ)|የደም (መፍሰስ|ጠብታ|ትልትል)|እየደማ|ደሜ|ደሙ (በድንገት|ጨመረ|አልቆመ)|ነጠብጣብ ደም)"),
    "headache_vision": R(r"(ራስ ምታት|ራሴ? .{0,10}(አሞ|ያመ)|ራሷን .{0,8}(አሞ|ያሞ|ሲያሞ)|"
                        r"(አይ|አይን|የአይን|አይኔ|አይኗ|አይኑ).{0,15}(ብዥ|ነጠብጣብ|መጋረድ|እየተደባበዘ|አይታየ|አይነቃነቅ)|ብዥታ|ብዥ ያለ|የማየት ችግር)"),
    "fever": R(r"(ትኩሳት|ትኩሳቴ|ብርድ ብርድ|ሰውነ(ቴ|ቷ|ቱ) .{0,6}(ሞቀ|ይሞቃል|ትኩስ)|ከፍተኛ ሙቀት)"),
    "fetal_movement": R(r"(እንቅስቃሴ.{0,25}(ቀነሰ|መቀነስ|አቆመ|ማቆም|የለም|አልተሰማ|ቀንሷል)|(ቀነሰ|መቀነስ|አቆመ).{0,15}እንቅስቃሴ|"
                       r"አልተንቀሳቀሰም|አይንቀሳቀስም|የማይነቃነቅ|አላንቀሳቀሰም|መንቀሳቀስ (አቆመ|አቁሟል)|ሲመታ .{0,20}አልተሰማ|ዝም ብሏል)"),
    "waters_breaking": R(r"(የእንሽርት ውሃ|እንሽርት|ውሃ(ው|ዬ|ዋ)? (መፍሰስ|ፈሰሰ|ቢፈስ|ይፈሳል|እየፈሰሰ|ተሰበረ|መቱአል)|ውሃ መሰል ፈሳሽ|ሽርት ውሃ)"),
    "abdominal_pain": R(r"(ከባድ (ህመም|ሕመም|የሆድ)|የሆድ (ህመም|ሕመም|ቁርጠት|ቁርጫ|ቁስለት)|ሆ(ዴ|ዷ|ዱ)(ን| ላይ)?.{0,15}(ህመም|ሕመም|ያመ|አመመ|ቁርጠት|ቁርጫ|የሚወጋ|አበጠ|ተቆረጠ)|ቁርጠት|የታችኛው ሆድ)"),
    "swelling_breathing": R(r"(የፊት እብጠት|(ፊ(ቴ|ቷ|ቱ)|እጆ(ቼ|ቿ|ቹ)|እጆች).{0,15}(አበጠ|አበጡ|አብጧል|አበጥ|እብጠት)|እብጠት|ማበጥ|ያበጠ|የመተንፈስ ችግር|ትንፋ(ሽ|ሿ|ሼ) .{0,6}(አጠራ|ማጠር|አጠረ|ፈጣን)|መተንፈስ)"),
    "pp_problem": R(r"(ደም|ትኩሳት|ህመም|ሕመም|ኢንፌክሽን|ጭንቀት|ድብርት|ድባቴ|መንፈስ|ችግር|ፈሳሽ|ሽታ|እብጠት|ማዞር|ራሷን|ራሴን)"),
    "newborn_problem": R(r"(ትኩሳት|ትኩላ|ትኩስ|ቢጫ|አይጠባም|አልጠባም|ጡት .{0,12}(አይጠባ|አልጠባ|መጥባት|አቆመ|ፍቃደኛ)|መተንፈስ|ትንፋ|ይፈጥናል|ማንቀጥቀጥ|መንቀጥቀጥ|እየተንቀጠቀጠ|"
                        r"እምብርት.{0,25}(መግል|ቀይ|ሽታ|ደም)|መግል|ቀዝቃዛ|ቀዝቅዝ|አይበላም|ደካማ)"),
    "abortion": R(r"(ውርጃ|(ጽንስ|ፅንስ) (ማቋረጥ|ማስወረድ|ማስወገድ|ማውረድ|ካጣ|ካጣች|ካጣሁ)|(ጽንሱን|ፅንሱን|ፅንሷን|ጽንሷን) (ያጣ|ካጣ)|አውርጃ|ማስወረድ|ጽንሱ ወረደ|ጭንግፍ|ሳይወለድ መሞት)"),
    "abortion_problem": R(r"(ደም|ትኩሳት|ህመም|ሕመም|ቁርጠት|ሽታ|ፈሳሽ|ኢንፌክሽን|ችግር|ማዞር|ደክሞ|ደክሟ)"),
}
STANDALONE = {
    "convulsions": R(r"(መንቀጥቀጥ|ማንቀጥቀጥ|ተንቀጠቀ|እየተንቀጠቀ|ትንቀጠቀጣ|የሚጥል|ያንቀጠቅ|ራስን መሳት|ራሷን (ሳተች|ስታ|አጥታ)|ራሴን (ሳትኩ|ሳተሁ|ስቼ|አጥቼ)|"
                    r"ኤክላምፕሲ|ኤክላምሲ|ፕሪ-?ኤክላምፕሲ|ፕሬ-?ኤክላምፕሲ|ቅድመ ኤክላምፕሲ)"),
    "prolonged_labour": R(r"((?<!" + E + r")[በከየ]?ምጥ.{0,30}(ሰአ[ትታ]|ተራዘመ|ረጅም|ቆየ|ቆይቷል|ከ ?12)|ከ ?12 ሰአት በላይ.{0,20}ምጥ|የተራዘመ ምጥ|ረጅም ምጥ|ቅድመ ወሊድ ምጥ|ያለጊዜው ምጥ)"),
    "suicidal": R(r"(ራስን (ማጥፋት|መግደል|የማጥፋት|የመግደል|የመጉዳት|መጉዳት)|ራሴን (ማጥፋት|ልገድል|ላጠፋ|መግደል|ለማጥፋት|ለመግደል|ለመጉዳት|ለመጎዳት|ካጠፋሁ)|"
                 r"ራሷን (ማጥፋት|ልትገድል|ለመግደል|ለመጉዳት|ልትጉዳ)|መሞት እፈልጋለሁ|መኖር አልፈልግም|ሕይወቴን ማጥፋት)"),
}


def danger_match(q):
    """Return the list of matched danger-sign groups for a question (empty if none)."""
    n = norm(q)
    hit = []
    preg, pp, nb = bool(PREG.search(n)), bool(POSTPARTUM.search(n)), bool(NEWBORN.search(n))
    for k in ["bleeding", "headache_vision", "fever", "fetal_movement", "waters_breaking", "abdominal_pain", "swelling_breathing"]:
        if preg and SIGNS[k].search(n):
            hit.append(k)
    for k, p in STANDALONE.items():
        if p.search(n):
            hit.append(k)
    if pp and SIGNS["pp_problem"].search(n):
        hit.append("postpartum")
    if nb and SIGNS["newborn_problem"].search(n):
        hit.append("newborn")
    if SIGNS["abortion"].search(n) and SIGNS["abortion_problem"].search(n):
        hit.append("abortion_complication")
    return hit


REFERRAL = re.compile(r"(ሐኪም|ሀኪም|ሃኪም|የጤና ባለሙያ|ጤና ባለሙያ|የጤና ተቋም|ጤና ተቋም|ጤና ጣቢያ|ሆስፒታል|ክሊኒክ|ዶክተር|የጤና ሰራተኛ|ጤና ኤክስቴንሽን)")
URGENT = re.compile(r"(ወዲያውኑ|አሁኑኑ|በአስቸኳይ|አስቸኳይ|በፍጥነት|ቶሎ|አፋጣኝ|ሳይዘገዩ|ሳትዘገይ|ሳይዘገይ|ድንገተኛ|ሳትዘገዩ|ሳይዘገዩ)")


LATIN = re.compile(r"[a-z\u00C0-\u024F\u1E00-\u1EFF]")
DIRECTIVE = re.compile(r"(ይሂዱ|ሂዱ|ሂጂ|ሂድ|ያማክሩ|ያማክሪ|ይጎብኙ|ይሄዱ|መሄድ (አለብዎት|አለባት|አለብሽ|ያስፈልጋል)|ማማከር (አለብዎት|ያስፈልጋል))")


def body_has_referral(body):
    """A sentence in the body that directs the reader to a health worker/facility (so 'body' alone is not referral-free)."""
    for sent in re.split(r"(?<=[።!?])\s*", body):
        if REFERRAL.search(sent) and (URGENT.search(sent) or DIRECTIVE.search(sent)):
            return True
    return False


def load_rows():
    return [json.loads(l) for l in open(f"{IN}/train.jsonl", encoding="utf-8")]


def cache_load(p):
    d = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            j = json.loads(l); d[j["id"]] = j
    return d


def run_pool(fn, batches, path, workers=10):
    stop = {"why": None}

    def wrap(b):
        if stop["why"]:
            return []
        try:
            return fn(b)
        except H.BillingError as e:
            stop["why"] = str(e); return []
        except Exception as e:
            print("batch failed", repr(e)[:200], flush=True); return []
    n = 0
    with ThreadPoolExecutor(workers) as ex, open(path, "a", encoding="utf-8") as f:
        for out in ex.map(wrap, batches):
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n"); n += 1
            f.flush()
    print("wrote", n, "stop", stop["why"], "HF spent total $%.4f" % H.spent_usd(), flush=True)


# ------------------------------------------------------------------ stage: classify
CLS_SYS = """You are an Ethiopian maternal and reproductive health physician triaging questions sent to an Amharic health assistant.
For each item (id + Amharic QUESTION) give:
- need: "urgent" if the question describes or asks about a danger sign or emergency that needs care at a health facility now (e.g. bleeding in pregnancy or after birth, severe headache or blurred vision in pregnancy, fever in pregnancy or after birth, convulsions/eclampsia, reduced or no fetal movement, labour longer than 12 hours, waters breaking early, severe abdominal pain, newborn danger signs, suicidal thoughts, complications after an abortion or miscarriage, difficulty breathing);
  "routine" if a good answer should advise seeing a health worker or facility but not as an emergency (symptoms that need diagnosis or testing, possible infection or STI, treatment or medicine choice, contraception method choice, fertility problems, antenatal care visits);
  "none" for general facts, definitions, myths, statistics, policy or social questions.
- sign: short English label of the danger sign if urgent, else "".
Reply with JSON only: {"<id>": {"need": "urgent"|"routine"|"none", "sign": str}, ...}"""


def stage_classify():
    rows = [r for r in load_rows() if r["source"] in HEALTH]
    p = f"{W}/referral_labels.jsonl"
    c = cache_load(p)
    todo = [r for r in rows if r["id"] not in c]
    print("health rows", len(rows), "todo", len(todo), flush=True)

    def fn(b):
        items = [{"id": f"q{k}", "QUESTION": r["question"][:600]} for k, r in enumerate(b)]
        txt, _ = H.chat(GEN_MODEL, [{"role": "system", "content": CLS_SYS},
                                    {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                        max_tokens=40 * len(b) + 200, tag="classify")
        res = H.parse_json(txt)
        out = []
        for k, r in enumerate(b):
            s = res.get(f"q{k}") or {}
            if s.get("need") in ("urgent", "routine", "none"):
                out.append({"id": r["id"], "need": s["need"], "sign": s.get("sign", ""), "model": GEN_MODEL})
        return out
    run_pool(fn, [todo[i:i + 20] for i in range(0, len(todo), 20)], p)



# ------------------------------------------------------------------ stage: synth (danger-sign questions the corpus lacks)
SYN_CATS = {
    "bleeding": "vaginal bleeding during pregnancy (any amount, early or late pregnancy)",
    "headache_vision": "severe headache, blurred vision or seeing spots in pregnancy",
    "fever": "fever or chills during pregnancy",
    "convulsions": "fits/convulsions or fainting in pregnancy or after birth",
    "fetal_movement": "baby moving less or not moving in the womb",
    "prolonged_labour": "labour pains lasting more than 12 hours / labour not progressing",
    "waters_breaking": "waters breaking before labour or before 37 weeks",
    "abdominal_pain": "severe abdominal pain in pregnancy",
    "postpartum_bleeding": "heavy bleeding after giving birth",
    "postpartum_fever": "fever or foul-smelling discharge after giving birth",
    "newborn": "newborn danger signs (not feeding, fever or cold, fast breathing, yellow skin, convulsions, red or pus at cord)",
    "suicidal": "thoughts of suicide or self-harm, including after birth or after an unwanted pregnancy",
    "abortion_complication": "heavy bleeding, fever, severe pain or bad-smelling discharge after an abortion or miscarriage",
    "swelling": "sudden swelling of face and hands in pregnancy, or difficulty breathing",
}
SYN_SYS = """You write realistic questions that Ethiopian women, partners or family members would type in Amharic to a health assistant.
Write {n} different questions about: {topic}.
Use natural everyday Amharic (Ethiopic script), first person or about a wife/sister/daughter/baby, varied wording, some short, some with details such as weeks of pregnancy or hours. No English. Each question 6 to 30 words.
Reply with JSON only: ["question 1", "question 2", ...]"""


def stage_synth():
    p = f"{W}/synth_questions.jsonl"
    c = cache_load(p)
    todo = [k for k in SYN_CATS if f"syn-{k}-00" not in c]

    def fn(b):
        k = b[0]
        txt, _ = H.chat(GEN_MODEL, [{"role": "user", "content": SYN_SYS.format(n=10, topic=SYN_CATS[k])}],
                        max_tokens=1500, temperature=0.8, tag="synth")
        qs = [q.strip() for q in H.parse_json(txt) if isinstance(q, str) and q.strip()]
        return [{"id": f"syn-{k}-{i:02d}", "cat": k, "question": q, "model": GEN_MODEL} for i, q in enumerate(qs[:10])]
    run_pool(fn, [[k] for k in todo], p)


def synth_rows():
    out = []
    for k, v in cache_load(f"{W}/synth_questions.jsonl").items():
        if re.search(r"[A-Za-z]", v["question"]):  # Gemma sometimes leaks Latin letters; drop those questions
            continue
        out.append({"id": "train-synth-" + k, "source": "synthetic_danger_sign", "license": "generated (run three, Gemma 4 31B)",
                    "question": v["question"], "answer": "", "context": "", "topic": "maternal", "split": "train",
                    "synthetic": True, "syn_cat": v["cat"]})
    return out

# ------------------------------------------------------------------ stage: rewrite
RW_SYS = """You are a native Amharic medical writer and an Ethiopian midwife/physician. You rewrite answers for an Amharic maternal, sexual and reproductive health assistant used by ordinary people in Ethiopia.
Given a QUESTION, an ORIGINAL answer (often stiff machine-translated Amharic) and the CARE level, write a better answer:
- natural, fluent, plain Amharic as a native Ethiopian health worker would speak; short clear sentences; no English words unless a medicine name needs it; no Markdown, no bullet symbols, no headings;
- keep what is medically correct in the ORIGINAL, fix anything wrong or unsafe, and answer the question directly; follow WHO / Ethiopian Ministry of Health guidance;
- 50 to 130 words for "body";
- "referral" is ONE separate final sentence: for CARE=urgent it must clearly say to go to a health worker or the nearest health facility now / immediately (e.g. using "አሁኑኑ" or "በአስቸኳይ"); for CARE=routine it must advise seeing a health worker or visiting a health facility (e.g. health centre) for proper check-up, testing or treatment.
- "body" must NOT contain any advice to see a health worker; that goes only in "referral".
Reply with JSON only: {"body": "...", "referral": "..."}"""


def select_rewrite(rows, labels):
    rng = random.Random(SEED)
    urgent = [r for r in rows if labels.get(r["id"], {}).get("need") == "urgent"]
    routine = [r for r in rows if labels.get(r["id"], {}).get("need") == "routine"]
    rng.shuffle(routine)
    return urgent, routine[:ROUTINE_CAP]


def stage_rewrite():
    rows = {r["id"]: r for r in load_rows()}  # synthetic rows are added to the urgent list below
    labels = cache_load(f"{W}/referral_labels.jsonl")
    urgent, routine = select_rewrite(list(rows.values()), labels)
    urgent += synth_rows()
    p = f"{W}/rewrites.jsonl"
    c = cache_load(p)
    todo = [(r, "urgent") for r in urgent if r["id"] not in c] + [(r, "routine") for r in routine if r["id"] not in c]
    print("urgent", len(urgent), "routine", len(routine), "todo", len(todo), flush=True)

    def fn(b):
        out = []
        for r, care in b:
            u = {"QUESTION": r["question"], "ORIGINAL": r["answer"][:2500] or "(none: write the answer from scratch)", "CARE": care}
            txt, _ = H.chat(GEN_MODEL, [{"role": "system", "content": RW_SYS},
                                        {"role": "user", "content": json.dumps(u, ensure_ascii=False)}],
                            max_tokens=900, temperature=0.3, tag="rewrite")
            j = H.parse_json(txt)
            if j.get("body") and j.get("referral"):
                out.append({"id": r["id"], "care": care, "body": j["body"].strip(), "referral": j["referral"].strip(),
                            "model": GEN_MODEL})
        return out
    run_pool(fn, [[t] for t in todo], p, workers=12)


# ------------------------------------------------------------------ stage: verify
VF_SYS = """You are a strict Ethiopian physician and native Amharic speaker reviewing answers written for an Amharic maternal and reproductive health assistant.
For each item (id, QUESTION, ANSWER, CARE level) score:
- correctness 1-5 (medically accurate, safe, answers the question; 5 = fully correct),
- fluency 1-5 (natural native Amharic; 5 = native-like),
- referral_ok true/false: for CARE=urgent, does the answer clearly tell the person to go to a health worker or health facility now/immediately? for CARE=routine, does it advise seeing a health worker or facility?
Reply with JSON only: {"<id>": {"correctness": int, "fluency": int, "referral_ok": bool}, ...}"""


def stage_verify():
    rows = {r["id"]: r for r in load_rows() + synth_rows()}
    rw = cache_load(f"{W}/rewrites.jsonl")
    p = f"{W}/rewrite_checks.jsonl"
    c = cache_load(p)
    todo = [v for k, v in rw.items() if k not in c]
    print("rewrites", len(rw), "todo", len(todo), flush=True)

    def fn(b):
        items = [{"id": f"a{k}", "QUESTION": rows[v["id"]]["question"], "ANSWER": v["body"] + " " + v["referral"],
                  "CARE": v["care"]} for k, v in enumerate(b)]
        txt, _ = H.chat(CHECK_MODEL, [{"role": "system", "content": VF_SYS},
                                      {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                        max_tokens=50 * len(b) + 300, tag="verify")
        res = H.parse_json(txt)
        out = []
        for k, v in enumerate(b):
            s = res.get(f"a{k}")
            try:
                out.append({"id": v["id"], "correctness": int(s["correctness"]), "fluency": int(s["fluency"]),
                            "referral_ok": bool(s["referral_ok"]), "model": CHECK_MODEL})
            except Exception:
                pass
        return out
    run_pool(fn, [todo[i:i + 6] for i in range(0, len(todo), 6)], p)


# ------------------------------------------------------------------ stage: build (CuratorKIT)
def prov(s, step, notes):
    s.append_provenance(ProvenanceRecord(step_name=step, step_version=STEP_VERSION, config_hash=step[:16], notes=notes))


def to_sample(r):
    return DataSample(id=r["id"], source_uri=f"data/curated/train.jsonl#{r['source']}", instruction=r["question"],
                      input=r.get("context") or "", output=r["answer"], metadata=dict(r))


class LLMQualityGate(BaseGate):
    NAME = "LLMQualityGate"

    def __init__(self, scores, min_f=4, min_c=4):
        self.scores, self.min_f, self.min_c, self.stats = scores, min_f, min_c, {}

    def run(self, samples):
        ok, bad = [], []
        c = collections.Counter()
        for s in samples:
            q = self.scores.get(s.id)
            if q is None:
                bad.append(RejectedSample(**s.model_dump(), rejection_reason="quality:unscored", rejecting_step=self.NAME)); continue
            s.metadata["q_fluency"], s.metadata["q_correctness"] = q["fluency"], q["correctness"]
            if q["fluency"] >= self.min_f and q["correctness"] >= self.min_c:
                prov(s, self.NAME, {"fluency": q["fluency"], "correctness": q["correctness"]}); ok.append(s)
                c["kept_" + s.metadata["source"]] += 1
            else:
                why = "low_fluency" if q["fluency"] < self.min_f else "low_correctness"
                bad.append(RejectedSample(**s.model_dump(), rejection_reason=f"quality:{why}", rejecting_step=self.NAME))
                c[f"{why}_{s.metadata['source']}"] += 1
        self.stats = dict(sorted(c.items()))
        return ok, bad


def main_build():
    rows = load_rows()
    byid = {r["id"]: r for r in rows + synth_rows()}
    scores = {}  # quality cache may hold one line per id per model; use the Gemma scorer
    for l in open(f"{W}/quality_scores.jsonl", encoding="utf-8"):
        j = json.loads(l)
        if j["model"] == GEN_MODEL:
            scores[j["id"]] = j
    labels = cache_load(f"{W}/referral_labels.jsonl")
    rw = cache_load(f"{W}/rewrites.jsonl")
    chk = cache_load(f"{W}/rewrite_checks.jsonl")
    steps, rej = [], []
    S = [to_sample(r) for r in rows]

    def rec(name, a, b, notes=None):
        steps.append({"step": name, "in": a, "out": b, "removed": a - b, "notes": notes or {}})

    # 1. danger-sign matcher (regex) + LLM verification (stats only; drives nothing by itself)
    health = [r for r in rows if r["source"] in HEALTH]
    m_regex = {r["id"]: danger_match(r["question"]) for r in health}
    m_regex = {k: v for k, v in m_regex.items() if v}
    llm_urgent = {k for k, v in labels.items() if v["need"] == "urgent"}
    llm_routine = {k for k, v in labels.items() if v["need"] == "routine"}
    by_group = collections.Counter(g for v in m_regex.values() for g in v)
    by_group_conf = collections.Counter(g for k, v in m_regex.items() if k in llm_urgent for g in v)
    danger_stats = {"health_rows": len(health), "llm_labelled": sum(1 for r in health if r["id"] in labels),
                    "regex_matches": len(m_regex), "regex_matches_llm_confirmed_urgent": len(set(m_regex) & llm_urgent),
                    "llm_urgent_total": len(llm_urgent), "llm_urgent_missed_by_regex": len(llm_urgent - set(m_regex)),
                    "llm_routine_total": len(llm_routine),
                    "regex_by_group": dict(by_group), "regex_by_group_llm_confirmed": dict(by_group_conf),
                    "run2_matcher_questions": 17}
    # 2. quality gate
    g = LLMQualityGate(scores)
    b = len(S); S, rj = g.run(S); rej += rj; rec("LLMQualityGate fluency>=4 & correctness>=4 (CuratorKIT BaseGate)", b, len(S), g.stats)
    # 3. referral rewriter (verified rewrites only)
    good = {k for k, v in rw.items() if k in chk and chk[k]["correctness"] >= 4 and chk[k]["fluency"] >= 4 and chk[k]["referral_ok"]
            and REFERRAL.search(v["referral"]) and (v["care"] == "routine" or URGENT.search(v["referral"]))
            and not body_has_referral(v["body"]) and not LATIN.search(v["body"] + v["referral"]) and k in byid}
    kept = {s.id for s in S}
    readd = 0
    for s in S:
        if s.id in good:
            v = rw[s.id]; s.output = v["body"] + " " + v["referral"]
            s.metadata["referral_rewrite"] = v["care"]; prov(s, "ReferralRewriter", {"care": v["care"]})
    for k in sorted(good - kept):
        s = to_sample(byid[k]); v = rw[k]; s.output = v["body"] + " " + v["referral"]
        s.metadata["referral_rewrite"] = v["care"]; prov(s, "ReferralRewriter", {"care": v["care"], "readmitted": True})
        S.append(s); readd += 1
    rw_stats = {"rewrites": len(rw), "verified_ok": len(good), "verified_urgent": sum(rw[k]["care"] == "urgent" for k in good),
                "verified_routine": sum(rw[k]["care"] == "routine" for k in good), "readmitted_after_quality_gate": readd - sum(k.startswith("train-synth-") for k in good),
                "synthetic_danger_sign_added": sum(k.startswith("train-synth-") for k in good),
                "failed_verification": len(rw) - len(good),
                "fail_reasons": dict(collections.Counter(
                    ("latin_letters" if LATIN.search(rw[k]["body"] + rw[k]["referral"]) else "body_has_referral" if body_has_referral(rw[k]["body"])
                     else "unchecked" if k not in chk else "low_correctness" if chk[k]["correctness"] < 4 else "low_fluency" if chk[k]["fluency"] < 4
                     else "referral_not_ok" if not chk[k]["referral_ok"] else "regex_referral_check") for k in rw if k not in good))}
    rec("ReferralRewriter (CuratorKIT BaseNormalizer)", len(S) - readd, len(S), rw_stats)
    # 4. schema gate
    sg = SchemaGate(min_tokens=2, max_tokens=100000)
    b = len(S); S, rj = sg.run(S); rej += rj; rec("SchemaGate (CuratorKIT)", b, len(S))
    os.makedirs(f"{OUT}/chat", exist_ok=True); os.makedirs(f"{OUT}/dpo", exist_ok=True)
    out_rows = []
    for s in S:
        r = {k: s.metadata[k] for k in ["id", "source", "license", "question", "answer", "context", "topic", "split", "synthetic"] if k in s.metadata}
        r["answer"] = s.output
        r["curation_v3"] = {"q_fluency": s.metadata.get("q_fluency"), "q_correctness": s.metadata.get("q_correctness"),
                            "referral_rewrite": s.metadata.get("referral_rewrite"), "danger_groups": m_regex.get(s.id, []),
                            "steps": [p.step_name for p in s.provenance_chain]}
        out_rows.append(r)
    random.Random(SEED).shuffle(out_rows)
    with open(f"{OUT}/train.jsonl", "w", encoding="utf-8") as f, open(f"{OUT}/chat/train.jsonl", "w", encoding="utf-8") as fc:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            user = (r["context"] + "\n\n" + r["question"]) if r.get("context") else r["question"]
            fc.write(json.dumps({"id": r["id"], "source": r["source"], "topic": r["topic"],
                                 "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["answer"]}]},
                                ensure_ascii=False) + "\n")
    # dev unchanged from run two (same eval_loss yardstick)
    for fn in ["dev.jsonl", "chat/dev.jsonl"]:
        open(f"{OUT}/{fn}", "w", encoding="utf-8").write(open(f"{IN}/{fn}", encoding="utf-8").read())
    with open(f"{OUT}/rejected.jsonl", "w", encoding="utf-8") as f:
        for r in rej:
            f.write(json.dumps({"id": r.id, "reason": r.rejection_reason, "step": r.rejecting_step}, ensure_ascii=False) + "\n")
    # 5. DPO pairs
    pairs = []
    for k in sorted(good):
        v, orig = rw[k], byid[k]
        chosen = v["body"] + " " + v["referral"]
        prompt = [{"role": "user", "content": orig["question"]}]
        pairs.append({"id": k + "#noref", "kind": "minus_referral", "care": v["care"], "prompt": prompt,
                      "chosen": [{"role": "assistant", "content": chosen}], "rejected": [{"role": "assistant", "content": v["body"]}]})
        oq = scores.get(k, {})
        if orig["answer"].strip() and not (REFERRAL.search(orig["answer"]) and URGENT.search(orig["answer"])) and orig["answer"].strip() != chosen:
            pairs.append({"id": k + "#mt", "kind": "original_mt_answer", "care": v["care"], "prompt": prompt,
                          "chosen": [{"role": "assistant", "content": chosen}],
                          "rejected": [{"role": "assistant", "content": orig["answer"]}], "orig_fluency": oq.get("fluency")})
    rng = random.Random(SEED)
    ids = sorted({p["id"].split("#")[0] for p in pairs}); rng.shuffle(ids)
    dev_ids = set(ids[:max(8, len(ids) // 20)])
    dtr = [p for p in pairs if p["id"].split("#")[0] not in dev_ids]
    ddv = [p for p in pairs if p["id"].split("#")[0] in dev_ids]
    rng.shuffle(dtr)
    for name, part in [("train", dtr), ("dev", ddv)]:
        with open(f"{OUT}/dpo/{name}.jsonl", "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    final_by_src = dict(collections.Counter(r["source"] for r in out_rows))
    stats = {"created": datetime.now(UTC).isoformat(), "input": f"data/curated/train.jsonl ({len(rows)} rows)", "steps": steps,
             "final_train": len(out_rows), "final_by_source": final_by_src,
             "final_by_topic": dict(collections.Counter(r["topic"] for r in out_rows)),
             "danger_matcher": danger_stats, "dpo": {"pairs_train": len(dtr), "pairs_dev": len(ddv),
                                                     "by_kind": dict(collections.Counter(p["kind"] for p in pairs)),
                                                     "by_care": dict(collections.Counter(p["care"] for p in pairs))},
             "models": {"quality_and_rewrite": GEN_MODEL, "verify": CHECK_MODEL},
             "hf_spend_usd_total_run3_so_far": round(H.spent_usd(), 4),
             "hf_spend_by_tag": {t: round(sum(json.loads(l)["usd"] for l in open(H.SPEND) if json.loads(l)["tag"] == t), 4)
                                 for t in sorted({json.loads(l)["tag"] for l in open(H.SPEND)})},
             "test_split": "not read, not modified"}
    json.dump(stats, open(f"{OUT}/curation_stats.json", "w"), indent=1, ensure_ascii=False)
    examples = [(byid[k]["question"][:140], m_regex.get(k, []), rw[k]["referral"]) for k in sorted(good) if rw[k]["care"] == "urgent"][:12]
    write_md(stats, examples)
    print(json.dumps({k: stats[k] for k in ["final_train", "final_by_source", "danger_matcher", "dpo"]}, indent=1, ensure_ascii=False))


def write_md(st, examples):
    L = ["# Curated dataset v3 (run three): CuratorKIT curation report", "",
         f"Generated by `scripts/curate_v3.py` on {st['created'][:16]} UTC from {st['input']}. The test split was not read.", "",
         "Every row is a `curatorkit.DataSample`; the run-three steps are CuratorKIT `BaseGate` / `BaseNormalizer` "
         "implementations plus CuratorKIT's built-in `SchemaGate`, so kept rows carry a provenance chain and drops "
         "are listed with a reason in `rejected.jsonl`.", "",
         "| Step | In | Out | Removed | Notes |", "|---|---|---|---|---|"]
    for s in st["steps"]:
        L.append(f"| {s['step']} | {s['in']} | {s['out']} | {s['removed']} | {json.dumps(s['notes'], ensure_ascii=False)} |")
    L += ["", f"Final train: **{st['final_train']}** rows. By source: {st['final_by_source']}. By topic: {st['final_by_topic']}.",
          "Dev is unchanged from run two (`data/curated/dev.jsonl`, 249 rows) so dev eval_loss stays comparable.", "",
          "## 1. LLM quality filter",
          f"Every curated train answer was scored 1-5 for Amharic fluency and correctness by `{st['models']['quality_and_rewrite']}` "
          "(HF Inference Providers, 8 rows per call; scores in `work/run3/quality_scores.jsonl`). Kept: fluency >= 4 and correctness >= 4 "
          "(about 50% of rows, so the top-60% fallback was not needed).", "",
          "## 2. Danger-sign matcher and referral",
          "Widened regex (bleeding, severe headache / blurred vision, fever, convulsions / eclampsia, reduced fetal movement, prolonged "
          "labour, waters breaking, severe abdominal pain, swelling / breathing, postpartum problems, newborn danger signs, suicidal "
          "thoughts, abortion complications), applied after normalising Amharic homophone letters (ሐ/ኀ/ሃ->ሀ, ሠ->ሰ, ዐ/ኣ->አ, ፀ->ጸ) so "
          "common spellings match. The LLM labelled every health question urgent / routine / none, which verifies the regex hits "
          "and catches urgent questions the regex missed.", "",
          "```", json.dumps(st["danger_matcher"], ensure_ascii=False, indent=1), "```", "",
          "For LLM-urgent questions (all) and a seeded sample of LLM-routine questions, the LLM rewrote the answer into natural "
          "Amharic with a separate final referral sentence (urgent: go to a health worker / facility now). "
          f"A second model (`{st['models']['verify']}`) checked every rewrite (correctness >= 4, fluency >= 4, referral clear) and a "
          "regex checked the referral sentence has a facility/health-worker word (plus an urgency word for urgent) and the body has none. "
          "Only verified rewrites replace the original answer.", "", "Sample urgent questions with their referral sentence:", ""]
    for q, gr, ref in examples:
        L.append(f"- {q} ({', '.join(gr) or 'LLM only'}) -> {ref}")
    L += ["", "## 3. DPO preference pairs (`dpo/train.jsonl`, `dpo/dev.jsonl`)",
          "chosen = verified fluent rewrite with referral; rejected = (a) the same rewrite with the referral sentence removed, or "
          "(b) the original (mostly machine-translated) answer when it had no urgent referral.", "",
          "```", json.dumps(st["dpo"], ensure_ascii=False, indent=1), "```", "",
          "## LLM spend (HF Inference Providers, priced from the router's per-token rates)", "",
          "```", json.dumps(st["hf_spend_by_tag"]), f"total so far ${st['hf_spend_usd_total_run3_so_far']}", "```"]
    open(f"{OUT}/curation_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    os.makedirs(W, exist_ok=True)
    {"classify": stage_classify, "rewrite": stage_rewrite, "verify": stage_verify, "build": main_build, "synth": stage_synth}[sys.argv[1]]()
