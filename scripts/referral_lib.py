"""Run four: shared referral regexes (no LLM). Used by curate_v4.py (strip / append boilerplate) and
referral_audit.py (AuditKIT-style over-referral metric).

- REF_WIDE: the run-three wide referral word list (facility / health-worker words), same as curate_v3.REFERRAL.
- referral_line(text): True if any sentence names a facility/health worker AND carries a directive or urgency verb
  (go / consult / take / get care / now ...). This is the "referral line" counted by the over-referral metric.
- BOILERPLATE: the generic closing line v3 DPO emits ("for a full examination and proper treatment go to a health
  centre/facility and consult a health professional"), i.e. a sentence that opens with a purpose phrase
  ለ(ተሟላ|ተገቢ|በለጠ|ተሻለ|ትክክለኛ)...: operationally any referral sentence with no condition (ካለ / ከሆነ / ከታዩ ...)
  and no urgency word. A specific 'see a provider if X' clause (has a condition) or an urgent 'go now' line is not boilerplate.
"""
import re

REF_WIDE = re.compile(r"(ሐኪም|ሀኪም|ሃኪም|የጤና ባለሙያ|ጤና ባለሙያ|የጤና ተቋም|ጤና ተቋም|ጤና ጣቢያ|ሆስፒታል|ክሊኒክ|ዶክተር|የጤና ሰራተኛ|ጤና ኤክስቴንሽን)")  # == curate_v3.REFERRAL
REF_LINE_WORDS = re.compile(REF_WIDE.pattern[:-1] + r"|ሕክምና ባለሙያ|ህክምና ባለሙያ)")
URGENT = re.compile(r"(ወዲያውኑ|አሁኑኑ|በአስቸኳይ|አስቸኳይ|በፍጥነት|ቶሎ|አፋጣኝ|ሳይዘገዩ|ሳትዘገይ|ሳይዘገይ|ድንገተኛ|ሳትዘገዩ)")
DIRECTIVE = re.compile(r"(ይሂዱ|ሂዱ|ሂጂ|ሂድ|ያማክሩ|ያማክሪ|ይማክሩ|ያማክ|ይጎብኙ|ይሄዱ|ይውሰዱ|ያግኙ|ያድርጉ|ያስመርምሩ|ይመርመሩ|"
                       r"መሄድ (አለብዎት|አለባት|አለብሽ|ያስፈልጋል|ይኖርብዎታል|ይመከራል)|ማማከር (አለብዎት|ያስፈልጋል|ይመከራል|ይኖርብዎታል)|"
                       r"ማየት (ያስፈልጋል|ይኖርብዎታል)|በመሄድ)")
COND = re.compile(r"(ካለ|ካሉ|ካለዎት|ካለብዎት|ከሆነ|ከሆኑ|ከታዩ|ከታየ|ካጋጠመ|ካጋጠሙ|ካጋጠመዎት|ቢከሰት|ቢያጋጥም|ቢኖር|ከቀጠለ|ካልቀነሰ|ካልተሻለ|ካልጠፋ|ቢቆይ|ከጨመረ|ቢታዩ|ቢጀምር)")
OPEN = re.compile(r"^(ስለሆነም[፣,]?\s*|ስለዚህ[፣,]?\s*|በመሆኑም[፣,]?\s*)?ለ(ተሟላ|ተሟላው|ተገቢ|ተገቢው|ተገቢውን|በለጠ|ተሻለ|ትክክለኛ|ዝርዝር|ተጨማሪ)")
SENT = re.compile(r"(?<=[።!?፧])\s*|(?<=\.)\s+")


def sentences(t):
    return [s.strip() for s in SENT.split(t.strip()) if s.strip()]


def is_ref_sentence(s):
    return bool(REF_LINE_WORDS.search(s) and (DIRECTIVE.search(s) or URGENT.search(s)))


def is_boilerplate(s):
    """Generic, unconditional referral: a referral sentence with no condition (ካለ / ከሆነ / ከታዩ ...) and no urgency word,
    e.g. 'for a full examination and proper treatment go to a health facility and consult a health professional'."""
    return bool(is_ref_sentence(s) and not COND.search(s) and not URGENT.search(s))


def referral_line(t):
    return any(is_ref_sentence(s) for s in sentences(t))


def boilerplate_line(t):
    return any(is_boilerplate(s) for s in sentences(t))


def wide_keyword(t):
    return bool(REF_WIDE.search(t))


def strip_boilerplate(t):
    """Remove generic boilerplate sentences (any position); keep conditional / specific referral clauses."""
    ss = sentences(t)
    kept = [s for s in ss if not is_boilerplate(s)]
    return " ".join(kept), len(ss) - len(kept)


LATIN_RE = re.compile(r"[a-z\u00C0-\u024F\u1E00-\u1EFF]")
