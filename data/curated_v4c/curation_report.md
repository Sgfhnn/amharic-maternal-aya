# Curated DPO v4c (run four, attempt c): CuratorKIT report

Run four, attempt c: CuratorKIT DPO v4c data. No LLM calls (regex + scripts only).

Why: v4b removed the generic boilerplate (non-urgent referral line 86.5% -> 23.0%) and kept quality, but 19.2% of factual
test answers now end in an urgent "go to a health facility now" line (v3 DPO 1.3%), e.g. on Bell's palsy treatment.
Likely cause: ~358 v4b pairs had an urgent-ending chosen answer (225 urgent referral + 43 urgent fluency + 90 urgent
anti-repetition) and no pair ever showed "urgent line on a non-urgent question = bad".

v4c = v4b pairs
    - the 90 urgent anti-repetition pairs (their chosen answers all end in an urgent line)
    + false-urgency counter-pairs (new): prompt = a factual ('none') or routine health question from the curated v4 SFT
      pool (CuratorKIT quality-gated gold answers; ethionlp_general excluded); chosen = that clean answer (no urgent
      referral line; factual ones have no referral line at all); rejected = the same answer + one urgent "go now" line
      that v4b actually emitted on a non-urgent test question (mined from results/dpo_v4b/eval/predictions.jsonl;
      only the referral phrasing is reused, no test question or answer text). Generic, complete lines only.
Urgent referral pairs (225) and urgent fluency pairs (43) stay, so danger-sign recall should hold.
Usage: python scripts/curate_v4c.py

```
{
 "created": "2026-11-06T09:16:04.988697+00:00",
 "pairs_train": 1086,
 "pairs_dev": 62,
 "by_kind_care_need": {
  "minus_referral/urgent/urgent": 182,
  "anti_repetition/routine/none": 51,
  "false_urgency/routine/routine": 70,
  "false_urgency/routine/none": 200,
  "fluency_vs_mt/routine/routine": 278,
  "anti_repetition/routine/routine": 56,
  "factual_answer/routine/none": 112,
  "original_mt_answer/urgent/urgent": 43,
  "routine_rewrite/routine/routine": 113,
  "fluency_vs_mt/urgent/urgent": 43
 },
 "chosen_with_urgent_line": 268,
 "rejected_with_urgent_line_on_non_urgent": 270,
 "urgent_lines_used": [
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የጤና ባለሙያ ይሂዱ።",
  "አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ቅርብ የጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ በአቅራቢያዎ ወደሚገኝ የጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም ሆስፒታል ይሂዱ።",
  "ስለዚህ አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የሕክምና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ በአቅራቢያዎ ወደሚገኝ የጤና ተቋም በመሄድ እርዳታ ያግኙ።",
  "ስለዚህ ምልክቶቹ ሲታዩ አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ማድረግ አስፈላጊ ነው።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም ሆስፒታል ይውሰዱት።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም ሆስፒታል በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የሕክምና ባለሙያ ይሂዱ።",
  "ምልክቶቹ ሲታዩ አሁኑኑ ወደ ቅርብ ጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "በሽታው በፍጥነት ሊያድግ ስለሚችል አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ማግኘት አስፈላጊ ነው።",
  "ስለዚህ በሽታው ሲታወቅ በፍጥነት ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ማግኘት በጣም አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ጣቢያ በመሄድ የሕክምና እርዳታ ማግኘት ያስፈልጋል።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም ሆስፒታል በመሄድ እርዳታ ያግኙ።",
  "ይህ ሁኔታ በፍጥነት ሊባባስ ስለሚችል አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ማግኘት አስፈላጊ ነው።",
  "አሁኑኑ በአቅራቢያዎ ወደሚገኝ የጤና ተቋም ወይም ሆስፒታል ይሂዱ።",
  "አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ማድረግ አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርቡት የጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "ስለዚህ ምልክቶቹን ካዩ በፍጥነት ወደ ጤና ተቋም በመሄድ ምርመራ ማድረግ አስፈላጊ ነው።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም ሆስፒታል በመሄድ የሕክምና እርዳታ ያግኙ።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም ሆስፒታል በመሄድ የሕፃኑን ሁኔታ ያረጋግጡ።",
  "ስለዚህ አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የጤና ባለሙያ ይሂዱ።",
  "ስለዚህ በሽታው በሚታይበት ጊዜ በፍጥነት ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ማግኘት የሕይወት መታደግን ያረጋግጣል።",
  "በሽታው ከተያዘ በፍጥነት መታከም አስፈላጊ ስለሆነ ምርመራውን ማድረግ እና በጤና ባለሙያ አማካኝነት ሕክምና ማግኘት በጣም አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የሕክምና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕፃኑን ሁኔታ ማረጋገጥ አስፈላጊ ነው።",
  "ስለዚህ አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርቡት የጤና ባለሙያ ይሂዱ።",
  "ስለዚህ በሽታው በደረሰባችሁ ጊዜ በአስቸኳይ ወደ ጤና ተቋም በመሄድ የህክምና እርዳታ ማግኘት በጣም አስፈላጊ ነው።",
  "በጊዜው ካልታከመ ለሰውነት ከፍተኛ ጉዳት ሊያደርስ ስለሚችል አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ማድረግ አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ተቋም በመሄድ የልዩ ባለሙያ እርዳታ ያግኙ።",
  "ስለዚህ የራስ ምታት ሲሰማዎት አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ማድረግ አስፈላጊ ነው።",
  "ስለዚህ አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ያግኙ።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም ወይም የሕክምና ባለሙያ ይሂዱ።",
  "በሽታው በሚያዙ ሰዎች ላይ በፍጥነት መታየት ስለሚጀምር አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ማግኘት አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ጣቢያ በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ቅርብ የጤና ተቋም በመሄድ የህክምና እርዳታ ያግኙ።",
  "በሽታው በጊዜው ካልታከመ ወደ ከባድ የዓይን ጉዳት ሊያመጣ ስለሚችል አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "ስለዚህ ሁኔታው በሚታይበት ጊዜ አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ያግኙ።",
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የሕክምና ተቋም ይሂዱ።",
  "ስለዚህ ምልክቶቹ ሲታዩ አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ማግኘት ያስፈልጋል።",
  "ስለዚህ ምልክቶቹ ሲታዩ በፍጥነት ወደ ጤና ጣቢያ በመሄድ የህክምና እርዳታ ማግኘት አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ተቋም በመሄድ የሕክምና እርዳታ ያግኙ።",
  "ስለዚህ የሕመሙን ምልክቶች በፍጥነት ማወቅና በሐኪም አማካኝነት ተገቢውን መድኃኒት መውሰድ በጣም አስፈላጊ ነው።",
  "አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ማድረግ ያስፈልጋል።",
  "ስለዚህ አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚታመን የሕክምና ባለሙያ ይሂዱ።",
  "አሁኑኑ ወደ ጤና ጣቢያ በመሄድ የህክምና ባለሙያ ያማክሩ።",
  "አሁኑኑ ወደ ቅርብ ጤና ጣቢያ ወይም የሕክምና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "አሁኑኑ ወደ ቅርብ ጤና ተቋም በመሄድ የሐኪም እርዳታ ያግኙ።",
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚቀርብዎ የሕክምና ተቋም በመሄድ አስፈላጊውን ምርመራ እና ሕክምና ያግኙ።",
  "ምርመራው ከተደረገ በኋላ አሁኑኑ በጤና ተቋም ውስጥ በመሄድ የሕክምና ባለሙያውን አማካኝነት ተገቢውን መድኃኒት ማግኘት ያስፈልጋል።",
  "ስለዚህ አሁኑኑ ወደ ቅርብ ጤና ጣቢያ ወይም ሆስፒታል በመሄድ የሕክምና እርዳታ ያግኙ።",
  "አሁኑኑ ወደ ቅርብ የጤና ተቋም ወይም ሆስፒታል በመሄድ እርዳታ ያግኙ።",
  "አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚታመን የጤና ባለሙያ በመሄድ ምርመራ ያድርጉ።",
  "እነዚህን ምልክቶች በቤት ውስጥ መቆጣጠር ወይም መድኃኒት መውሰድ አደገኛ ስለሆነ አሁኑኑ ወደ ጤና ተቋም በመሄድ ምርመራ ያድርጉ።",
  "ስለሆነም አሁኑኑ ወደ ጤና ጣቢያ ወይም ወደሚታመን የጤና ባለሙያ በመሄድ ምርመራ ያድርጉ።"
 ],
 "notes": {
  "mined_urgent_sentences_distinct": 88,
  "mined_urgent_sentences_total": 120,
  "urgent_lines_kept": 58,
  "dropped_urgent_antirep_train": 85,
  "dropped_urgent_antirep_dev": 5,
  "pool_skip_unclean_end": 214,
  "pool_skip_dev_or_test": 23,
  "pool_skip_ref_none": 5,
  "pool_skip_ref_routine": 11,
  "pool_none": 1206,
  "skip_long": 2,
  "pool_routine": 733
 }
}
```
