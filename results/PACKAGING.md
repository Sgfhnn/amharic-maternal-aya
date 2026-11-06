# Submission packaging - PUBLIC since 2026-11-06 ~16:45 EAT (Anu approved going public 16:13 EAT)

Submission model: **v4c DPO** = official CohereLabs/tiny-aya-earth + run-2 B adapter + v3 SFT adapter + v4c DPO adapter.
Deadline: 2026-11-15 14:59 EAT.

## Public links

| What | URL |
|---|---|
| Models collection | https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-models-6aedaabef1e7b1fa856a2eec |
| Data and logs collection | https://huggingface.co/collections/5LeaderW/amharic-maternal-health-tiny-aya-data-and-logs-6aedaabfea4165a8357c79d6 |
| Merged model (v4c, fp16, 6.70 GB) | https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health |
| LoRA adapters (run2_B, v3_sft, v3_dpo, v4b_dpo, v4c_dpo) | https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health-adapters |
| Curated SFT data (licence-cleaned) | https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-sft-curated-v3 |
| DPO pairs (licence-cleaned) | https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-dpo-pairs |
| Eval results and logs | https://huggingface.co/datasets/5LeaderW/amharic-maternal-health-aya-eval-logs |
| Demo Space (Gradio, ZeroGPU, free) | https://huggingface.co/spaces/5LeaderW/amharic-maternal-health-aya |
| GitHub (not pushed yet; main agent pushes) | https://github.com/Sgfhnn/amharic-maternal-aya |
| HF article | not published: community articles need the web UI; the GitHub README / docs/article.md serve as the article |

HF user 5LeaderW; token "Meta Hackaton" (fine-grained, vault-injected on the wire; the sandbox env value is a placeholder,
so it cannot be handed to a Space as a secret).

## Licence cleanup (done 2026-11-06, Anu approved)
Removed every row from danilior_maternal and ethionlp_general (no stated licence) from the public dataset repos, plus
every DPO pair derived from them (matched by source in the row id). Repo history squashed on both dataset repos so the
old rows are not in git history. Verified: 30,497 published rows, 0 from those sources.

| File | Before -> after | danilior | EthioNLP |
|---|---|---|---|
| SFT v3/train (+ chat copy) | 2,691 -> 2,320 | 59 | 312 |
| SFT v3/dev = run2/dev (+ chat copies) | 249 -> 210 | 4 | 35 |
| SFT run2/train (+ chat copy) | 4,806 -> 4,053 | 100 | 653 |
| SFT v3/rejected, run2/rejected | 2,411 -> 2,015; 130 -> 104 | 55; 1 | 341; 25 |
| SFT LLM logs: quality_scores, referral_labels, rewrites, rewrite_checks, run2 llm_checks | 5,358->4,529; 5,453->5,325; 528->499; 526->497; 650->635 | 108; 128; 29; 29; 15 | 721; 0; 0; 0; 0 |
| DPO v3 train/dev | 763/40 -> 726/35 | 37/5 | 0 |
| DPO v4 train/dev | 425/25 -> 392/22 | 33/3 | 0 |
| DPO v4b train/dev | 915/53 -> 856/47 | 59/6 | 0 |
| DPO v4c train/dev | 1,086/62 -> 1,032/57 | 54/5 | 0 |

Per-file report: work/hf_public/removal_report.json. Published copies: work/hf_public/ (work/hf_stage keeps the
original private-era files, unchanged). Dataset cards state the removal and the new counts; curation_report.md files got
a header note that their counts describe the original run. Model and adapter cards disclose that the models were
trained on those rows and were not retrained. Eval-logs repo had no rows from these sources (test = AmHQA + Walelign).
Everything stays CC BY-NC 4.0.

## Gating
`gated="auto"` worked via the API, but the Space then failed with 401 (the vault token cannot be given to a Space), so
gating was turned OFF again. The model card carries the non-commercial / not-medical-advice notice and keeps the
`extra_gated_*` agreement in its YAML, so re-gating is one call (`update_repo_settings(gated="auto")`) once a real read
token is added as Space secret HF_TOKEN (app.py already reads it).

## Space
Created public with hardware zero-a10g (ZeroGPU, free). A private Gradio Space was refused (402: free cpu-basic
Gradio/Docker Spaces now need PRO). Pinned gradio 5.49.1, Python 3.12. Model loads from the public merged repo.
Test 2026-11-06 ~16:50 EAT in a browser (anonymous API calls from the sandbox got "exceeded your ZeroGPU runs limit"):
"ነፍሰ ጡር ሴት ምን አይነት ምግብ መመገብ አለባት?" -> "ነፍሰ ጡር ሴቶች በቂ ፕሮቲን፣ የብረት፣ የካልሲየም እና የቫይታሚን አቅርቦት የሚያረጋግጥ
ምግብ መመገብ አለባቸው። በበለጸጉ ፍራፍሬዎች፣ አትክልቶች፣ ሙሉ እህሎች እና በቂ ፈሳሽ መጠጣት ጤናማ የልጅ እድገት እንዲኖር ይረዳል።"
(protein, iron, calcium, vitamins; fruit, vegetables, whole grains, fluids), about 4 s of GPU time. Disclaimer with
"not medical advice" and "Non-commercial use only" is shown. Source: space/ (same as work/space_public/).

## GitHub (local only, not pushed)
work/github_repo/amharic-maternal-aya: MIT LICENSE for code, NOTICE.md (model and data CC BY-NC 4.0 + Tiny Aya
licence), README links to the public repos, account email removed from scripts/kaggle/pull_*.sh. History squashed to
one commit so the old email is not in git history. Push: Anu creates the empty repo Sgfhnn/amharic-maternal-aya and adds
it to the Hark GitHub App, then `git remote add origin https://github.com/Sgfhnn/amharic-maternal-aya.git && git push -u origin main`.

## Merge (free Kaggle, private kernel neimashube/amh-tinyaya-merge-v4c, version 2, 6 min)
16 test rows: merged vs adapter path 16/16 identical; merged vs stored v4c predictions 13/16 identical (mean chrF 97.7);
the same 3 rows differ for the adapter path, so the cause is batch padding, not the merge. results/merge_check/.
Kaggle datasets and kernels stay private.
