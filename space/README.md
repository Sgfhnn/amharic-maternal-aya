---
title: Amharic Maternal Health Aya
emoji: 🤰
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
models:
- 5LeaderW/tiny-aya-earth-amharic-maternal-health
short_description: Amharic maternal health Q&A, Tiny Aya Earth fine-tune
---

# Amharic maternal and reproductive health Q&A (Tiny Aya Earth, v4c DPO)

Gradio chat demo for [`5LeaderW/tiny-aya-earth-amharic-maternal-health`](https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health),
Tiny Aya Earth (3.35B) fine-tuned with CuratorKIT -> AlignTune (SFT + DPO) -> AuditKIT for the Lexsi Labs hackathon.
Research prototype, **not medical advice**. Non-commercial use only (CC BY-NC 4.0, inherited from CohereLabs/tiny-aya-earth).

## Notes

- Hardware: ZeroGPU when available (free), otherwise CPU basic (free, slow; answers capped at 96 new tokens).
- Decoding is greedy, as in the reported evaluation. Optional env vars: `MODEL_ID`, `MAX_NEW_TOKENS`.
- Code: MIT. Model and data: CC BY-NC 4.0, non-commercial only (Tiny Aya licence).
