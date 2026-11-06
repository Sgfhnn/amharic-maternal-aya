"""Gradio demo: Tiny Aya Earth fine-tuned for Amharic maternal and reproductive health Q&A (v4c DPO, merged).

Works on: a ZeroGPU Space (`spaces` is used when present), a plain GPU Space, or CPU basic (slow: a 3.35B model in
bf16 on 2 vCPU takes a minute or more per answer, so answers are capped at 96 new tokens there).
The model repo is public; HF_TOKEN is optional (only needed if MODEL_ID points at a private or gated repo).
Research prototype, not medical advice. Non-commercial use only (CC BY-NC 4.0).
"""
import os
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "5LeaderW/tiny-aya-earth-amharic-maternal-health")
TOKEN = os.environ.get("HF_TOKEN") or None  # optional; the model repo is public

try:
    import spaces  # ZeroGPU Spaces
    gpu = spaces.GPU(duration=120)
except Exception:  # plain GPU or CPU Space
    def gpu(f):
        return f

CUDA = torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU")
DTYPE = torch.float16 if CUDA else torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=TOKEN, dtype=DTYPE, low_cpu_mem_usage=True)
if CUDA:
    model.to("cuda")  # on ZeroGPU this is deferred until a GPU is attached
model.eval()
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "256" if CUDA else "96"))  # eval used 256
STOP = sorted({i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|END_RESPONSE|>"),
                           tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")] if isinstance(i, int) and i >= 0})

DISCLAIMER = ("**ይህ የምርምር ማሳያ ነው፤ የህክምና ምክር አይደለም።** This is a research demo, **not medical advice**, and answers can "
              "be wrong. If you are pregnant and have heavy bleeding, a severe headache or blurred vision, convulsions, "
              "high fever, reduced baby movements, waters breaking early or severe abdominal pain, go to a health "
              "facility now.\n\n**Non-commercial use only.** The model is licensed CC BY-NC 4.0 under the Tiny Aya "
              "licence (CohereLabs/tiny-aya-earth) and Cohere Labs' Acceptable Use Policy. "
              "Model: [5LeaderW/tiny-aya-earth-amharic-maternal-health]"
              "(https://huggingface.co/5LeaderW/tiny-aya-earth-amharic-maternal-health).")
EXAMPLES = ["ነፍሰ ጡር ሴት ምን አይነት ምግብ መመገብ አለባት?",
            "በእርግዝና ወቅት የደም መፍሰስ ቢያጋጥመኝ ምን ማድረግ አለብኝ?",
            "የቤተሰብ ምጣኔ ዘዴዎች ምን ምን ናቸው?",
            "ጡት ማጥባት ለህፃኑ ምን ጥቅም አለው?"]


@gpu
def answer(message, history):
    msgs = []
    for turn in history or []:
        if isinstance(turn, dict):
            msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": message})
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**enc, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=STOP,  # greedy, as in the reported eval
              pad_token_id=tok.pad_token_id, streamer=streamer)
    Thread(target=model.generate, kwargs=kw).start()
    out = ""
    for piece in streamer:
        out += piece
        yield out


demo = gr.ChatInterface(
    answer,
    type="messages",
    title="Amharic maternal and reproductive health assistant (Tiny Aya Earth, v4c DPO)",
    description=DISCLAIMER,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
