"""LoRA SFT of Tiny Aya Earth on the Amharic health chat data with AlignTune (TRL backend).

Falls back to plain TRL SFTTrainer (same hyper-parameters) if AlignTune fails, and records which ran.
Loss is on assistant tokens only: the stock Cohere chat template is copied with {% generation %}
tags around the assistant turn (rendered text unchanged) and assistant_only_loss=True.

Usage: python train_sft.py --out /kaggle/working/results/sft [--stop_hours 4.25] [--epochs 1]
"""
import argparse, json, os, shutil, sys, time, glob, traceback

T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}] [{time.time() - T0:7.0f}s]", *a, flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def prep_model(src_id, dst="/tmp/tinyaya"):
    if os.path.isdir(src_id):
        shutil.copytree(src_id, dst, dirs_exist_ok=True); p = dst
    else:
        from huggingface_hub import snapshot_download
        p = snapshot_download(src_id, local_dir=dst)
    tc = json.load(open(f"{p}/tokenizer_config.json"))
    t = tc.get("chat_template")
    if os.path.exists(f"{p}/chat_template.jinja") and not t:
        t = open(f"{p}/chat_template.jinja").read()
    if isinstance(t, list):
        t = [x for x in t if x["name"] == "default"][0]["template"]
    old = "<|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>"
    assert old in t, "unexpected chat template"
    t2 = t.replace(old, "<|CHATBOT_TOKEN|><|START_RESPONSE|>{% generation %}{{ message['content'] }}<|END_RESPONSE|>{% endgeneration %}")
    open(f"{p}/chat_template.jinja", "w").write(t2)
    tc.pop("chat_template", None)
    json.dump(tc, open(f"{p}/tokenizer_config.json", "w"), ensure_ascii=False, indent=1)
    return p


def prep_data(dst="/tmp/sftdata"):
    for sp, name in [("train", "train"), ("dev", "val")]:
        os.makedirs(f"{dst}/{name}", exist_ok=True)
        src = find(f"chat_{sp}.jsonl")
        n = 0
        with open(f"{dst}/{name}/sft.jsonl", "w", encoding="utf-8") as f:
            for l in open(src, encoding="utf-8"):
                r = json.loads(l)
                f.write(json.dumps({"prompt": [r["messages"][0]], "completion": [r["messages"][1]]}, ensure_ascii=False) + "\n")
                n += 1
        log("data", name, n)
    return dst


def time_cb(stop_s):
    from transformers import TrainerCallback

    class StopAfter(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if time.time() - T0 > stop_s:
                log(f"time budget hit at step {state.global_step}; stopping")
                control.should_training_stop = True
                control.should_save = True
            if state.global_step % 25 == 0:
                try:
                    json.dump({"step": state.global_step, "max_steps": state.max_steps, "elapsed_s": time.time() - T0,
                               "log": state.log_history[-3:]}, open(os.environ.get("PROGRESS", "/kaggle/working/progress.json"), "w"))
                except Exception:
                    pass
            return control
    return StopAfter()


TEST = os.environ.get("SFT_TEST") == "1"  # CPU smoke test: fp32, 2 steps
HP = dict(epochs=1, lr=2e-4, r=16, alpha=32, dropout=0.05, max_len=1024, bs=2, grad_acc=8,
          targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


if TEST:
    HP.update(bs=1, grad_acc=1, max_len=512)


def run_aligntune(model_dir, data_dir, out, stop_s):
    from aligntune.core.backend_factory import create_sft_trainer
    t = create_sft_trainer(
        model_name=model_dir, dataset_name=data_dir, backend="trl", output_dir=out,
        num_epochs=HP["epochs"], batch_size=HP["bs"], gradient_accumulation_steps=HP["grad_acc"],
        learning_rate=HP["lr"], max_seq_length=HP["max_len"], warmup_ratio=0.03, lr_scheduler="cosine",
        use_peft=True, lora_r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"],
        lora_target_modules=HP["targets"], precision="fp32" if TEST else "fp16", fp16=not TEST,
        **({"max_steps": 2, "device_map": "cpu"} if TEST else {"device_map": "auto"}),
        assistant_only_loss=True, use_gradient_checkpointing=True, gradient_checkpointing=True,
        eval_strategy="steps", eval_interval=150, save_interval=150, save_strategy="steps",
        load_best_model_at_end=False, logging_steps=10, loggers=[], report_to="none",
        compute_rouge=False, compute_bleu=False, compute_perplexity=False, seed=42)
    orig = t.get_hf_callbacks
    t.get_hf_callbacks = lambda: (orig() or []) + [time_cb(stop_s)]
    res = t.train()
    hist = t.trainer.state.log_history
    return res, hist, t.trainer


def run_trl(model_dir, data_dir, out, stop_s):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    def to_msgs(r):
        return {"messages": r["prompt"] + r["completion"]}
    tr = load_dataset("json", data_files=f"{data_dir}/train/sft.jsonl")["train"].map(to_msgs, remove_columns=["prompt", "completion"])
    dv = load_dataset("json", data_files=f"{data_dir}/val/sft.jsonl")["train"].map(to_msgs, remove_columns=["prompt", "completion"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, device_map="auto")
    cfg = SFTConfig(output_dir=out, num_train_epochs=HP["epochs"], per_device_train_batch_size=HP["bs"],
                    per_device_eval_batch_size=HP["bs"], gradient_accumulation_steps=HP["grad_acc"],
                    learning_rate=HP["lr"], lr_scheduler_type="cosine", warmup_steps=0.03, max_length=HP["max_len"],
                    assistant_only_loss=True, fp16=True, gradient_checkpointing=True, logging_steps=10,
                    eval_strategy="steps", eval_steps=150, save_steps=150, save_total_limit=2, report_to="none", seed=42)
    peft_cfg = LoraConfig(r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"], target_modules=HP["targets"],
                          task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=tr, eval_dataset=dv, processing_class=tok,
                         peft_config=peft_cfg, callbacks=[time_cb(stop_s)])
    res = trainer.train()
    trainer.save_model(out)
    return {"train_loss": res.training_loss, "steps": res.global_step}, trainer.state.log_history, trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/results/sft")
    ap.add_argument("--model_id", default="1-800-LLMs/tiny-aya-earth")
    ap.add_argument("--stop_hours", type=float, default=4.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model_dir = prep_model(a.model_id)
    data_dir = prep_data()
    ck = f"{a.out}/ckpt"
    info = {"hp": HP, "model_id": a.model_id, "stop_hours": a.stop_hours}
    trainer = None
    try:
        res, hist, trainer = run_aligntune(model_dir, data_dir, ck, a.stop_hours * 3600)
        info["framework"] = "aligntune (TRL backend)"
    except Exception as e:
        info["aligntune_error"] = "".join(traceback.format_exception_only(type(e), e))[-2000:]
        log("AlignTune failed:", info["aligntune_error"]); traceback.print_exc()
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
        res, hist, trainer = run_trl(model_dir, data_dir, ck, a.stop_hours * 3600)
        info["framework"] = "trl SFTTrainer (fallback)"
    # final eval loss on dev
    try:
        info["final_eval"] = trainer.evaluate()
    except Exception as e:
        info["final_eval_error"] = str(e)[:500]
    adir = f"{a.out}/adapter"
    trainer.model.save_pretrained(adir)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        if os.path.exists(f"{model_dir}/{f}"):
            shutil.copy(f"{model_dir}/{f}", adir)
    info["result"] = res if isinstance(res, dict) else str(res)
    info["log_history"] = hist
    info["train_seconds"] = time.time() - T0
    json.dump(info, open(f"{a.out}/train_info.json", "w"), indent=2, default=str)
    shutil.rmtree(ck, ignore_errors=True)  # keep only final adapter in outputs
    log("saved adapter", adir)


if __name__ == "__main__":
    main()
