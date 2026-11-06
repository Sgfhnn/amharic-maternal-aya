"""LoRA SFT of Tiny Aya Earth on the Amharic health chat data with AlignTune (TRL backend).

Falls back to plain TRL SFTTrainer (same hyper-parameters) if AlignTune fails, and records which ran.
Loss is on assistant tokens only: the stock Cohere chat template is copied with {% generation %}
tags around the assistant turn (rendered text unchanged) and assistant_only_loss=True.

Run two changes (2026-11-05):
- Dev eval_loss was NaN in run one because 119/374 dev rows (and 2,046/7,109 train rows, all ethionlp_general)
  have a prompt (Cohere preamble ~360 tokens + long article) that already fills max_length=1024. TRL truncates
  from the right, so every assistant label is masked; an eval batch with zero supervised tokens gives 0/0 = NaN,
  which poisons the mean eval loss. (Train loss stayed finite because TRL normalises by num_items_in_batch over
  the accumulated batch.) Fix: a length guard drops rows with fewer than --min_asst_tokens supervised tokens
  inside max_length, applied identically to every model; counts go to train_info.json.
- LoRA params forced to fp32 (base frozen fp16, fp16 autocast), lr 1e-4, max_grad_norm 1.0, NaN watcher.
- --model_dir loads a local snapshot (assembled CohereLabs/tiny-aya-earth); --train_file/--dev_file pick data.

Usage: python train_sft.py --out DIR --model_dir /tmp/tinyaya_official --train_file chat_train.jsonl --dev_file chat_dev.jsonl
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


def prep_model(src_id, dst="/tmp/tinyaya_train"):
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
    if "{% generation %}" in t:  # already tagged (run three: merged snapshots)
        t = t.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    assert old in t, "unexpected chat template"
    t2 = t.replace(old, "<|CHATBOT_TOKEN|><|START_RESPONSE|>{% generation %}{{ message['content'] }}<|END_RESPONSE|>{% endgeneration %}")
    open(f"{p}/chat_template.jinja", "w").write(t2)
    tc.pop("chat_template", None)
    json.dump(tc, open(f"{p}/tokenizer_config.json", "w"), ensure_ascii=False, indent=1)
    return p


def prep_data(model_dir, train_file, dev_file, min_asst, dev_limit=0, dst="/tmp/sftdata"):
    """Write AlignTune prompt/completion files, dropping rows whose assistant span would be fully (or almost
    fully) truncated at max_length. Returns per-split stats."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    stats = {}
    for fn, name in [(train_file, "train"), (dev_file, "val")]:
        os.makedirs(f"{dst}/{name}", exist_ok=True)
        src = find(fn)
        st = {"file": fn, "in": 0, "kept": 0, "dropped_no_supervised_tokens": 0, "dropped_by_source": {},
              "empty_assistant": 0, "answer_truncated_kept": 0}
        with open(f"{dst}/{name}/sft.jsonl", "w", encoding="utf-8") as f:
            for l in open(src, encoding="utf-8"):
                r = json.loads(l)
                st["in"] += 1
                u, a = r["messages"][0], r["messages"][1]
                if not a["content"].strip():
                    st["empty_assistant"] += 1; continue
                enc = tok.apply_chat_template([u], add_generation_prompt=True, tokenize=True)
                ids = enc["input_ids"] if hasattr(enc, "keys") else enc
                if ids and isinstance(ids[0], list): ids = ids[0]
                p_len = len(ids)
                a_len = len(tok(a["content"], add_special_tokens=False)["input_ids"]) + 2
                room = HP["max_len"] - p_len - 1
                if room < min_asst:
                    st["dropped_no_supervised_tokens"] += 1
                    s = r.get("source", "?"); st["dropped_by_source"][s] = st["dropped_by_source"].get(s, 0) + 1
                    continue
                if a_len > room:
                    st["answer_truncated_kept"] += 1
                if name == "val" and dev_limit and st["kept"] >= dev_limit:
                    continue
                f.write(json.dumps({"prompt": [u], "completion": [a]}, ensure_ascii=False) + "\n")
                st["kept"] += 1
        log("data", name, st)
        stats[name] = st
    return dst, stats


def time_cb(stop_s):
    from transformers import TrainerCallback

    class StopAfter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            import math
            for k in ("loss", "eval_loss", "grad_norm"):
                v = (logs or {}).get(k)
                if isinstance(v, float) and not math.isfinite(v):
                    NAN_EVENTS.append({"step": state.global_step, "key": k, "value": str(v)})
                    log("NON-FINITE", k, v, "at step", state.global_step)
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


NAN_EVENTS = []
TEST = os.environ.get("SFT_TEST") == "1"  # CPU smoke test: fp32, 2 steps
MAX_STEPS, EVAL_STEPS = 0, 150
HP = dict(epochs=1, lr=1e-4, max_grad_norm=1.0, r=16, alpha=32, dropout=0.05, max_len=1024, bs=2, grad_acc=8,
          targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


if TEST:
    HP.update(bs=1, grad_acc=1, max_len=512)


def fp32_trainable(model):
    import torch
    n = 0; dts = {}
    for p in model.parameters():
        if p.requires_grad:
            if p.dtype != torch.float32:
                p.data = p.data.float(); n += 1
            dts[str(p.dtype)] = dts.get(str(p.dtype), 0) + p.numel()
    frozen = {}
    for p in model.parameters():
        if not p.requires_grad:
            frozen[str(p.dtype)] = frozen.get(str(p.dtype), 0) + p.numel()
    info = {"recast_to_fp32": n, "trainable_dtypes": dts, "frozen_dtypes": frozen}
    log("param dtypes", info)
    return info


def run_aligntune(model_dir, data_dir, out, stop_s, info):
    from aligntune.core.backend_factory import create_sft_trainer
    t = create_sft_trainer(
        model_name=model_dir, dataset_name=data_dir, backend="trl", output_dir=out,
        num_epochs=HP["epochs"], batch_size=HP["bs"], gradient_accumulation_steps=HP["grad_acc"],
        learning_rate=HP["lr"], max_seq_length=HP["max_len"], warmup_ratio=0.03, lr_scheduler="cosine",
        use_peft=True, lora_r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"],
        lora_target_modules=HP["targets"], precision="fp32" if TEST else "fp16", fp16=not TEST,
        max_grad_norm=HP["max_grad_norm"],
        **({"max_steps": 2, "device_map": "cpu"} if TEST else {"device_map": "auto"}),
        **({"max_steps": MAX_STEPS} if MAX_STEPS and not TEST else {}),
        assistant_only_loss=True, use_gradient_checkpointing=True, gradient_checkpointing=True,
        eval_strategy="steps", eval_interval=EVAL_STEPS, save_interval=10**6, save_strategy="steps",
        load_best_model_at_end=False, logging_steps=10, loggers=[], report_to="none",
        compute_rouge=False, compute_bleu=False, compute_perplexity=False, seed=42)
    orig = t.get_hf_callbacks
    t.get_hf_callbacks = lambda: (orig() or []) + [time_cb(stop_s)]
    t.setup_model()
    info["param_dtypes"] = fp32_trainable(t.model)
    res = t.train()
    hist = t.trainer.state.log_history
    return res, hist, t.trainer


def run_trl(model_dir, data_dir, out, stop_s, info):
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
                    max_grad_norm=HP["max_grad_norm"], max_steps=MAX_STEPS or -1,
                    assistant_only_loss=True, fp16=True, gradient_checkpointing=True, logging_steps=10,
                    eval_strategy="steps", eval_steps=EVAL_STEPS, save_strategy="no", report_to="none", seed=42)
    peft_cfg = LoraConfig(r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"], target_modules=HP["targets"],
                          task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=tr, eval_dataset=dv, processing_class=tok,
                         peft_config=peft_cfg, callbacks=[time_cb(stop_s)])
    info["param_dtypes"] = fp32_trainable(trainer.model)
    res = trainer.train()
    trainer.save_model(out)
    return {"train_loss": res.training_loss, "steps": res.global_step}, trainer.state.log_history, trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/results/sft")
    ap.add_argument("--model_id", default="CohereLabs/tiny-aya-earth", help="recorded id (or hub id if no --model_dir)")
    ap.add_argument("--model_dir", default=None, help="local snapshot (assemble_model.py output)")
    ap.add_argument("--train_file", default="chat_train.jsonl")
    ap.add_argument("--dev_file", default="chat_dev.jsonl")
    ap.add_argument("--stop_hours", type=float, default=4.25)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_steps", type=int, default=150)
    ap.add_argument("--dev_limit", type=int, default=0)
    ap.add_argument("--min_asst_tokens", type=int, default=16)
    a = ap.parse_args()
    global MAX_STEPS, EVAL_STEPS
    MAX_STEPS, EVAL_STEPS = a.max_steps, a.eval_steps
    os.makedirs(a.out, exist_ok=True)
    model_dir = prep_model(a.model_dir or a.model_id)
    data_dir, dstats = prep_data(model_dir, a.train_file, a.dev_file, a.min_asst_tokens, a.dev_limit)
    ck = f"{a.out}/ckpt"
    info = {"hp": HP, "model_id": a.model_id, "model_dir": a.model_dir, "stop_hours": a.stop_hours,
            "max_steps": a.max_steps, "eval_steps": a.eval_steps, "data": dstats, "min_asst_tokens": a.min_asst_tokens}
    trainer = None
    try:
        res, hist, trainer = run_aligntune(model_dir, data_dir, ck, a.stop_hours * 3600, info)
        info["framework"] = "aligntune (TRL backend)"
    except Exception as e:
        info["aligntune_error"] = "".join(traceback.format_exception_only(type(e), e))[-2000:]
        log("AlignTune failed:", info["aligntune_error"]); traceback.print_exc()
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
        res, hist, trainer = run_trl(model_dir, data_dir, ck, a.stop_hours * 3600, info)
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
    info["nan_events"] = NAN_EVENTS
    import math
    fe = info.get("final_eval", {}).get("eval_loss")
    info["sanity_ok"] = (not NAN_EVENTS) and isinstance(fe, float) and math.isfinite(fe)
    log("SANITY", "OK" if info["sanity_ok"] else "FAIL", "final eval_loss", fe, "nan events", len(NAN_EVENTS))
    info["train_seconds"] = time.time() - T0
    json.dump(info, open(f"{a.out}/train_info.json", "w"), indent=2, default=str)
    shutil.rmtree(ck, ignore_errors=True)  # keep only final adapter in outputs
    log("saved adapter", adir)


if __name__ == "__main__":
    main()
