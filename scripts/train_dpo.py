"""Run three: LoRA DPO with AlignTune (TRL backend) on top of a merged SFT model.

The policy is a fresh LoRA on the merged v3-SFT model; with PEFT, TRL's DPOTrainer uses the same model with the
adapter disabled as the reference, so the reference is exactly the v3-SFT model. Data: data/curated_v3/dpo/*.jsonl
(conversational prompt / chosen / rejected). Same LoRA shape as SFT (r16, alpha32, all projections), beta 0.1,
1 epoch, max_len 1024, clip 1.0. LoRA params are kept fp32 under fp16 autocast (same fix as SFT run two).
No plain-TRL fallback on purpose: if AlignTune DPO fails, the error is recorded and the caller ships SFT-only.

Usage: python train_dpo.py --out DIR --model_dir /tmp/merged_v3 --train_file dpo_train.jsonl --dev_file dpo_dev.jsonl
"""
import argparse, json, math, os, shutil, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as S  # reuse prep_model (chat template w/ generation tags), find, log, time_cb, fp32_trainable, NAN_EVENTS

HP = dict(epochs=1, lr=5e-5, beta=0.1, max_grad_norm=1.0, r=16, alpha=32, dropout=0.05, max_len=1024, max_prompt_len=384,
          bs=2, grad_acc=8, targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
TEST = os.environ.get("SFT_TEST") == "1"
if TEST:
    HP.update(bs=1, grad_acc=1, max_len=256, max_prompt_len=128)


def prep(train_file, dev_file, dst="/tmp/dpodata"):
    st = {}
    for fn, name in [(train_file, "train"), (dev_file, "val")]:
        os.makedirs(f"{dst}/{name}", exist_ok=True)
        n = 0
        with open(f"{dst}/{name}/dpo.jsonl", "w", encoding="utf-8") as f:
            for l in open(S.find(fn), encoding="utf-8"):
                r = json.loads(l)
                f.write(json.dumps({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}, ensure_ascii=False) + "\n")
                n += 1
                if TEST and n >= 8:
                    break
        st[name] = n
    S.log("dpo data", st)
    return dst, st


def patch_dpo_config():
    """AlignTune builds trl.DPOConfig without fp16/bf16, and TRL then defaults to bf16, which a T4 rejects
    ("Your setup doesn't support bf16/gpu"). Inject the precision (and our clip / accumulation / checkpointing)
    into DPOConfig's constructor; AlignTune imports trl.DPOConfig at call time, so it picks this up."""
    import trl
    if getattr(trl.DPOConfig, "_run3_patched", False):
        return
    Orig = trl.DPOConfig

    class DPOConfigT4(Orig):
        _run3_patched = True

        def __init__(self, *a, **k):
            k["bf16"] = False
            k["fp16"] = not TEST
            if TEST:
                k["use_cpu"] = True
            k.setdefault("max_grad_norm", HP["max_grad_norm"])
            k.setdefault("gradient_accumulation_steps", HP["grad_acc"])
            k.setdefault("gradient_checkpointing", True)
            k.setdefault("max_length", HP["max_len"])
            k.setdefault("report_to", "none")
            super().__init__(*a, **k)
    trl.DPOConfig = DPOConfigT4


def run(model_dir, data_dir, out, stop_s, info, attempt):
    from aligntune.core.backend_factory import create_rl_trainer
    patch_dpo_config()
    kw = dict(model_name=model_dir, dataset_name=data_dir, algorithm="dpo", backend="trl", output_dir=out,
              num_epochs=HP["epochs"], batch_size=HP["bs"], gradient_accumulation_steps=HP["grad_acc"],
              learning_rate=HP["lr"], max_seq_length=HP["max_len"], max_prompt_length=HP["max_prompt_len"],
              beta=HP["beta"], loss_type="sigmoid", warmup_ratio=0.1, lr_scheduler="cosine",
              use_peft=True, lora_r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"],
              lora_target_modules=HP["targets"], precision="fp32" if TEST else "fp16", fp16=not TEST,
              max_grad_norm=HP["max_grad_norm"], use_gradient_checkpointing=True, gradient_checkpointing=True,
              eval_strategy="steps", eval_interval=20, save_interval=10**6, save_strategy="no", logging_steps=5,
              loggers=[], report_to="none", seed=42, reward_model_name=None,
              **({"max_steps": 2, "device_map": "cpu"} if TEST else {"device_map": "auto"}))
    if attempt == 2:  # second try: drop optional knobs that older/newer DPOConfig may reject
        for k in ["max_prompt_length", "warmup_ratio", "lr_scheduler", "eval_strategy", "save_strategy"]:
            kw.pop(k, None)
    t = create_rl_trainer(**kw)
    orig_setup = t.setup_model

    def setup_model_fp32():
        orig_setup()
        info["param_dtypes"] = S.fp32_trainable(t.model)
    t.setup_model = setup_model_fp32
    orig_cb = t.get_hf_callbacks
    t.get_hf_callbacks = lambda: (orig_cb() or []) + [S.time_cb(stop_s)]
    res = t.train()
    return res, t.trainer.state.log_history, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--train_file", default="dpo_train.jsonl")
    ap.add_argument("--dev_file", default="dpo_dev.jsonl")
    ap.add_argument("--stop_hours", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = S.prep_model(a.model_dir, dst="/tmp/tinyaya_dpo")
    data_dir, dstats = prep(a.train_file, a.dev_file)
    info = {"hp": HP, "model_dir": a.model_dir, "data": dstats, "attempts": []}
    ok = False
    for attempt in (1, 2):
        try:
            res, hist, t = run(md, data_dir, f"{a.out}/ckpt", a.stop_hours * 3600, info, attempt)
            info["framework"] = f"aligntune (TRL backend) DPO, attempt {attempt}"
            ok = True
            break
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]
            info["attempts"].append({"attempt": attempt, "error": err})
            S.log("AlignTune DPO attempt", attempt, "failed:", err[-1500:])
            import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if ok:
        try:
            info["final_eval"] = t.trainer.evaluate()
        except Exception as e:
            info["final_eval_error"] = str(e)[:500]
        adir = f"{a.out}/adapter"
        t.trainer.model.save_pretrained(adir)
        for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
            if os.path.exists(f"{md}/{f}"):
                shutil.copy(f"{md}/{f}", adir)
        info["result"] = res if isinstance(res, dict) else str(res)
        info["log_history"] = hist
    info["nan_events"] = S.NAN_EVENTS
    info["ok"] = ok
    info["train_seconds"] = time.time() - S.T0
    json.dump(info, open(f"{a.out}/train_info.json", "w"), indent=2, default=str)
    shutil.rmtree(f"{a.out}/ckpt", ignore_errors=True)
    shutil.rmtree("/tmp/tinyaya_dpo", ignore_errors=True)
    S.log("DPO", "OK" if ok else "FAILED (ship SFT-only)")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
