"""Run three, DPO retry (low-memory): LoRA DPO with AlignTune (TRL backend) on the merged v3-SFT model.

Why a second script: the first run-three DPO (scripts/train_dpo.py) OOM'd twice on a 16 GB T4 (TRL's entropy
metric reshaped the full [4, 1024, 262144] logits; then the in-step reference forward). Changes here, all memory:
- per-device batch 1, grad accumulation 16 (same 16 pairs per optimiser step as before: 2 x 8)
- gradient checkpointing (non-reentrant), enforced on the final DPOConfig
- precompute_ref_log_probs=True: reference log-probs (v3-SFT model = PEFT adapter disabled) are computed once
  before training, so no reference forward runs inside a training step and no second model copy is ever loaded
- max_length 768, truncation keep_start. 512 was requested, but the stock Tiny Aya chat template adds a ~370-token
  system preamble, so prompts are 376-440 tokens: at 512, 453/763 pairs become byte-identical after truncation
  (the referral sentence the pairs differ on is cut), which would make DPO a no-op. 768 truncates no chosen answer
  and leaves 0 identical pairs. TRL 1.7.1 has no max_prompt_length, and a 256-token prompt cap would cut into the
  system preamble the model was SFT'd and is evaluated with, so prompts are not truncated (recorded, not hidden).
- fp16 base weights, LoRA r16 params kept fp32 (same fix as SFT)
- model spread over both T4s with a small max_memory on GPU 0 (embeddings + tied LM head and the logits live there)
- TRL's entropy metric is computed in row/seq chunks instead of reshaping the whole logits tensor (metrics-only patch;
  same value)
- TRL 1.7.1's _precompute_ref_logps is replaced by an in-memory version (the original round-trips through a datasets
  cache file that does not exist for AlignTune's in-memory datasets; found in the CPU smoke test)
Unchanged: beta 0.1, lr 5e-5, 1 epoch, sigmoid loss, cosine schedule, warmup 0.1, clip 1.0, same LoRA shape.

Usage: python train_dpo_lowmem.py --out DIR --model_dir /tmp/m3_v3sft [--train_file dpo_train.jsonl --dev_file dpo_dev.jsonl]
CPU smoke test: SFT_TEST=1 python train_dpo_lowmem.py --out /tmp/x --model_dir /tmp/tinycohere
"""
import argparse, json, os, shutil, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _p in ["/kaggle/input"]:  # train_sft.py lives in the run-three dataset on Kaggle
    import glob as _g
    _h = _g.glob(f"{_p}/**/train_sft.py", recursive=True)
    if _h:
        sys.path.insert(0, os.path.dirname(_h[0]))
import train_sft as S  # prep_model (chat template w/ generation tags), find, log, time_cb, fp32_trainable, NAN_EVENTS

TEST = os.environ.get("SFT_TEST") == "1"
HP = dict(epochs=1, lr=5e-5, beta=0.1, loss="sigmoid", max_grad_norm=1.0, r=16, alpha=32, dropout=0.05,
          max_len=int(os.environ.get("DPO_MAX_LEN", "768")), max_prompt_len=None, truncation="keep_start",
          bs=1, grad_acc=16, precompute_ref_log_probs=True, gradient_checkpointing=True,
          max_memory={0: "3500MiB", 1: "13GiB"},
          targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
if TEST:
    HP.update(grad_acc=2, max_len=int(os.environ.get("DPO_MAX_LEN", "600")))


def prep(model_dir, train_file, dev_file, max_len, dst="/tmp/dpodata"):
    """Copy pairs; drop pairs that are identical after max_len truncation (no DPO signal). Returns stats."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    st = {}
    for fn, name in [(train_file, "train"), (dev_file, "val")]:
        os.makedirs(f"{dst}/{name}", exist_ok=True)
        s = {"in": 0, "kept": 0, "dropped_identical_after_trunc": 0, "chosen_truncated": 0, "rejected_truncated": 0,
             "prompt_tokens_max": 0}
        with open(f"{dst}/{name}/dpo.jsonl", "w", encoding="utf-8") as f:
            for l in open(S.find(fn), encoding="utf-8"):
                r = json.loads(l)
                s["in"] += 1
                p = tok.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
                pc = tok.apply_chat_template(r["prompt"] + r["chosen"], tokenize=True, return_dict=True)["input_ids"]
                pr = tok.apply_chat_template(r["prompt"] + r["rejected"], tokenize=True, return_dict=True)["input_ids"]
                s["prompt_tokens_max"] = max(s["prompt_tokens_max"], len(p))
                s["chosen_truncated"] += len(pc) > max_len
                s["rejected_truncated"] += len(pr) > max_len
                if pc[:max_len] == pr[:max_len]:
                    s["dropped_identical_after_trunc"] += 1
                    continue
                f.write(json.dumps({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}, ensure_ascii=False) + "\n")
                s["kept"] += 1
                if TEST and s["kept"] >= 8:
                    break
        st[name] = s
    S.log("dpo data", st)
    return dst, st


def chunked_entropy(logits, chunk_size=128):
    import torch, torch.nn.functional as F
    out = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0); out = out.unsqueeze(0); squeeze = True
    else:
        squeeze = False
    for b in range(logits.shape[0]):
        for s in range(0, logits.shape[1], chunk_size):
            lp = F.log_softmax(logits[b, s:s + chunk_size].float(), dim=-1)
            out[b, s:s + chunk_size] = -(lp.exp() * lp).sum(-1)
    return out.squeeze(0) if squeeze else out


def forced_args(max_len, precompute):
    return dict(bf16=False, fp16=not TEST, per_device_train_batch_size=HP["bs"], per_device_eval_batch_size=1,
                gradient_accumulation_steps=HP["grad_acc"], gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=max_len,
                truncation_mode=HP["truncation"], precompute_ref_log_probs=precompute, precompute_ref_batch_size=1,
                max_grad_norm=HP["max_grad_norm"], report_to="none")


def precompute_ref_logps_inmemory(self, dataset, name, batch_size):
    """Replacement for TRL 1.7.1 DPOTrainer._precompute_ref_logps. The original writes the reference log-probs to a
    datasets cache file and re-opens it with Dataset.from_file; AlignTune hands DPOTrainer in-memory datasets, whose
    map() writes no cache file, so the original fails with 'Failed to open local file .../cache-*.arrow' (reproduced
    in the CPU smoke test). Same computation (TRL's own compute_ref_log_probs, adapter disabled), columns added in memory."""
    import torch
    from torch.utils.data import DataLoader
    dl = self.accelerator.prepare(DataLoader(dataset, batch_size=batch_size, collate_fn=self.data_collator, shuffle=False))
    ch, rj = [], []
    t0 = time.time()
    for i, b in enumerate(dl):
        c, r = self.compute_ref_log_probs(b)
        ch.append(c.float().cpu()); rj.append(r.float().cpu())
        if i % 100 == 0:
            S.log(f"ref logps {name} {i}/{len(dl)} {time.time() - t0:.0f}s")
    ch, rj = torch.cat(ch).tolist(), torch.cat(rj).tolist()
    S.log(f"ref logps {name} done n={len(ch)} in {time.time() - t0:.0f}s")
    return dataset.add_column("ref_chosen_logps", ch).add_column("ref_rejected_logps", rj)


def patch_trl(max_len, precompute):
    """AlignTune builds trl.DPOConfig without fp16/bf16 (TRL then defaults to bf16, which a T4 rejects) and with
    eval batch 4. Force the memory settings in DPOConfig's constructor; AlignTune imports trl.DPOConfig at call time."""
    import trl, trl.trainer.dpo_trainer as DT
    DT.entropy_from_logits = chunked_entropy
    DT.DPOTrainer._precompute_ref_logps = precompute_ref_logps_inmemory
    Orig = getattr(trl.DPOConfig, "_lowmem_orig", trl.DPOConfig)
    fa = forced_args(max_len, precompute)

    class DPOConfigLowMem(Orig):
        _lowmem_orig = Orig

        def __init__(self, *a, **k):
            k.update(fa)
            if TEST:
                k["use_cpu"] = True
            super().__init__(*a, **k)
    trl.DPOConfig = DPOConfigLowMem


def mem_cb(info):
    from transformers import TrainerCallback
    import torch

    class M(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            keys = ["per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
                    "gradient_checkpointing", "gradient_checkpointing_kwargs", "max_length", "truncation_mode",
                    "precompute_ref_log_probs", "fp16", "bf16", "beta", "loss_type", "learning_rate",
                    "num_train_epochs", "max_steps", "lr_scheduler_type", "warmup_steps", "max_grad_norm"]
            info["final_dpo_args"] = {k: getattr(args, k, None) for k in keys}
            S.log("final DPOConfig", info["final_dpo_args"])
            m = kw.get("model")
            dm = getattr(m, "hf_device_map", None)
            if dm:
                info["device_map_counts"] = {str(d): sum(1 for v in dm.values() if v == d) for d in set(dm.values())}
                S.log("device map counts", info["device_map_counts"])

        def on_log(self, args, state, control, logs=None, **kw):
            if torch.cuda.is_available():
                info["peak_mem_gib"] = {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                                        for i in range(torch.cuda.device_count())}
                S.log("step", state.global_step, "peak GiB", info["peak_mem_gib"], logs)
    return M()


def run(model_dir, data_dir, out, stop_s, info, max_len, precompute):
    import torch
    from aligntune.core.backend_factory import create_rl_trainer
    patch_trl(max_len, precompute)
    kw = dict(model_name=model_dir, dataset_name=data_dir, algorithm="dpo", backend="trl", output_dir=out,
              num_epochs=HP["epochs"], batch_size=HP["bs"], per_device_eval_batch_size=1,
              gradient_accumulation_steps=HP["grad_acc"], learning_rate=HP["lr"], max_seq_length=max_len,
              max_length=max_len, truncation_mode=HP["truncation"], beta=HP["beta"], loss_type=HP["loss"],
              warmup_ratio=0.1, lr_scheduler="cosine", use_peft=True, lora_r=HP["r"], lora_alpha=HP["alpha"],
              lora_dropout=HP["dropout"], lora_target_modules=HP["targets"],
              precision="fp32" if TEST else "fp16", fp16=not TEST, max_grad_norm=HP["max_grad_norm"],
              use_gradient_checkpointing=True, gradient_checkpointing=True,
              gradient_checkpointing_kwargs={"use_reentrant": False},
              precompute_ref_log_probs=precompute, precompute_ref_batch_size=1,
              eval_strategy="steps", eval_interval=20, save_interval=10**6, save_strategy="no", logging_steps=2,
              loggers=[], report_to="none", seed=42, reward_model_name=None,
              **({"max_steps": 2, "device_map": "cpu"} if TEST else {"device_map": "auto"}))
    t = create_rl_trainer(**kw)
    if not TEST and torch.cuda.device_count() > 1:
        t.config.model.max_memory = HP["max_memory"]
    orig_setup = t.setup_model

    def setup_model_fp32():
        orig_setup()
        info["param_dtypes"] = S.fp32_trainable(t.model)
        if torch.cuda.is_available():
            info["mem_after_load_gib"] = {i: round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}
            S.log("memory after model load", info["mem_after_load_gib"])
    t.setup_model = setup_model_fp32
    orig_cb = t.get_hf_callbacks
    t.get_hf_callbacks = lambda: (orig_cb() or []) + [S.time_cb(stop_s), mem_cb(info)]
    res = t.train()
    return res, t.trainer.state.log_history, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--train_file", default="dpo_train.jsonl")
    ap.add_argument("--dev_file", default="dpo_dev.jsonl")
    ap.add_argument("--stop_hours", type=float, default=2.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = S.prep_model(a.model_dir, dst="/tmp/tinyaya_dpo")
    info = {"hp": HP, "model_dir": a.model_dir, "attempts": [], "script": "train_dpo_lowmem.py"}
    ok = False
    plan = [(HP["max_len"], HP["precompute_ref_log_probs"])]
    for attempt in (1, 2):
        max_len, precompute = plan[-1]
        data_dir, dstats = prep(md, a.train_file, a.dev_file, max_len)
        info["data"] = dstats
        try:
            res, hist, t = run(md, data_dir, f"{a.out}/ckpt", a.stop_hours * 3600, info, max_len, precompute)
            info["framework"] = f"aligntune (TRL backend) DPO, low-mem attempt {attempt}"
            info["used"] = {"max_len": max_len, "precompute_ref_log_probs": precompute}
            ok = True
            break
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]
            info["attempts"].append({"attempt": attempt, "max_len": max_len, "precompute": precompute, "error": err})
            S.log("AlignTune DPO attempt", attempt, "failed:", err[-1500:])
            oom = "OutOfMemory" in err or "out of memory" in err
            plan.append((640, precompute) if oom else (max_len, not precompute))
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
    sys.stdout.flush()
    os._exit(0 if ok else 3)  # skip interpreter teardown (run three segfaulted at exit)


if __name__ == "__main__":
    main()
