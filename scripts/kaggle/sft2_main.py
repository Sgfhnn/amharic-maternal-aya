# ---- Kaggle kernel (run two): assemble official Tiny Aya Earth, AlignTune LoRA SFT, after-eval ----
# Prepended constants: VARIANT ("sanity" | "raw" | "curated"), TRAIN_FILE, DEV_FILE, EXTRA_TRAIN_ARGS, STOP_HOURS, BASELINE_EVAL
install_base()
sh('pip install -q "trl==1.7.1" "datasets>=4.7" curatorkit sqlglot heapdict msgspec tyro pypdf python-docx rich typer', 1800)
sh("pip install -q --no-deps aligntune==0.1.30")
sh("pip uninstall -y -q torchao")  # Kaggle torchao 0.10 breaks PEFT LoRA dispatch
for i in range(12):
    rc, out = sh('python -c "from aligntune.core.backend_factory import create_sft_trainer"')
    if rc == 0: log("aligntune import OK"); break
    import re
    m = re.findall(r"No module named '([^'.]+)", out)
    if not m: log("aligntune import failed (not a missing module)"); break
    pkg = {"docx": "python-docx", "yaml": "pyyaml", "sklearn": "scikit-learn"}.get(m[-1], m[-1])
    sh(f"pip install -q {pkg}")
sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|peft|accelerate|trl|datasets|aligntune|curatorkit) '")
MD = "/tmp/tinyaya_official"
rc, _ = sh(f"python -u {SRC}/assemble_model.py {MD} > {W}/assemble.log 2>&1", 3600)
sh(f"cat {W}/assemble.log | tail -n 8")
if rc != 0:
    raise SystemExit("model assembly failed")
OUT = f"{W}/results/sft_{VARIANT}"
os.makedirs(OUT, exist_ok=True)
rc, _ = sh(f"cd /tmp && python -u {SRC}/train_sft.py --out {OUT} --model_id CohereLabs/tiny-aya-earth --model_dir {MD} "
           f"--train_file {TRAIN_FILE} --dev_file {DEV_FILE} --stop_hours {STOP_HOURS} {EXTRA_TRAIN_ARGS} > {OUT}/train.log 2>&1", 11 * 3600)
sh(f"grep -E 'SANITY|NON-FINITE|data (train|val)|param dtypes|AlignTune failed' {OUT}/train.log | tail -n 20; tail -n 5 {OUT}/train.log")
if VARIANT != "sanity" and os.path.exists(f"{OUT}/adapter/adapter_config.json"):
    eval_parallel(f"{OUT}/eval", extra=f"--model {MD} --adapter {OUT}/adapter")
if BASELINE_EVAL:
    import torch, gc; gc.collect()
    eval_parallel(f"{W}/results/baseline_official", extra=f"--model {MD}")
log("kernel done", VARIANT, "total s", round(time.time() - T0))
