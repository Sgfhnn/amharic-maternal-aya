# ---- Kaggle kernel: AlignTune LoRA SFT + "after" eval ----
install_base()
sh('pip install -q "trl==1.7.1" "datasets>=4.7" curatorkit sqlglot heapdict msgspec tyro pypdf python-docx rich typer', 1800)
sh("pip install -q --no-deps aligntune==0.1.30")
# Kaggle image ships torchao 0.10, which makes PEFT's LoRA dispatch raise ImportError (needs >=0.16): remove it.
sh("pip uninstall -y -q torchao")
for i in range(12):
    rc, out = sh('python -c "from aligntune.core.backend_factory import create_sft_trainer"')
    if rc == 0: log("aligntune import OK"); break
    import re
    m = re.findall(r"No module named '([^'.]+)", out)
    if not m: log("aligntune import failed (not a missing module)"); break
    pkg = {"docx": "python-docx", "yaml": "pyyaml", "sklearn": "scikit-learn"}.get(m[-1], m[-1])
    sh(f"pip install -q {pkg}")
sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|peft|accelerate|trl|datasets|aligntune|curatorkit) '")
os.makedirs(f"{W}/results/sft", exist_ok=True)
rc, _ = sh(f"cd /tmp && python -u {SRC}/train_sft.py --out {W}/results/sft --stop_hours {STOP_HOURS} > {W}/results/sft/train.log 2>&1", 11 * 3600)
sh(f"tail -n 30 {W}/results/sft/train.log")
if os.path.exists(f"{W}/results/sft/adapter/adapter_config.json"):
    eval_parallel(f"{W}/results/sft/eval", extra=f"--adapter {W}/results/sft/adapter")
else:
    log("no adapter; skipping after-eval")
log("sft kernel done total s", round(time.time() - T0))
