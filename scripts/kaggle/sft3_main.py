# ---- Kaggle kernel (run three): official Tiny Aya Earth -> merge run-two curated adapter -> AlignTune SFT on
# curated_v3 -> eval -> merge -> AlignTune DPO -> eval. Everything stays on free T4 x2; no token in the kernel.
install_base()
sh('pip install -q "trl==1.7.1" "datasets>=4.7" curatorkit sqlglot heapdict msgspec tyro pypdf python-docx rich typer', 1800)
sh("pip install -q --no-deps aligntune==0.1.30")
sh("pip uninstall -y -q torchao")  # Kaggle torchao 0.10 breaks PEFT LoRA dispatch
import re
for i in range(12):
    rc, out = sh('python -c "from aligntune.core.backend_factory import create_sft_trainer, create_rl_trainer"')
    if rc == 0: log("aligntune import OK"); break
    m = re.findall(r"No module named '([^'.]+)", out)
    if not m: log("aligntune import failed (not a missing module)"); break
    pkg = {"docx": "python-docx", "yaml": "pyyaml", "sklearn": "scikit-learn"}.get(m[-1], m[-1])
    sh(f"pip install -q {pkg}")
sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|peft|accelerate|trl|datasets|aligntune|curatorkit) '")
MD = "/tmp/tinyaya_official"
rc, _ = sh(f"python -u {SRC}/assemble_model.py {MD} > {W}/assemble.log 2>&1", 3600)
sh(f"tail -n 8 {W}/assemble.log")
if rc != 0:
    raise SystemExit("model assembly failed")
# run-two curated adapter -> merged starting point M2
R2 = "/tmp/r2_adapter"; os.makedirs(R2, exist_ok=True)
for f in glob.glob(f"{SRC}/r2adapter__*"):
    shutil.copy(f, f"{R2}/" + os.path.basename(f).split("__", 1)[1])
M2, M3 = "/tmp/m2_r2curated", "/tmp/m3_v3sft"
rc, _ = sh(f"python -u {SRC}/merge_adapter.py {MD} {R2} {M2} > {W}/merge_m2.log 2>&1", 3600)
sh(f"tail -n 3 {W}/merge_m2.log; df -h /tmp | tail -1")
if rc != 0:
    raise SystemExit("merge M2 failed")
json.dump({"stage": "sft_v3", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
OUT = f"{W}/results/sft_v3"; os.makedirs(OUT, exist_ok=True)
rc, _ = sh(f"cd /tmp && python -u {SRC}/train_sft.py --out {OUT} --model_id CohereLabs/tiny-aya-earth+r2curated --model_dir {M2} "
           f"--train_file chat_train_v3.jsonl --dev_file chat_dev_v3.jsonl --stop_hours {STOP_HOURS} > {OUT}/train.log 2>&1", 6 * 3600)
sh(f"grep -E 'SANITY|NON-FINITE|data (train|val)|param dtypes|AlignTune failed' {OUT}/train.log | tail -n 20; tail -n 5 {OUT}/train.log")
shutil.rmtree("/tmp/tinyaya_train", ignore_errors=True)
if not os.path.exists(f"{OUT}/adapter/adapter_config.json"):
    raise SystemExit("v3 SFT produced no adapter")
json.dump({"stage": "eval_sft_v3", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
eval_parallel(f"{OUT}/eval", extra=f"--model {M2} --adapter {OUT}/adapter")
rc, _ = sh(f"python -u {SRC}/merge_adapter.py {M2} {OUT}/adapter {M3} > {W}/merge_m3.log 2>&1", 3600)
sh(f"tail -n 3 {W}/merge_m3.log; df -h /tmp | tail -1")
shutil.rmtree(M2, ignore_errors=True)
if rc == 0:
    json.dump({"stage": "dpo_v3", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
    DOUT = f"{W}/results/dpo_v3"; os.makedirs(DOUT, exist_ok=True)
    rc, _ = sh(f"cd /tmp && python -u {SRC}/train_dpo.py --out {DOUT} --model_dir {M3} --train_file dpo_train.jsonl "
               f"--dev_file dpo_dev.jsonl --stop_hours 2.0 > {DOUT}/train.log 2>&1", 4 * 3600)
    sh(f"tail -n 30 {DOUT}/train.log")
    if os.path.exists(f"{DOUT}/adapter/adapter_config.json"):
        json.dump({"stage": "eval_dpo_v3", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
        eval_parallel(f"{DOUT}/eval", extra=f"--model {M3} --adapter {DOUT}/adapter")
    else:
        log("DPO produced no adapter; shipping SFT-only v3")
json.dump({"stage": "done", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
log("kernel done run3 total s", round(time.time() - T0))
