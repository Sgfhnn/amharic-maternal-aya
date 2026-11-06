# ---- Kaggle kernel (run four, DPO v4 over-referral fix): same as dpo3b_main.py (rebuild the exact merged v3 SFT model, then
# low-memory AlignTune DPO via train_dpo_lowmem.py, unchanged settings) but on the CuratorKIT v4 pairs
# data/curated_v4/dpo (dpo4_train.jsonl / dpo4_dev.jsonl, private dataset neimashube/amh-tinyaya-r4dpo). Output results/dpo_v4.
# Free T4 x2; no token in the kernel.
install_base()
sh('pip install -q "trl==1.7.1" "datasets>=4.7" curatorkit sqlglot heapdict msgspec tyro pypdf python-docx rich typer', 1800)
sh("pip install -q --no-deps aligntune==0.1.30")
sh("pip uninstall -y -q torchao")  # Kaggle torchao 0.10 breaks PEFT LoRA dispatch
import re, hashlib
for i in range(12):
    rc, out = sh('python -c "from aligntune.core.backend_factory import create_sft_trainer, create_rl_trainer"')
    if rc == 0: log("aligntune import OK"); break
    m = re.findall(r"No module named '([^'.]+)", out)
    if not m: log("aligntune import failed (not a missing module)"); break
    pkg = {"docx": "python-docx", "yaml": "pyyaml", "sklearn": "scikit-learn"}.get(m[-1], m[-1])
    sh(f"pip install -q {pkg}")
sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|peft|accelerate|trl|datasets|aligntune|curatorkit) '")
DPO_SCRIPT = glob.glob("/kaggle/input/**/train_dpo_lowmem.py", recursive=True)[0]
log("dpo script", DPO_SCRIPT)
MD = "/tmp/tinyaya_official"
rc, _ = sh(f"python -u {SRC}/assemble_model.py {MD} > {W}/assemble.log 2>&1", 3600)
sh(f"tail -n 8 {W}/assemble.log")
if rc != 0:
    raise SystemExit("model assembly failed")
def unpack(prefix, dst):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"/kaggle/input/**/{prefix}__*", recursive=True):
        shutil.copy(f, f"{dst}/" + os.path.basename(f).split("__", 1)[1])
    return dst
R2 = unpack("r2adapter", "/tmp/r2_adapter")
V3 = unpack("sftv3adapter", "/tmp/sftv3_adapter")
h = hashlib.sha256(open(f"{V3}/adapter_model.safetensors", "rb").read()).hexdigest()
log("v3 SFT adapter sha256", h)
M2, M3 = "/tmp/m2_r2curated", "/tmp/m3_v3sft"
json.dump({"stage": "merge", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
rc, _ = sh(f"python -u {SRC}/merge_adapter.py {MD} {R2} {M2} > {W}/merge_m2.log 2>&1", 3600)
sh(f"tail -n 3 {W}/merge_m2.log")
if rc != 0:
    raise SystemExit("merge M2 failed")
rc, _ = sh(f"python -u {SRC}/merge_adapter.py {M2} {V3} {M3} > {W}/merge_m3.log 2>&1", 3600)
sh(f"tail -n 3 {W}/merge_m3.log")
if rc != 0:
    raise SystemExit("merge M3 failed")
shutil.rmtree(MD, ignore_errors=True); shutil.rmtree(M2, ignore_errors=True)  # free disk; merges ran in their own processes
sh("df -h /tmp | tail -1; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv")
json.dump({"stage": "dpo_v4", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
DOUT = f"{W}/results/dpo_v4"; os.makedirs(DOUT, exist_ok=True)
rc, _ = sh(f"cd /tmp && python -u {DPO_SCRIPT} --out {DOUT} --model_dir {M3} --train_file dpo4_train.jsonl "
           f"--dev_file dpo4_dev.jsonl --stop_hours 2.5 > {DOUT}/train.log 2>&1", 4 * 3600)
sh(f"grep -E 'dpo data|final DPOConfig|device map|memory after|ref logps .* done|peak GiB|attempt|DPO ' {DOUT}/train.log | tail -n 40; tail -n 5 {DOUT}/train.log")
if os.path.exists(f"{DOUT}/adapter/adapter_config.json"):
    json.dump({"stage": "eval_dpo_v4", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
    eval_parallel(f"{DOUT}/eval", extra=f"--model {M3} --adapter {DOUT}/adapter")
else:
    log("DPO produced no adapter")
json.dump({"stage": "done", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
log("kernel done r4 dpo total s", round(time.time() - T0))
