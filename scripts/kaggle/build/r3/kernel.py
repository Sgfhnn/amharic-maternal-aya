# Shared kernel driver code (prepended to each Kaggle kernel script by build_kernels.py).
import os, sys, json, time, glob, subprocess, shutil
T0 = time.time()
W = "/kaggle/working"
os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/driver_log.txt", "a")
def log(*a):
    s = f"[{time.strftime('%H:%M:%S')}] [{time.time()-T0:6.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def sh(cmd, timeout=3600, env=None):
    t = time.time(); log("$", cmd[:300])
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    log(f"  -> rc={p.returncode} {time.time()-t:.0f}s")
    tail = (p.stdout[-1500:] + "\n" + p.stderr[-2500:]).strip()
    if p.returncode: log(tail)
    return p.returncode, p.stdout + p.stderr
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python -V; df -h /tmp | tail -1; ls -R /kaggle/input | head -40")
SRC = os.path.dirname(glob.glob("/kaggle/input/**/eval_lib.py", recursive=True)[0])
log("scripts dir", SRC)

def install_base():
    sh('pip install -q "transformers==5.15.0" "peft>=0.18" "accelerate>=1.6" sentencepiece', 1800)
    sh("pip list 2>/dev/null | grep -i -E '^(torch|transformers|peft|accelerate|trl|datasets|aligntune|curatorkit) '")

def eval_parallel(out, extra=""):
    """gen on GPU0 and MCQ on GPU1 at the same time; sequential fp32 multi-GPU fallback."""
    os.makedirs(out, exist_ok=True)
    procs = {}
    for task, gpu in [("gen", "0"), ("mcq", "1")]:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        f = open(f"{out}/{task}.log", "w")
        procs[task] = (subprocess.Popen(f"python -u {SRC}/eval_lib.py --task {task} --out {out} {extra}",
                                        shell=True, stdout=f, stderr=subprocess.STDOUT, env=env), f)
    rc = {}
    for task, (p, f) in procs.items():
        rc[task] = p.wait(); f.close()
        log(task, "rc", rc[task]); sh(f"tail -n 5 {out}/{task}.log")
    for task, r in rc.items():
        if r != 0:
            log("retrying", task, "on both GPUs (fp16 then fp32)")
            sh(f"python -u {SRC}/eval_lib.py --task {task} --out {out} --multi_gpu {extra} > {out}/{task}_retry.log 2>&1", 4 * 3600)
    return rc

STOP_HOURS = 3.0
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
