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

VARIANT = 'sanity'
TRAIN_FILE = 'chat_train.jsonl'
DEV_FILE = 'chat_dev.jsonl'
STOP_HOURS = 1.0
EXTRA_TRAIN_ARGS = '--max_steps 50 --eval_steps 25 --dev_limit 64'
BASELINE_EVAL = False
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
