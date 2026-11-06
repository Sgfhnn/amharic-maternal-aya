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

# ---- Kaggle kernel (run four, attempt b): same as dpo4_main.py (rebuild the merged v3 SFT model, low-memory AlignTune
# DPO with unchanged settings, eval) on the v4b pairs (data/curated_v4b: v4 pairs + boilerplate-free fluency-vs-MT pairs +
# anti-repetition pairs; private dataset neimashube/amh-tinyaya-r4bdpo), then greedy-generate the 82-question held-out
# urgent probe with the v4b adapter. Output results/dpo_v4b and probe/dpo_v4b.jsonl. Free T4 x2; no token in the kernel.
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
json.dump({"stage": "dpo_v4b", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
DOUT = f"{W}/results/dpo_v4b"; os.makedirs(DOUT, exist_ok=True)
rc, _ = sh(f"cd /tmp && python -u {DPO_SCRIPT} --out {DOUT} --model_dir {M3} --train_file dpo4b_train.jsonl "
           f"--dev_file dpo4b_dev.jsonl --stop_hours 2.5 > {DOUT}/train.log 2>&1", 4 * 3600)
sh(f"grep -E 'dpo data|final DPOConfig|device map|memory after|ref logps .* done|peak GiB|attempt|DPO ' {DOUT}/train.log | tail -n 40; tail -n 5 {DOUT}/train.log")
if os.path.exists(f"{DOUT}/adapter/adapter_config.json"):
    json.dump({"stage": "eval_dpo_v4b", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
    eval_parallel(f"{DOUT}/eval", extra=f"--model {M3} --adapter {DOUT}/adapter")
else:
    log("DPO produced no adapter")
if os.path.exists(f"{DOUT}/adapter/adapter_config.json"):
    PG = glob.glob("/kaggle/input/**/probe_gen.py", recursive=True)[0]
    os.makedirs(f"{W}/probe", exist_ok=True)
    sh(f"CUDA_VISIBLE_DEVICES=0 python -u {PG} --questions probe_urgent_v4.jsonl --model {M3} --adapter {DOUT}/adapter "
       f"--out {W}/probe/dpo_v4b.jsonl > {W}/probe/dpo_v4b.log 2>&1", 3600)
json.dump({"stage": "done", "t": time.time() - T0}, open(f"{W}/stage.json", "w"))
log("kernel done r4b dpo total s", round(time.time() - T0))
