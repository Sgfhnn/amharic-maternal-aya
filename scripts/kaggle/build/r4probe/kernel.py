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

# ---- Kaggle kernel (run four, held-out urgent probe): rebuild the exact merged v3 SFT model (official -> run-2 B
# adapter -> v3 SFT adapter, as in dpo3b_main.py), then greedy-generate the 82 held-out urgent probe questions with
# v3 SFT (no adapter), v3 DPO (adapter from kernel output neimashube/amh-tinyaya-r3-dpo) and v4 DPO (kernel output
# neimashube/amh-tinyaya-r4-dpo). Free T4 x2; no token in the kernel.
install_base()
sh("pip uninstall -y -q torchao")
import hashlib
MD = "/tmp/tinyaya_official"
rc, _ = sh(f"python -u {SRC}/assemble_model.py {MD} > {W}/assemble.log 2>&1", 3600)
if rc != 0:
    raise SystemExit("model assembly failed")
def unpack(prefix, dst):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"/kaggle/input/**/{prefix}__*", recursive=True):
        shutil.copy(f, f"{dst}/" + os.path.basename(f).split("__", 1)[1])
    return dst
R2 = unpack("r2adapter", "/tmp/r2_adapter"); V3 = unpack("sftv3adapter", "/tmp/sftv3_adapter")
log("v3 SFT adapter sha256", hashlib.sha256(open(f"{V3}/adapter_model.safetensors", "rb").read()).hexdigest())
M2, M3 = "/tmp/m2_r2curated", "/tmp/m3_v3sft"
for src, ad, dst in [(MD, R2, M2), (M2, V3, M3)]:
    rc, _ = sh(f"python -u {SRC}/merge_adapter.py {src} {ad} {dst} > {W}/merge_{os.path.basename(dst)}.log 2>&1", 3600)
    if rc != 0:
        raise SystemExit("merge failed " + dst)
shutil.rmtree(MD, ignore_errors=True); shutil.rmtree(M2, ignore_errors=True)
PG = glob.glob("/kaggle/input/**/probe_gen.py", recursive=True)[0]
A = {k: os.path.dirname(glob.glob(f"/kaggle/input/**/{k}/adapter/adapter_config.json", recursive=True)[0]) for k in ["dpo_v3", "dpo_v4"]}
log("adapters", A)
os.makedirs(f"{W}/probe", exist_ok=True)
procs = []
for name, ad, gpu in [("dpo_v3", A["dpo_v3"], "0"), ("dpo_v4", A["dpo_v4"], "1")]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs.append(subprocess.Popen(f"python -u {PG} --questions probe_urgent_v4.jsonl --model {M3} --adapter {ad} "
                                  f"--out {W}/probe/{name}.jsonl > {W}/probe/{name}.log 2>&1", shell=True, env=env))
for p in procs:
    log("rc", p.wait())
sh(f"CUDA_VISIBLE_DEVICES=0 python -u {PG} --questions probe_urgent_v4.jsonl --model {M3} --out {W}/probe/sft_v3.jsonl > {W}/probe/sft_v3.log 2>&1", 3600)
sh(f"wc -l {W}/probe/*.jsonl; tail -n 2 {W}/probe/*.log")
log("kernel done r4 probe total s", round(time.time() - T0))
