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

# ---- Kaggle kernel (packaging): merge the full v4c adapter stack into ONE fp16 model and sanity-check it.
# official CohereLabs/tiny-aya-earth (assembled, SHA-256-checked) + run-2 B adapter -> M2, + v3 SFT adapter -> M3,
# + v4c DPO adapter -> M4 (= submission model). Sanity: greedy-generate the first 16 test rows (one batch, the same
# batch the v4c eval generated first) with (a) merged M4 and (b) M3 + v4c adapter exactly as the v4c eval ran it,
# and compare both with results/dpo_v4c/eval/predictions.jsonl offline. Output /kaggle/working/merged_v4c (fp16
# safetensors + stock tokenizer files) and /kaggle/working/merge_check/. Free T4 x2; no token in the kernel.
install_base()
sh("pip uninstall -y -q torchao")  # Kaggle torchao 0.10 breaks PEFT LoRA dispatch (seen in attempt 1)
import hashlib
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
V4C = unpack("dpo4cadapter", "/tmp/dpo4c_adapter")
shas = {n: hashlib.sha256(open(f"{d}/adapter_model.safetensors", "rb").read()).hexdigest() for n, d in
        [("r2_B", R2), ("v3_sft", V3), ("v4c_dpo", V4C)]}
log("adapter sha256", shas)
M2, M3, M4 = "/tmp/m2_r2curated", "/tmp/m3_v3sft", f"{W}/merged_v4c"
for base, ad, out in [(MD, R2, M2), (M2, V3, M3), (M3, V4C, M4)]:
    rc, _ = sh(f"python -u {SRC}/merge_adapter.py {base} {ad} {out} > {W}/merge_{os.path.basename(out)}.log 2>&1", 3600)
    sh(f"tail -n 3 {W}/merge_{os.path.basename(out)}.log")
    if rc != 0:
        raise SystemExit(f"merge {out} failed")
shutil.rmtree(MD, ignore_errors=True); shutil.rmtree(M2, ignore_errors=True)
sh(f"ls -la {M4}; du -sh {M4}; df -h /tmp {W} | tail -2")
C = f"{W}/merge_check"; os.makedirs(C, exist_ok=True)
json.dump({"adapter_sha256": shas, "stack": ["CohereLabs/tiny-aya-earth", "run-2 B", "v3 SFT", "v4c DPO"]},
          open(f"{C}/stack.json", "w"), indent=1)
procs = []
for name, gpu, extra in [("merged", "0", f"--model {M4}"), ("adapter", "1", f"--model {M3} --adapter {V4C}")]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    f = open(f"{C}/{name}.log", "w")
    procs.append((name, subprocess.Popen(f"python -u {SRC}/eval_lib.py --task gen --limit 16 --bs 16 --out {C}/{name} {extra}",
                                         shell=True, stdout=f, stderr=subprocess.STDOUT, env=env), f))
for name, p, f in procs:
    log(name, "rc", p.wait()); f.close(); sh(f"tail -n 4 {C}/{name}.log")
shutil.rmtree(M3, ignore_errors=True)
log("kernel done merge v4c total s", round(time.time() - T0))
