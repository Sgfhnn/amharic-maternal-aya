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
