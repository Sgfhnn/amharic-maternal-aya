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
