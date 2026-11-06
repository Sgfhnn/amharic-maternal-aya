"""Assemble single-file Kaggle kernel scripts from common_driver.py + *_main.py."""
import os, json
D = os.path.dirname(os.path.abspath(__file__))
common = open(f"{D}/common_driver.py").read()
# run one kernels
for name, pre in [("baseline", ""), ("sft", "STOP_HOURS = float(os.environ.get('STOP_HOURS', '4.0'))\n")]:
    os.makedirs(f"{D}/build/{name}", exist_ok=True)
    open(f"{D}/build/{name}/kernel.py", "w").write(common + "\n" + pre + open(f"{D}/{name}_main.py").read())
    print("built", name)
# run two kernels: dataset sources
RUN2_DS = ["neimashube/amh-tinyaya-run2"]
V = {
    "sanity":  dict(slug="amh-tinyaya-r2-sanity", TRAIN_FILE="chat_train.jsonl", DEV_FILE="chat_dev.jsonl", STOP_HOURS=1.0,
                    EXTRA_TRAIN_ARGS="--max_steps 50 --eval_steps 25 --dev_limit 64", BASELINE_EVAL=False),
    "raw":     dict(slug="amh-tinyaya-r2-raw", TRAIN_FILE="chat_train.jsonl", DEV_FILE="chat_dev.jsonl", STOP_HOURS=4.0,
                    EXTRA_TRAIN_ARGS="", BASELINE_EVAL=False),
    "curated": dict(slug="amh-tinyaya-r2-curated", TRAIN_FILE="chat_train_curated.jsonl", DEV_FILE="chat_dev_curated.jsonl",
                    STOP_HOURS=4.0, EXTRA_TRAIN_ARGS="", BASELINE_EVAL=True),
}
for v, c in V.items():
    pre = (f"VARIANT = {v!r}\nTRAIN_FILE = {c['TRAIN_FILE']!r}\nDEV_FILE = {c['DEV_FILE']!r}\nSTOP_HOURS = {c['STOP_HOURS']}\n"
           f"EXTRA_TRAIN_ARGS = {c['EXTRA_TRAIN_ARGS']!r}\nBASELINE_EVAL = {c['BASELINE_EVAL']}\n")
    os.makedirs(f"{D}/build/r2_{v}", exist_ok=True)
    open(f"{D}/build/r2_{v}/kernel.py", "w").write(common + "\n" + pre + open(f"{D}/sft2_main.py").read())
    ds = RUN2_DS + (["neimashube/amh-tinyaya-curated-v2"] if v == "curated" else [])
    json.dump({"id": f"neimashube/{c['slug']}", "title": c["slug"], "dataset_sources": ds},
              open(f"{D}/build/r2_{v}/meta.json", "w"), indent=1)
    print("built r2", v)

# run three kernel
os.makedirs(f"{D}/build/r3", exist_ok=True)
open(f"{D}/build/r3/kernel.py", "w").write(common + "\nSTOP_HOURS = 3.0\n" + open(f"{D}/sft3_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-r3", "title": "amh-tinyaya-r3", "dataset_sources": ["neimashube/amh-tinyaya-run3"]},
          open(f"{D}/build/r3/meta.json", "w"), indent=1)
print("built r3")

# run three DPO retry kernel (low-memory AlignTune DPO from the merged v3 SFT model)
os.makedirs(f"{D}/build/r3dpo", exist_ok=True)
open(f"{D}/build/r3dpo/kernel.py", "w").write(common + "\n" + open(f"{D}/dpo3b_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-r3-dpo", "title": "amh-tinyaya-r3-dpo",
           "dataset_sources": ["neimashube/amh-tinyaya-run3", "neimashube/amh-tinyaya-r3dpo"]},
          open(f"{D}/build/r3dpo/meta.json", "w"), indent=1)
print("built r3dpo")

# run four DPO v4 kernel (over-referral fix): same low-memory AlignTune DPO from the merged v3 SFT model, v4 pairs
os.makedirs(f"{D}/build/r4dpo", exist_ok=True)
open(f"{D}/build/r4dpo/kernel.py", "w").write(common + "\n" + open(f"{D}/dpo4_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-r4-dpo", "title": "amh-tinyaya-r4-dpo",
           "dataset_sources": ["neimashube/amh-tinyaya-run3", "neimashube/amh-tinyaya-r3dpo", "neimashube/amh-tinyaya-r4dpo"]},
          open(f"{D}/build/r4dpo/meta.json", "w"), indent=1)
print("built r4dpo")

# run four held-out urgent probe kernel (needs the r3-dpo and r4-dpo kernel outputs as kernel sources)
os.makedirs(f"{D}/build/r4probe", exist_ok=True)
open(f"{D}/build/r4probe/kernel.py", "w").write(common + "\n" + open(f"{D}/probe4_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-r4-probe", "title": "amh-tinyaya-r4-probe",
           "dataset_sources": ["neimashube/amh-tinyaya-run3", "neimashube/amh-tinyaya-r3dpo", "neimashube/amh-tinyaya-r4probe"],
           "kernel_sources": ["neimashube/amh-tinyaya-r3-dpo", "neimashube/amh-tinyaya-r4-dpo"]},
          open(f"{D}/build/r4probe/meta.json", "w"), indent=1)
print("built r4probe")

# run four attempt b (v4b pairs) + probe generation in the same kernel
os.makedirs(f"{D}/build/r4bdpo", exist_ok=True)
open(f"{D}/build/r4bdpo/kernel.py", "w").write(common + "\n" + open(f"{D}/dpo4b_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-r4b-dpo", "title": "amh-tinyaya-r4b-dpo",
           "dataset_sources": ["neimashube/amh-tinyaya-run3", "neimashube/amh-tinyaya-r3dpo", "neimashube/amh-tinyaya-r4bdpo",
                               "neimashube/amh-tinyaya-r4probe"]},
          open(f"{D}/build/r4bdpo/meta.json", "w"), indent=1)
print("built r4bdpo")

# run four attempt c (v4c pairs: false-urgency counter-pairs) + probe generation in the same kernel
os.makedirs(f"{D}/build/r4cdpo", exist_ok=True)
open(f"{D}/build/r4cdpo/kernel.py", "w").write(common + "\n" + open(f"{D}/dpo4c_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-r4c-dpo", "title": "amh-tinyaya-r4c-dpo",
           "dataset_sources": ["neimashube/amh-tinyaya-run3", "neimashube/amh-tinyaya-r3dpo", "neimashube/amh-tinyaya-r4cdpo",
                               "neimashube/amh-tinyaya-r4probe"]},
          open(f"{D}/build/r4cdpo/meta.json", "w"), indent=1)
print("built r4cdpo")

# packaging: merge official + run-2 B + v3 SFT + v4c DPO into one fp16 model, sanity-check on 16 test rows
os.makedirs(f"{D}/build/merge4c", exist_ok=True)
open(f"{D}/build/merge4c/kernel.py", "w").write(common + "\n" + open(f"{D}/merge4c_main.py").read())
json.dump({"id": "neimashube/amh-tinyaya-merge-v4c", "title": "amh-tinyaya-merge-v4c",
           "dataset_sources": ["neimashube/amh-tinyaya-run3", "neimashube/amh-tinyaya-r3dpo", "neimashube/amh-tinyaya-merge4c"]},
          open(f"{D}/build/merge4c/meta.json", "w"), indent=1)
print("built merge4c")
