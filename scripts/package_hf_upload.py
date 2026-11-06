"""Packaging: create the PRIVATE HF repos and upload the staged folders (work/hf_stage/*).
Token comes from the sandbox (HF_TOKEN, injected); it is never printed. Every repo is created with private=True and
re-checked as private after upload; the script aborts if any repo reads back as public.
Usage: python package_hf_upload.py ds_sft ds_dpo ds_eval   (large folders: upload per subfolder, see README "Packaging")
"""
import os, sys
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = f"{ROOT}/work/hf_stage"
OWNER = "5LeaderW"
REPOS = {
    "model": (f"{OWNER}/tiny-aya-earth-amharic-maternal-health", "model"),
    "adapters": (f"{OWNER}/tiny-aya-earth-amharic-maternal-health-adapters", "model"),
    "ds_sft": (f"{OWNER}/amharic-maternal-health-sft-curated-v3", "dataset"),
    "ds_dpo": (f"{OWNER}/amharic-maternal-health-dpo-pairs", "dataset"),
    "ds_eval": (f"{OWNER}/amharic-maternal-health-aya-eval-logs", "dataset"),
}
api = HfApi()
assert api.whoami()["name"] == OWNER
for key in sys.argv[1:]:
    rid, rtype = REPOS[key]
    url = api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
    info = api.repo_info(rid, repo_type=rtype)
    if not info.private:
        raise SystemExit(f"{rid} is PUBLIC - aborting before upload")
    api.upload_folder(repo_id=rid, repo_type=rtype, folder_path=f"{S}/{key}",
                      commit_message="Initial private upload (Lexsi hackathon packaging)",
                      ignore_patterns=[".cache/**", "**/.cache/**"])
    info = api.repo_info(rid, repo_type=rtype, files_metadata=False)
    assert info.private, f"{rid} became public?"
    print(f"OK {rtype} {rid} private={info.private} files={len(info.siblings)} url={url}")
