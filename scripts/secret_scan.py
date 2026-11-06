"""Scan a directory tree for secrets before anything is pushed. Prints file:line and the pattern NAME only, never the match.
Usage: python secret_scan.py DIR [DIR ...]   (exit 1 if anything is found)
Also checks for the literal values of HF_TOKEN / KAGGLE_API_TOKEN / GITHUB_TOKEN from the environment, if set."""
import os, re, sys

PAT = {
    "hf_token": r"\bhf_[A-Za-z0-9]{30,}",
    "github_token": r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}",
    "kaggle_token": r"\bKGAT_[A-Za-z0-9]{20,}",
    "openai_like_key": r"\bsk-[A-Za-z0-9_\-]{24,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_\-]{30,}",
    "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "bearer_literal": r"Bearer\s+(?!x\b|\"\s*\+|\$|\{|<)[A-Za-z0-9_\-\.]{20,}",
    "jwt": r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}",
    "kaggle_json_key": r"\"key\"\s*:\s*\"[0-9a-f]{32}\"",
    "password_assign": r"(?i)\b(password|passwd|secret)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]",
}
RX = {k: re.compile(v) for k, v in PAT.items()}
LIT = [v for k in ("HF_TOKEN", "KAGGLE_API_TOKEN", "GITHUB_TOKEN") if (v := os.environ.get(k)) and len(v) >= 12]
SKIP_EXT = {".safetensors", ".bin", ".pt", ".parquet", ".xlsx", ".png", ".jpg"}
hits = 0
for root in sys.argv[1:]:
    for dp, dns, fs in os.walk(root):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
        for f in fs:
            p = os.path.join(dp, f)
            if os.path.splitext(f)[1] in SKIP_EXT or os.path.getsize(p) > 50_000_000:
                continue
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                for k, rx in RX.items():
                    if rx.search(line):
                        print(f"{p}:{i}: {k}"); hits += 1
                if any(v in line for v in LIT):
                    print(f"{p}:{i}: env_token_literal"); hits += 1
print("secret scan:", "CLEAN" if not hits else f"{hits} hit(s)")
sys.exit(1 if hits else 0)
