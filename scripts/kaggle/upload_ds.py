"""Upload files as a PRIVATE Kaggle dataset (owner neimashube). Usage: python upload_ds.py SLUG FILE..."""
import os, sys, json, subprocess
import kag
slug, files = sys.argv[1], sys.argv[2:]
toks = []
for p in files:
    st = os.stat(p)
    r = kag.req("POST", "/blobs/upload", {"type": "dataset", "name": os.path.basename(p), "contentLength": st.st_size,
                                          "lastModifiedEpochSeconds": int(st.st_mtime)})
    assert "token" in r, r
    out = subprocess.run(["curl", "-sS", "-X", "PUT", "-H", "Content-Type: application/octet-stream", "--data-binary", "@" + p,
                          "-w", "%{http_code}", r["createUrl"], "-o", "/dev/null", "--max-time", "300"], capture_output=True, text=True)
    print(os.path.basename(p), out.stdout, out.stderr[:200], flush=True)
    assert out.stdout.startswith("20"), out
    toks.append({"token": r["token"]})
print(kag.req("POST", "/datasets/create/new", {"ownerSlug": "neimashube", "slug": slug, "title": slug, "licenseName": "unknown",
                                               "isPrivate": True, "convertToCsv": False, "files": toks}))
