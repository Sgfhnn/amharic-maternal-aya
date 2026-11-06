"""Minimal Kaggle API helper (bearer token from env KAGGLE_API_TOKEN; never printed)."""
import os,sys,json,time,urllib.request,urllib.error
B="https://www.kaggle.com/api/v1"
H={"Authorization":"Bearer "+os.environ["KAGGLE_API_TOKEN"],"Content-Type":"application/json"}
def req(method,path,body=None,raw=False):
    r=urllib.request.Request(B+path,data=None if body is None else json.dumps(body).encode(),headers=H,method=method)
    try:
        with urllib.request.urlopen(r,timeout=300) as f: d=f.read(); return d if raw else json.loads(d or b"{}")
    except urllib.error.HTTPError as e: return {"_http":e.code,"_body":e.read()[:800].decode(errors="replace")}
def upload_blob(path):
    st=os.stat(path)
    r=req("POST","/blobs/upload",{"type":"dataset","name":os.path.basename(path),"contentLength":st.st_size,"lastModifiedEpochSeconds":int(st.st_mtime)})
    if "token" not in r: raise SystemExit(f"blob init failed {r}")
    with open(path,"rb") as f:
        u=urllib.request.Request(r["createUrl"],data=f.read(),method="PUT",headers={"Content-Type":"application/octet-stream"})
        urllib.request.urlopen(u,timeout=600).read()
    return r["token"]
def create_dataset(slug,title,files,owner="neimashube"):
    toks=[{"token":upload_blob(p)} for p in files]
    return req("POST","/datasets/create/new",{"ownerSlug":owner,"slug":slug,"title":title,"licenseName":"unknown","isPrivate":True,"convertToCsv":False,"files":toks})
def dataset_status(slug,owner="neimashube"): return req("GET",f"/datasets/status/{owner}/{slug}")
def push(meta,code):
    body={"slug":meta["id"],"newTitle":meta["title"],"text":open(code).read(),"language":"python","kernelType":"script","isPrivate":True,
          "enableGpu":True,"enableInternet":True,"machineShape":"NvidiaTeslaT4","datasetDataSources":meta.get("dataset_sources",[]),
          "competitionDataSources":[],"kernelDataSources":meta.get("kernel_sources",[]),"modelDataSources":[],"categoryIds":[]}
    return req("POST","/kernels/push",body)
def status(ref):
    o,s=ref.split("/"); return req("GET",f"/kernels/status?userName={o}&kernelSlug={s}")
def output(ref,dest):
    o,s=ref.split("/"); r=req("GET",f"/kernels/output?userName={o}&kernelSlug={s}")
    os.makedirs(dest,exist_ok=True)
    for f in r.get("files",[]):
        fn=os.path.join(dest,f["fileName"]); os.makedirs(os.path.dirname(fn),exist_ok=True)
        urllib.request.urlretrieve(f["url"],fn)
    if r.get("log"): open(os.path.join(dest,"log.json"),"w").write(r["log"])
    return [f["fileName"] for f in r.get("files",[])]
if __name__=="__main__":
    cmd=sys.argv[1]
    if cmd=="ds": print(create_dataset(sys.argv[2],sys.argv[3],sys.argv[4:]))
    elif cmd=="dss": print(dataset_status(sys.argv[2]))
    elif cmd=="push": print(push(json.load(open(sys.argv[2])),sys.argv[3]))
    elif cmd=="status": print(status(sys.argv[2]))
    elif cmd=="output": print(output(sys.argv[2],sys.argv[3]))
