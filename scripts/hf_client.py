"""Hugging Face Inference Providers client (router.huggingface.co, OpenAI-compatible) for run three.

The HF token is injected by the sandbox proxy; nothing secret lives here. Every call's token usage is priced
from the router's own /v1/models pricing table and appended to work/run3/hf_spend.jsonl, and a hard cap
(HF_CAP_USD, default 1.80 of Anu's approved $2.00) stops all further calls once reached. Billing/credit
errors (402 etc.) raise BillingError so callers stop instead of retrying.
"""
import json, os, threading, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://router.huggingface.co/v1/chat/completions"
SPEND = f"{ROOT}/work/run3/hf_spend.jsonl"
# HF's billed amount ran ~1.44x the router-list-price estimate logged here (checked 2026-11-05 via
# /api/settings/billing/usage-v2), so the local cap is 1.20 logged ~= 1.75 billed, under the approved $2.00.
CAP = float(os.environ.get("HF_CAP_USD", "1.20"))
_lock = threading.Lock()
_price = {}


class BillingError(RuntimeError):
    pass


def _pricing():
    if not _price:
        r = urllib.request.Request("https://router.huggingface.co/v1/models", headers={"Authorization": "Bearer x"})
        for m in json.loads(urllib.request.urlopen(r, timeout=60).read())["data"]:
            for p in m.get("providers", []):
                if p.get("pricing"):
                    _price[f"{m['id']}:{p['provider']}"] = (p["pricing"]["input"], p["pricing"]["output"])
    return _price


def spent_usd():
    if not os.path.exists(SPEND):
        return 0.0
    return sum(json.loads(l)["usd"] for l in open(SPEND))


def chat(model, messages, max_tokens=800, temperature=0.0, retries=3, tag="", extra=None):
    """model like 'google/gemma-4-31B-it:deepinfra' (provider pinned so pricing is known)."""
    pin, pout = _pricing()[model]
    if spent_usd() >= CAP:
        raise BillingError(f"local HF cap ${CAP} reached")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if extra:
        body.update(extra)
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": "Bearer x", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as f:
                j = json.loads(f.read())
            u = j.get("usage") or {}
            usd = (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
            with _lock, open(SPEND, "a") as f:
                f.write(json.dumps({"t": time.time(), "model": model, "tag": tag, "in": u.get("prompt_tokens", 0),
                                    "out": u.get("completion_tokens", 0), "usd": usd}) + "\n")
            return j["choices"][0]["message"].get("content") or "", u
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode(errors="replace")
            if e.code == 402 or any(k in msg.lower() for k in ("credit", "billing", "payment", "insufficient", "quota")):
                raise BillingError(f"HTTP {e.code}: {msg}")
            last = f"HTTP {e.code}: {msg}"
        except BillingError:
            raise
        except Exception as e:
            last = repr(e)[:300]
        time.sleep(4 * (a + 1))
    raise RuntimeError(f"hf chat failed after {retries}: {last}")


def parse_json(text):
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    s, e = t.find("["), t.rfind("]")
    s2, e2 = t.find("{"), t.rfind("}")
    if s != -1 and (s2 == -1 or s < s2):
        return json.loads(t[s:e + 1])
    return json.loads(t[s2:e2 + 1])
