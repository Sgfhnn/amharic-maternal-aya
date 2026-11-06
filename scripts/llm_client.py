"""Tiny OpenAI-compatible chat client for the Cline API (api.cline.bot), used for the curation mismatch check
and the LLM judge. The key is injected by the sandbox proxy (service 'cline'); nothing secret lives here.
Cline only returns content when stream=true, so we parse SSE. Stops hard on any billing / credit error.
"""
import json, time, urllib.request, urllib.error

URL = "https://api.cline.bot/api/v1/chat/completions"
BAL = "https://api.cline.bot/api/v1/users/usr-01KX87YYQ541V4TWQ89G33NTGP/balance"


class BillingError(RuntimeError):
    pass


def balance():
    r = urllib.request.Request(BAL, headers={"Authorization": "Bearer x"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())["data"]["balance"]  # micro-USD


def chat(model, messages, max_tokens=800, temperature=0.0, retries=3, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    if extra:
        body.update(extra)
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": "Bearer x", "Content-Type": "application/json"})
            out, usage = [], None
            with urllib.request.urlopen(req, timeout=300) as f:
                for raw in f:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        j = json.loads(d)
                    except Exception:
                        continue
                    if j.get("error"):
                        raise RuntimeError(str(j["error"])[:300])
                    for c in j.get("choices", []):
                        t = (c.get("delta") or {}).get("content")
                        if t:
                            out.append(t)
                    if j.get("usage"):
                        usage = j["usage"]
            return "".join(out), usage
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode(errors="replace")
            if e.code in (402,) or any(k in msg.lower() for k in ("credit", "billing", "payment", "insufficient")):
                raise BillingError(f"HTTP {e.code}: {msg}")
            last = f"HTTP {e.code}: {msg}"
        except BillingError:
            raise
        except Exception as e:
            last = repr(e)[:300]
            if any(k in last.lower() for k in ("credit", "billing", "payment", "insufficient")):
                raise BillingError(last)
        time.sleep(3 * (a + 1))
    raise RuntimeError(f"chat failed after {retries}: {last}")


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
