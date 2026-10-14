import json, os, re, io, contextlib
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import urllib.parse as _up
_raw = "".join(os.environ["LEDGER_URL"].split())            # remove ALL whitespace/newlines
_u = _up.urlparse(_raw); _q = dict(_up.parse_qsl(_u.query)); _q["path"] = "/"
LEDGER_URL = _up.urlunparse(_u._replace(query=_up.urlencode(_q)))   # front door, path exactly "/"                     # your long URL (with path=/)
LLM_KEY = os.environ["LLM_API_KEY"].strip()
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def with_path(path):
    u = urlparse(LEDGER_URL)
    q = dict(parse_qsl(u.query)); q["path"] = path
    return urlunparse(u._replace(query=urlencode(q)))


def table_name(url):
    q = dict(parse_qsl(urlparse(url).query))
    return re.sub(r"\W+", "_", q.get("path", "root")).strip("_") or "root"


SESSION = requests.Session()


def fetch(url):
    try:
        resp = SESSION.get(url, timeout=20)
    except Exception:
        return url, None, ""
    try:
        return url, resp.json(), resp.text
    except Exception:
        pass
    try:                                   # NDJSON: one JSON object per line
        recs = [json.loads(l) for l in resp.text.splitlines() if l.strip()]
        if recs and all(isinstance(r, dict) for r in recs):
            return url, recs, resp.text
    except Exception:
        pass
    return url, None, resp.text


def crawl(max_requests=400, time_budget=90):
    """Breadth-first, 10 requests at a time. Follows full URLs on the same host,
    guesses the next pages when a URL has page=N, keeps non-table JSON as notes."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    host = urlparse(LEDGER_URL).netloc
    seen, rows, notes = set(), {}, {}
    queue = [LEDGER_URL]
    with ThreadPoolExecutor(max_workers=10) as pool:
        while queue and len(seen) < max_requests and time.time() - t0 < time_budget:
            batch = []
            for u in queue:
                u = "".join(u.split())
                if u not in seen:
                    seen.add(u); batch.append(u)
            queue = []
            for url, data, text in pool.map(fetch, batch):
                name = table_name(url)
                if data is None:                       # maybe CSV
                    try:
                        df = pd.read_csv(io.StringIO(text))
                        if len(df.columns) > 1:
                            rows.setdefault(name, []).extend(df.to_dict("records"))
                    except Exception:
                        pass
                    continue
                found = [False]

                def walk(x, label):
                    if isinstance(x, list) and x and all(isinstance(i, dict) for i in x):
                        rows.setdefault(label, []).extend(x); found[0] = True
                    elif isinstance(x, dict):
                        for k, v in x.items():
                            walk(v, label if k in ("data", "items", "results", "orders", "rows", "records") or label == name and False else f"{label}_{k}" if label != name else label)
                            if isinstance(v, str) and v.startswith("http") and urlparse(v).netloc == host:
                                queue.append(v)
                    elif isinstance(x, list):
                        for i in x:
                            if isinstance(i, str) and i.startswith("http") and urlparse(i).netloc == host:
                                queue.append(i)
                walk(data, name)
                q = dict(parse_qsl(urlparse(url).query))
                if found[0] and q.get("page", "").isdigit():
                    for k in range(1, 9):              # speculative next pages, fetched in parallel
                        q2 = dict(q); q2["page"] = str(int(q["page"]) + k)
                        queue.append(urlunparse(urlparse(url)._replace(query=urlencode(q2))))
                if not found[0]:
                    notes[name] = data
    dfs = {k: pd.DataFrame(v) for k, v in rows.items()}
    return dfs, notes


DFS, NOTES = {}, {}


def build_ledger(dfs, notes):
    """Merge orders + export, keep newest row per id, add USD amount and IST dates."""
    frames = [d for n, d in dfs.items() if "id" in d.columns and "amount" in d.columns]
    df = pd.concat(frames, ignore_index=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.sort_values("updated_at", kind="stable").drop_duplicates("id", keep="last")
    rates = notes.get("rates", {}).get("usd_per_unit", {})
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["amount_usd"] = df["amount"] * df["currency"].map(rates)
    df["created_ist"] = df["created_at"].dt.tz_convert("Asia/Kolkata")
    df["updated_ist"] = df["updated_at"].dt.tz_convert("Asia/Kolkata")
    df["year"] = df["created_ist"].dt.year
    df["month"] = df["created_ist"].dt.month
    return df.reset_index(drop=True)



def describe():
    hints = {k: ({kk: vv for kk, vv in v.items() if kk != "links"} if isinstance(v, dict) else v)
             for k, v in NOTES.items()}
    parts = ["API hints/metadata:\n" + json.dumps(hints, default=str)[:4000]]
    for name, df in DFS.items():
        uniq = {c: sorted(map(str, df[c].dropna().unique()))[:60] for c in df.columns
                if df[c].dtype == object and df[c].nunique() <= 60}
        parts.append(f"\nDataFrame T['{name}'] ({len(df)} rows)\ncolumns/dtypes: {dict(df.dtypes.astype(str))}\n"
                     f"allowed values: {json.dumps(uniq)}\nsample:\n{df.head(3).to_json(orient='records', date_format='iso')}")
    return "\n".join(parts)


def llm(messages):
    r = requests.post(f"{LLM_BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {LLM_KEY}"},
                      json={"model": LLM_MODEL, "messages": messages, "temperature": 0},
                      timeout=10)
    return r.json()["choices"][0]["message"]["content"]


SYSTEM = """You answer questions about Acme's order ledger using pandas.
`T['ledger']` is ALREADY cleaned: duplicates removed (newest row per id kept), and these columns added:
- amount_usd: amount converted to USD (use this for ALL money questions)
- created_ist / updated_ist: timezone-aware datetimes in Asia/Kolkata; year, month come from created_ist
Rules:
- Revenue = sum of amount_usd where status == 'paid'. For refunds or other statuses, look at the
  allowed status values listed below and choose the matching one.
- An order's date is created_ist unless the question clearly means the update/refund date.
- Round money to 2 decimals. Use product/customer/region names exactly as in the data.
- Counts of orders = number of rows (ids are unique already).
Write Python that stores the final result in a variable `answer` (number, string or list).
Output ONLY a python code block."""


class Q(BaseModel):
    question: str


import threading
READY = threading.Event()
STATUS = {"state": "loading"}


def _load():
    global DFS, NOTES
    try:
        raw, NOTES = crawl()
        DFS = {"ledger": build_ledger(raw, NOTES)}
        STATUS["state"] = "ready"
    except Exception as e:
        STATUS["state"] = f"error: {e}"
    READY.set()


@app.on_event("startup")
def load():
    threading.Thread(target=_load, daemon=True).start()   # server starts right away


@app.get("/")
def health():
    led = DFS.get("ledger")
    values = {}
    if led is not None:
        values = {c: sorted(map(str, led[c].dropna().unique()))[:60] for c in led.columns
                  if led[c].dtype == object and led[c].nunique() <= 60}
    hints = {k: ({kk: vv for kk, vv in v.items() if kk != "links"} if isinstance(v, dict) else v)
             for k, v in NOTES.items()}
    return {"status": STATUS["state"],
            "rows": None if led is None else len(led),
            "columns": None if led is None else list(led.columns),
            "values": values, "notes": hints}


@app.post("/")
def ask(q: Q):
    READY.wait(timeout=12)
    msgs = [{"role": "system", "content": SYSTEM + "\n\n" + describe()},
            {"role": "user", "content": q.question}]
    err = ""
    for _ in range(2):                                   # one retry if the code crashes
        code = llm(msgs)
        code = re.sub(r"^```(?:python)?\s*|```\s*$", "", code.strip(), flags=re.M)
        ns = {"T": DFS, "pd": pd, "answer": None}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, ns)
            ans = ns["answer"]
            def clean(x):
                if hasattr(x, "item") and not isinstance(x, (list, dict)):
                    try: x = x.item()
                    except Exception: pass
                if isinstance(x, (list, tuple)): return [clean(i) for i in x]
                if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
                if isinstance(x, float): return round(x, 2)
                if x is None or isinstance(x, (int, str, bool)): return x
                return str(x)
            ans = clean(ans)
            if isinstance(ans, float):
                ans = round(ans, 2)
            return {"answer": ans}
        except Exception as e:
            err = str(e)
            msgs += [{"role": "assistant", "content": code},
                     {"role": "user", "content": f"That crashed: {err}. Fix it. Code block only."}]
    return {"answer": None, "error": err}