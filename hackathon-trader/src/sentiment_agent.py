"""v2 sentiment agent: an LLM "analyst" that judges how each new document CHANGES its running view of a
ticker (trend, not an absolute score), with point-in-time RAG over that ticker's earlier documents.
Writes one sentiment_events row per (doc_id, symbol); see docs/V2_CONTRACT.md.
Days are processed chronologically; all docs of a day (all symbols) go to the LLM as one batch.
python -m src.sentiment_agent [--backend cortex|hf|fake] [--source tiger|bundle] [--sink tiger|bundle]
    [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--only AAPL,MSFT] [--batch-size N] [--estimate] [--import-bundle]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, cfg, get_logger, query_df, tiger, upsert

log = get_logger("sentiment_agent")

PROMPT_VERSION = "v1"  # bump whenever PROMPT or the parsing rules change
PROMPT = """You are an equity research analyst who maintains a running view of market sentiment toward {symbol} ({company}).
Your job is to judge how ONE new document CHANGES your prior view, not to score the document in isolation.
Example: a view that moves from 70 to 60 is still positive overall, but the direction is "deteriorating".

Rules:
- Use ONLY the text below. Do not use outside knowledge, and assume you know nothing that happened after {as_of}.
- direction and delta describe the change relative to the PRIOR VIEW: delta from -1 (much worse) to 1 (much better), 0 = no change.
- level (0-100, 50 = neutral) is your updated overall view after reading the new document.
- novelty (0-1): rate it LOW (below 0.2) if the document repeats information already in the prior view, the recent events or the earlier coverage; rate it high only for genuinely new information.
- materiality (0-1): how likely the new information is to move the stock price.
- horizon: how long the effect is likely to matter: "hours", "days" or "weeks".
- drivers: at most 4 short phrases (a few words each) naming what changed.
- updated_summary: at most 60 words, your updated running view of sentiment toward the stock.

PRIOR VIEW (as of {prior_date}): level {level:.0f}/100
{summary}

RECENT SENTIMENT EVENTS (oldest first):
{events}

EARLIER RELATED COVERAGE:
{related}

NEW DOCUMENT ({source} {doc_type}, published {published}):
Title: {title}
{body}

Answer with ONLY a JSON object, no other text:
{{"direction": "improving|deteriorating|stable", "delta": 0.0, "level": 50, "novelty": 0.0, "materiality": 0.0, "horizon": "hours|days|weeks", "drivers": ["..."], "updated_summary": "..."}}"""

OUTPUT_SCHEMA = {"type": "object", "properties": {
    "direction": {"type": "string", "enum": ["improving", "deteriorating", "stable"]},
    "delta": {"type": "number"}, "level": {"type": "number"}, "novelty": {"type": "number"},
    "materiality": {"type": "number"}, "horizon": {"type": "string", "enum": ["hours", "days", "weeks"]},
    "drivers": {"type": "array", "items": {"type": "string"}}, "updated_summary": {"type": "string"}},
    "required": ["direction", "delta", "level", "novelty", "materiality", "horizon", "drivers", "updated_summary"]}

DOC_COLS = ["doc_id", "symbol", "source", "doc_type", "published_at", "title", "body", "url", "content_hash"]
EV_COLS = ["doc_id", "symbol", "published_at", "direction", "delta", "level", "novelty", "materiality", "horizon",
           "drivers", "summary", "backend", "model", "prompt_version"]
KEYS = ["doc_id", "symbol"]
NO_PRIOR = "No prior coverage."
NEWS_CHARS, FILING_CHARS, RELATED_CHARS, EMBED_CHARS = 2500, 5000, 400, 1500
OUT_TOKENS = 150  # typical JSON answer length, for --estimate
# Snowflake credits per 1M tokens (Service Consumption Table; check current pricing before a big backfill)
CREDITS_PER_M = {"llama3.1-8b": 0.19, "llama3.1-70b": 1.21, "snowflake-arctic-embed-m-v1.5": 0.03}


# ---------- selection ----------

def trading_day(ts: pd.Series) -> pd.Series:
    """US/Eastern calendar date of published_at, weekends rolled forward to Monday."""
    d = pd.to_datetime(ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_datetime(np.busday_offset(d.values.astype("datetime64[D]"), 0, roll="forward")),
                     index=ts.index)


def select_docs(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Dedupe syndicated copies per symbol by content_hash (earliest wins), then keep at most `cap` docs per
    symbol-day (EDGAR filings first, then longest bodies). Adds `day`. Sorted by published_at."""
    if df.empty:
        return df.assign(day=pd.Series(dtype="datetime64[ns]"))
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["day"] = trading_day(df["published_at"])
    norm_title = df["title"].fillna("").str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df["_h"] = df["content_hash"].where(df["content_hash"].notna() & (df["content_hash"] != ""),
                                        norm_title.map(lambda t: hashlib.md5(t.encode()).hexdigest()))
    df = df.sort_values(["published_at", "doc_id"]).drop_duplicates(["symbol", "_h"], keep="first")
    df["_edgar"], df["_len"] = df["source"].eq("edgar"), df["body"].fillna("").str.len()
    df = (df.sort_values(["symbol", "day", "_edgar", "_len"], ascending=[True, True, False, False])
            .groupby(["symbol", "day"]).head(cap))
    return df.drop(columns=["_h", "_edgar", "_len"]).sort_values(["published_at", "doc_id"]).reset_index(drop=True)


def doc_text(row) -> str:
    n = NEWS_CHARS if (row.doc_type or "news") == "news" else FILING_CHARS
    return (row.body or "")[:n]


def embed_text(row) -> str:
    return f"{row.title or ''}\n{(row.body or '')[:EMBED_CHARS]}"


# ---------- retrieval (point-in-time) ----------

class Retriever:
    """Cosine top-k over the same symbol's docs published strictly before the query doc, within lookback_days."""

    def __init__(self, docs: pd.DataFrame, emb: np.ndarray):
        self.emb = emb
        self.ts = docs["published_at"].values.astype("datetime64[ns]").astype(np.int64)
        self.sym = docs["symbol"].values
        self.by_sym = {s: np.array(sorted(ix, key=lambda j: self.ts[j]), dtype=int)
                       for s, ix in docs.groupby("symbol").indices.items()}

    def query(self, i: int, k: int, lookback_days: float) -> list:
        ix = self.by_sym[self.sym[i]]
        t, ts = self.ts[i], self.ts[ix]
        lo = np.searchsorted(ts, t - int(lookback_days * 86400e9), "left")
        hi = np.searchsorted(ts, t, "left")  # first doc at/after t -> excluded
        cand = ix[lo:hi]
        if k <= 0 or not len(cand):
            return []
        sims = self.emb[cand] @ self.emb[i]
        return [int(j) for j in cand[np.argsort(-sims, kind="stable")[:k]]]


def embed_docs(docs: pd.DataFrame, backend, cache_dir: Path | None = None, chunk: int = 2000) -> np.ndarray:
    """Embed every selected doc once; in bundle mode reuse/extend <bundle>/embeddings.npy (+ _ids, _meta)."""
    ids = docs["doc_id"].astype(np.int64).values
    have, old_e, old_i = {}, None, None
    if cache_dir is not None:
        pe, pi, pm = cache_dir / "embeddings.npy", cache_dir / "embeddings_ids.npy", cache_dir / "embeddings_meta.json"
        if pe.exists() and pi.exists() and pm.exists() and json.loads(pm.read_text()).get("tag") == backend.embed_tag:
            old_e, old_i = np.load(pe), np.load(pi)
            have = {int(d): j for j, d in enumerate(old_i)}
    miss = [j for j, d in enumerate(ids) if int(d) not in have]
    texts = [embed_text(r) for r in docs.iloc[miss].itertuples()] if miss else []
    log.info("embeddings: %d cached, %d to compute", len(ids) - len(miss), len(miss))
    new = [backend.embed(texts[j:j + chunk]) for j in range(0, len(texts), chunk)]
    new = np.concatenate(new) if new else None
    dim = (new if new is not None else old_e).shape[1] if (new is not None or old_e is not None) else 1
    out = np.zeros((len(ids), dim), dtype=np.float32)
    for j, d in enumerate(ids):
        if int(d) in have:
            out[j] = old_e[have[int(d)]]
    if miss:
        out[miss] = new
        if cache_dir is not None:
            all_e = new if old_e is None else np.concatenate([old_e, new])
            all_i = ids[miss] if old_i is None else np.concatenate([old_i, ids[miss]])
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / "embeddings.npy", all_e.astype(np.float32))
            np.save(cache_dir / "embeddings_ids.npy", all_i.astype(np.int64))
            (cache_dir / "embeddings_meta.json").write_text(json.dumps({"tag": backend.embed_tag}))
    return out


# ---------- prompt / parsing ----------

def _drivers(v) -> list:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = [v]
    return [str(x) for x in (v if isinstance(v, (list, tuple, np.ndarray)) else [])]


def build_prompt(doc, mem: dict | None, recent: list, related: list) -> str:
    from .sectors import SYMBOLS
    mem = mem or {"level": 50.0, "summary": NO_PRIOR, "date": None}
    ev = "\n".join(f"- {e['date']:%Y-%m-%d} {e['direction']} ({'; '.join(e['drivers']) or 'n/a'})" for e in recent)
    rel = "\n".join(f"- [{r.published_at:%Y-%m-%d} {r.doc_type}] {r.title or ''}: "
                    f"{' '.join((r.body or '').split())[:RELATED_CHARS]}" for r in related)
    return PROMPT.format(
        symbol=doc.symbol, company=SYMBOLS.get(doc.symbol, (doc.symbol,))[0],
        as_of=f"{doc.published_at:%Y-%m-%d %H:%M} UTC", prior_date=f"{mem['date']:%Y-%m-%d}" if mem["date"] is not None else "n/a",
        level=float(mem["level"]), summary=mem["summary"], events=ev or "none", related=rel or "none",
        source=doc.source, doc_type=doc.doc_type, published=f"{doc.published_at:%Y-%m-%d %H:%M} UTC",
        title=doc.title or "", body=doc_text(doc))


def _first_json(text: str):
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(x) else float(min(hi, max(lo, x)))


def parse_output(text: str, prior_level: float = 50.0) -> dict:
    """Extract the first JSON object, clamp ranges; on failure return a neutral 'parse_error' answer."""
    o = _first_json(text)
    if o is None:
        return dict(direction="stable", delta=0.0, level=float(prior_level), novelty=0.0, materiality=0.0,
                    horizon="days", drivers=[], updated_summary="parse_error")
    delta = _num(o.get("delta"), -1, 1, 0.0)
    d = str(o.get("direction", "")).strip().lower()
    if d not in ("improving", "deteriorating", "stable"):
        d = "improving" if delta > 0.05 else "deteriorating" if delta < -0.05 else "stable"
    h = str(o.get("horizon", "")).strip().lower()
    summ = " ".join(str(o.get("updated_summary") or "").split()[:60])
    return dict(direction=d, delta=delta, level=_num(o.get("level"), 0, 100, float(prior_level)),
                novelty=_num(o.get("novelty"), 0, 1, 0.0), materiality=_num(o.get("materiality"), 0, 1, 0.0),
                horizon=h if h in ("hours", "days", "weeks") else "days",
                drivers=[s.strip()[:80] for s in _drivers(o.get("drivers")) if s.strip()][:4],
                updated_summary=summ or "parse_error")


# ---------- memory ----------

def update_memory(mem: dict, recent: dict, day_events: pd.DataFrame, n_recent: int) -> None:
    """Fold one day's events into each symbol's state. level moves toward the materiality x novelty weighted
    mean of the day's levels by min(1, total weight); summary = that of the day's most material doc."""
    if day_events.empty:
        return
    for sym, g in day_events.sort_values("published_at").groupby("symbol"):
        st = mem.get(sym, {"level": 50.0, "summary": NO_PRIOR, "date": None})
        w = (g["materiality"].astype(float) * g["novelty"].astype(float)).clip(0, 1).fillna(0).values
        lv, W = g["level"].astype(float).fillna(st["level"]).values, w.sum()
        level = st["level"] + min(1.0, W) * ((w * lv).sum() / W - st["level"]) if W > 0 else st["level"]
        ok = g[g["summary"].fillna("parse_error") != "parse_error"]
        summary = ok.sort_values(["materiality", "novelty"], kind="stable")["summary"].iloc[-1] if len(ok) else st["summary"]
        mem[sym] = {"level": float(level), "summary": summary, "date": g["day"].iloc[0]}
        recent[sym] = (recent[sym] + [{"date": r.day, "direction": r.direction, "drivers": _drivers(r.drivers)}
                                      for r in g.itertuples()])[-n_recent:]


# ---------- core loop ----------

def run(docs: pd.DataFrame, emb: np.ndarray, backend, existing: pd.DataFrame, process, sc: dict,
        batch_size: int = 512, sink=None, flush_rows: int = 500) -> pd.DataFrame:
    """Score docs[process] day by day. `existing` events are replayed so memory is identical on resume.
    sink(df) is called with new rows every ~flush_rows rows and at the end (also on errors)."""
    process = np.asarray(process, dtype=bool)
    if not process.any():
        return pd.DataFrame(columns=EV_COLS)
    k, n_recent, lookback = int(sc.get("retrieve_k", 3)), int(sc.get("memory_events", 5)), float(sc.get("lookback_days", 90))
    docs = docs.reset_index(drop=True)  # emb is positional
    ret = Retriever(docs, emb)
    existing = existing.copy()
    existing["day"] = trading_day(existing["published_at"]) if len(existing) else pd.Series(dtype="datetime64[ns]")
    todo_days = docs.loc[process, "day"]
    ex_by_day = {d: g for d, g in existing[existing["day"] <= todo_days.max()].groupby("day")}
    days = sorted(set(todo_days) | set(ex_by_day))
    mem, recent, out, buf, t0, n_done = {}, defaultdict(list), [], [], time.time(), 0

    def flush():
        if buf and sink is not None:
            sink(pd.DataFrame(buf, columns=EV_COLS))
        out.extend(buf)
        buf.clear()

    try:
        for day in days:
            todo = np.flatnonzero(process & (docs["day"] == day).values)
            rows = []
            for c in range(0, len(todo), batch_size):
                chunk = todo[c:c + batch_size]
                prompts = [build_prompt(docs.iloc[i], mem.get(docs.at[i, "symbol"]),
                                        recent[docs.at[i, "symbol"]],
                                        [docs.iloc[j] for j in ret.query(i, k, lookback)]) for i in chunk]
                for i, txt in zip(chunk, backend.complete_batch(prompts)):
                    d = docs.iloc[i]
                    p = parse_output(txt, mem.get(d.symbol, {"level": 50.0})["level"])
                    rows.append([int(d.doc_id), d.symbol, d.published_at, p["direction"], p["delta"], p["level"],
                                 p["novelty"], p["materiality"], p["horizon"], json.dumps(p["drivers"]),
                                 p["updated_summary"], backend.name, backend.model_name, PROMPT_VERSION])
            new = pd.DataFrame(rows, columns=EV_COLS).assign(day=day)
            parts = [x for x in (ex_by_day.get(day), new) if x is not None and len(x)]
            if parts:
                update_memory(mem, recent, pd.concat(parts, ignore_index=True), n_recent)
            buf.extend(rows)
            n_done += len(rows)
            if rows:
                log.info("%s: %d docs (%d total, %.2f docs/s)", f"{day:%Y-%m-%d}", len(rows), n_done,
                         n_done / max(time.time() - t0, 1e-9))
            if len(buf) >= flush_rows:
                flush()
    finally:
        flush()
    return pd.DataFrame(out, columns=EV_COLS)


# ---------- IO ----------

def load_documents(source: str, bundle: Path, start=None, end=None, only=None) -> pd.DataFrame:
    if source == "bundle":
        df = pd.read_parquet(bundle / "documents.parquet", columns=DOC_COLS)
    else:
        where, params = [], []
        if start is not None:
            where.append("published_at >= %s"), params.append(start.to_pydatetime())
        if end is not None:
            where.append("published_at < %s"), params.append(end.to_pydatetime())
        if only:
            where.append("symbol = ANY(%s)"), params.append(list(only))
        cols = ", ".join(c if c != "body" else f"LEFT(body, {FILING_CHARS}) AS body" for c in DOC_COLS)
        df = query_df(f"SELECT {cols} FROM documents" + (" WHERE " + " AND ".join(where) if where else ""), params)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["published_at"] >= start
    if end is not None:
        m &= df["published_at"] < end
    if only:
        m &= df["symbol"].isin(only)
    return df[m].reset_index(drop=True)


def load_events(sink: str, bundle: Path, only=None) -> pd.DataFrame:
    if sink == "bundle":
        p = bundle / "sentiment_events.parquet"
        df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=EV_COLS)
    else:
        sql = f"SELECT {', '.join(EV_COLS)} FROM sentiment_events" + (" WHERE symbol = ANY(%s)" if only else "")
        df = query_df(sql, [list(only)] if only else None)
    if only:
        df = df[df["symbol"].isin(only)]
    df = df.reset_index(drop=True)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def write_events(sink: str, bundle: Path, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df[EV_COLS]
    if sink == "bundle":
        p = bundle / "sentiment_events.parquet"
        if p.exists():
            df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        df = df.drop_duplicates(KEYS, keep="last").sort_values(["published_at", "doc_id"]).reset_index(drop=True)
        df["doc_id"] = df["doc_id"].astype(np.int64)
        for c in ("delta", "level", "novelty", "materiality"):
            df[c] = df[c].astype(float)
        bundle.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)  # atomic: a killed job never leaves a half-written file
        return len(df)
    with tiger() as conn:
        return upsert(conn, "sentiment_events", df, KEYS)


def import_bundle(bundle: Path) -> int:
    df = load_events("bundle", bundle)
    n = 0
    for j in range(0, len(df), 5000):
        n += write_events("tiger", bundle, df.iloc[j:j + 5000])
    log.info("imported %d sentiment_events rows into Tiger", n)
    return n


def estimate(docs: pd.DataFrame, process, sc: dict, backend: str) -> dict:
    sel = docs[np.asarray(process, dtype=bool)]
    k, n_ev = int(sc.get("retrieve_k", 3)), int(sc.get("memory_events", 5))
    overhead = len(PROMPT) + 400 + k * (RELATED_CHARS + 60) + n_ev * 80 + 150  # template, summary, RAG, events
    chars = sum(len(doc_text(r)) + len(r.title or "") for r in sel.itertuples()) + overhead * len(sel)
    emb_chars = sum(len(embed_text(r)) for r in docs.itertuples())
    r = {"docs": len(sel), "symbols": int(sel["symbol"].nunique()), "day_batches": int(sel["day"].nunique()),
         "input_tokens": int(chars / 4), "output_tokens": OUT_TOKENS * len(sel), "embed_docs": len(docs),
         "embed_tokens": int(emb_chars / 4)}
    if backend == "cortex":
        llm = CREDITS_PER_M.get(sc["cortex_model"], float("nan"))
        emb = CREDITS_PER_M.get(sc["cortex_embed_model"], float("nan"))
        r["approx_credits"] = round(((r["input_tokens"] + r["output_tokens"]) * llm + r["embed_tokens"] * emb) / 1e6, 2)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    sc = cfg()["sentiment"]
    ap.add_argument("--backend", choices=["cortex", "hf", "fake"], default=sc.get("backend", "cortex"))
    ap.add_argument("--source", choices=["tiger", "bundle"], default="tiger")
    ap.add_argument("--sink", choices=["tiger", "bundle"], default="tiger")
    ap.add_argument("--bundle-dir", default=cfg().get("data", {}).get("bundle_dir", "data/bundle"))
    ap.add_argument("--since", help="first trading day to score (default: 2 days before the oldest per-symbol "
                    "latest stored event; pass it explicitly for newly added tickers)")
    ap.add_argument("--until", help="last day to score (inclusive)")
    ap.add_argument("--only", help="comma-separated tickers")
    ap.add_argument("--batch-size", type=int, default=512, help="max prompts per LLM call (splits huge days)")
    ap.add_argument("--hf-batch", type=int, default=16, help="GPU micro-batch for --backend hf")
    ap.add_argument("--estimate", action="store_true", help="print doc/token counts, call no LLM")
    ap.add_argument("--import-bundle", action="store_true", help="load <bundle>/sentiment_events.parquet into Tiger")
    a = ap.parse_args(argv)
    bundle = Path(a.bundle_dir) if Path(a.bundle_dir).is_absolute() else ROOT / a.bundle_dir
    if a.import_bundle:
        return import_bundle(bundle)
    if sc.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
        log.warning("config prompt_version=%s but code PROMPT_VERSION=%s; rows store the code version",
                    sc.get("prompt_version"), PROMPT_VERSION)
    only = [s.strip().upper() for s in a.only.split(",")] if a.only else None
    try:
        existing = load_events(a.sink, bundle, only)
    except Exception as e:
        if not a.estimate:
            raise
        log.warning("could not read existing events (%s); estimating as if none exist", e)
        existing = pd.DataFrame(columns=EV_COLS)
    if a.since:
        since = pd.Timestamp(a.since)
    elif len(existing):  # incremental: every symbol with events is complete up to its own latest day
        since = trading_day(existing["published_at"]).groupby(existing["symbol"]).max().min() - pd.Timedelta(days=2)
    else:
        since = None
    until = pd.Timestamp(a.until) if a.until else None
    lookback = float(sc.get("lookback_days", 90))
    start = (since - pd.Timedelta(days=lookback + 1)).tz_localize("UTC") if since is not None else None
    end = (until + pd.Timedelta(days=2)).tz_localize("UTC") if until is not None else None
    docs = select_docs(load_documents(a.source, bundle, start, end, only), int(sc.get("max_docs_per_symbol_day", 6)))
    if until is not None:
        docs = docs[docs["day"] <= until].reset_index(drop=True)
    done = set(zip(existing["doc_id"].astype(np.int64), existing["symbol"]))
    process = np.array([(int(d), s) not in done for d, s in zip(docs["doc_id"], docs["symbol"])], dtype=bool)
    if since is not None:
        process &= (docs["day"] >= since).values
    log.info("%d docs in window (incl. lookback pool), %d to score, %d existing events", len(docs),
             int(process.sum()), len(existing))
    if a.estimate:
        r = estimate(docs, process, sc, a.backend)
        print(json.dumps(r, indent=2))
        return r
    if not process.any():
        log.info("nothing to score")
        return pd.DataFrame(columns=EV_COLS)
    from .llm_backends import make_backend
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # SLURM time limit -> flush buffered rows
    backend = make_backend(a.backend, sc, OUTPUT_SCHEMA, hf_batch=a.hf_batch)
    emb = embed_docs(docs, backend, bundle if a.source == "bundle" else None)
    new = run(docs, emb, backend, existing, process, sc, a.batch_size,
              sink=lambda df: write_events(a.sink, bundle, df))
    log.info("wrote %d sentiment_events (%s, %s, prompt %s)%s", len(new), backend.name, backend.model_name,
             PROMPT_VERSION, f", cortex tokens={backend.tokens}" if hasattr(backend, "tokens") else "")
    return new


if __name__ == "__main__":
    main()
