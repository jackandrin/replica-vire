"""Export TigerData v2 tables -> config data.bundle_dir/*.parquet + manifest.json (docs/V2_CONTRACT.md).
Streams with a server-side cursor so memory stays bounded; the bundle is what the HPC side consumes.
python -m src.export_dataset [--start 2019-01-01] [--end 2026-01-01] [--only AAPL,SPY]"""
import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import ROOT, cfg, get_logger, tiger

log = get_logger("export_dataset")
CHUNK = 200_000
TS, F, S = pa.timestamp("us", tz="UTC"), pa.float64(), pa.string()
TABLES = {  # name -> (time column used by --start/--end, ORDER BY, arrow schema)
    "symbols": (None, "symbol", pa.schema([("symbol", S), ("company", S), ("sector", S), ("sector_etf", S),
                                            ("kind", S), ("cik", S)])),
    "bars_15m": ("ts", "symbol, ts", pa.schema([("symbol", S), ("ts", TS), ("open", F), ("high", F), ("low", F),
                                                 ("close", F), ("volume", F), ("vwap", F), ("trade_count", F)])),
    "bars_1d": ("d", "symbol, d", pa.schema([("symbol", S), ("d", pa.date32()), ("open", F), ("high", F),
                                              ("low", F), ("close", F), ("volume", F), ("vwap", F)])),
    "events": ("event_time", "symbol, kind, event_time", pa.schema([
        ("symbol", S), ("kind", S), ("event_time", TS), ("available_at", TS), ("value", F), ("key", S),
        ("payload", S)])),
    "documents": ("published_at", "symbol, published_at, doc_id", pa.schema([
        ("doc_id", pa.int64()), ("symbol", S), ("source", S), ("doc_type", S), ("published_at", TS),
        ("title", S), ("body", S), ("url", S), ("content_hash", S)])),
    "sentiment_events": ("published_at", "symbol, published_at, doc_id", pa.schema([
        ("doc_id", pa.int64()), ("symbol", S), ("published_at", TS), ("direction", S), ("delta", F),
        ("level", F), ("novelty", F), ("materiality", F), ("horizon", S), ("drivers", S), ("summary", S),
        ("backend", S), ("model", S), ("prompt_version", S)])),
}
OPTIONAL = {"sentiment_events"}  # written by the sentiment agent; only exported when the table has rows


def to_arrow(rows: list, schema: pa.Schema) -> pa.Table:
    df = pd.DataFrame(rows, columns=schema.names)
    for f in schema:
        if pa.types.is_timestamp(f.type):
            df[f.name] = pd.to_datetime(df[f.name], utc=True)
        elif pa.types.is_date32(f.type):
            df[f.name] = pd.to_datetime(df[f.name]).dt.date
        elif pa.types.is_floating(f.type):
            df[f.name] = pd.to_numeric(df[f.name], errors="coerce").astype("float64")
        elif pa.types.is_integer(f.type):
            df[f.name] = pd.to_numeric(df[f.name]).astype("int64")
        else:
            df[f.name] = df[f.name].astype(object).where(df[f.name].notna(), None)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def build_query(name: str, start, end, only) -> tuple:
    tcol, order, schema = TABLES[name]
    where, params = [], []
    if tcol and start:
        where.append(f"{tcol} >= %s")
        params.append(start)
    if tcol and end:
        where.append(f"{tcol} < %s")
        params.append(end)
    if only:
        where.append("symbol = ANY(%s)")
        params.append(list(only))
    sql = f"SELECT {', '.join(schema.names)} FROM {name}" + (" WHERE " + " AND ".join(where) if where else "")
    return sql + f" ORDER BY {order}", params


def export_table(name: str, out_dir, start, end, only) -> int:
    schema = TABLES[name][2]
    sql, params = build_query(name, start, end, only)
    path, tmp = out_dir / f"{name}.parquet", out_dir / f".{name}.parquet.tmp"
    n, writer = 0, None
    with tiger() as conn:
        with conn.cursor() as c:
            c.execute("SET TIME ZONE 'UTC'")
        with conn.cursor(name=f"export_{name}") as cur:  # server-side cursor
            cur.itersize = CHUNK
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(CHUNK)
                if not rows:
                    break
                if writer is None:
                    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
                writer.write_table(to_arrow(rows, schema))
                n += len(rows)
                log.info("%s: %d rows", name, n)
    if writer is None:
        if name in OPTIONAL:
            log.info("%s: table empty, leaving any existing %s untouched", name, path.name)
            return 0
        pq.write_table(schema.empty_table(), tmp, compression="zstd")
    else:
        writer.close()
    os.replace(tmp, path)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default config data.start)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD exclusive (default: no limit)")
    ap.add_argument("--only", help="comma-separated symbols")
    a = ap.parse_args(argv)
    start = pd.Timestamp(a.start or cfg()["data"]["start"], tz="UTC").to_pydatetime()
    end = pd.Timestamp(a.end, tz="UTC").to_pydatetime() if a.end else None
    only = [s.strip().upper() for s in a.only.split(",")] if a.only else None
    out_dir = ROOT / cfg()["data"].get("bundle_dir", "data/bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in TABLES:
        try:
            n = export_table(name, out_dir, start, end, only)
        except Exception as e:  # noqa: BLE001
            if name not in OPTIONAL:
                raise
            log.warning("%s skipped: %s", name, str(e)[:200])
            n = 0
        if n or name not in OPTIONAL:
            counts[name] = n
    syms = sorted(pd.read_parquet(out_dir / "symbols.parquet", columns=["symbol"]).symbol)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "start": start.date().isoformat(),
                "end": end.date().isoformat() if end else None, "symbols": syms, "rows": counts}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("bundle written to %s: %s", out_dir, counts)


if __name__ == "__main__":
    main()
