"""Sentiment agent checks with the deterministic FakeBackend. Run: pytest -q tests/test_sentiment_agent.py
(no network, database, torch or transformers needed)."""
import json

import numpy as np
import pandas as pd

from src import sentiment_agent as sa
from src.llm_backends import FakeBackend

SC = {"retrieve_k": 3, "memory_events": 5, "lookback_days": 90, "max_docs_per_symbol_day": 6,
      "cortex_model": "llama3.1-8b", "cortex_embed_model": "snowflake-arctic-embed-m-v1.5"}


def _docs(rows):
    """rows: (doc_id, symbol, 'YYYY-MM-DD HH:MM' ET, title, body[, source, doc_type, content_hash])"""
    out = []
    for r in rows:
        r = list(r) + [None] * (8 - len(r))
        out.append(dict(doc_id=r[0], symbol=r[1], source=r[5] or "alpaca", doc_type=r[6] or "news",
                        published_at=pd.Timestamp(r[2], tz="America/New_York").tz_convert("UTC"),
                        title=r[3], body=r[4], url=f"https://x/{r[0]}", content_hash=r[7] or f"h{r[0]}"))
    return pd.DataFrame(out)


def _corpus():
    return _docs([
        (1, "AAA", "2024-03-04 09:00", "AAA beats estimates", "AAA beat and raised guidance"),
        (2, "AAA", "2024-03-04 15:00", "AAA product launch", "AAA launches a widget"),
        (3, "BBB", "2024-03-04 10:00", "BBB misses", "BBB miss on revenue, cut outlook"),
        (4, "AAA", "2024-03-05 11:00", "AAA guidance cut", "AAA cut guidance, miss expected"),
        (5, "BBB", "2024-03-05 12:00", "BBB flat quarter", "BBB reported a flat quarter"),
        (6, "AAA", "2024-03-06 09:30", "AAA beats again", "AAA beat estimates again"),
    ])


def _run(docs, existing=None, process=None, sink=None, be=None):
    docs = sa.select_docs(docs, 6)
    be = be or FakeBackend()
    emb = be.embed([sa.embed_text(r) for r in docs.itertuples()])
    existing = existing if existing is not None else pd.DataFrame(columns=sa.EV_COLS)
    process = np.ones(len(docs), bool) if process is None else process
    return sa.run(docs, emb, be, existing, process, SC, sink=sink), be, docs


def test_retrieval_is_point_in_time():
    docs = sa.select_docs(_docs(
        [(i, "AAA" if i % 3 else "BBB", f"2024-01-{1 + i // 2:02d} {9 + i % 2}:00", f"t{i}", "beat cut widget")
         for i in range(40)] + [(100, "AAA", "2024-01-05 10:00", "same time", "beat"),
                                (101, "AAA", "2024-01-05 10:00", "same time 2", "beat cut")]), 99)
    emb = FakeBackend().embed([sa.embed_text(r) for r in docs.itertuples()])
    ret = sa.Retriever(docs, emb)
    for i in range(len(docs)):
        got = ret.query(i, 50, 7)
        t = docs.published_at[i]
        assert all(docs.published_at[j] < t for j in got)  # never at/after the current doc
        assert all(docs.published_at[j] >= t - pd.Timedelta(days=7) for j in got)
        assert all(docs.symbol[j] == docs.symbol[i] for j in got)
        assert i not in got
    assert len(ret.query(int(docs.index[docs.doc_id == 100][0]), 3, 90)) == 3


def test_memory_only_uses_earlier_days():
    new, be, docs = _run(_corpus())
    assert len(new) == 6 and set(new.backend) == {"fake"} and set(new.prompt_version) == {sa.PROMPT_VERSION}
    assert [len(b) for b in be.calls] == [3, 2, 1]  # one batch per day, all symbols together
    day1, day2, day3 = be.calls
    for p in day1:  # even the 15:00 AAA doc must not see the 09:00 AAA doc's judgement
        assert "level 50/100" in p and sa.NO_PRIOR in p and "RECENT SENTIMENT EVENTS (oldest first):\nnone" in p
    p_aaa2 = next(p for p in day2 if "AAA guidance cut" in p)
    assert "level 50/100" not in p_aaa2 and "fake view after up news" in p_aaa2  # day-1 view arrived
    assert "fake view after down news" not in p_aaa2.split("NEW DOCUMENT")[0]  # nothing from day 2 itself
    assert "AAA beats estimates" in p_aaa2  # RAG over earlier docs
    p_aaa3 = day3[0]
    assert "fake view after down news" in p_aaa3  # day-2 cut became the most material summary
    assert "BBB" not in p_aaa3.split("NEW DOCUMENT")[0]  # other symbols never leak in


def test_parse_and_clamp():
    p = sa.parse_output('Sure! {"direction": "IMPROVING", "delta": 3, "level": 140, "novelty": "0.4", '
                        '"materiality": -2, "horizon": "months", "drivers": ["a", "b {x}", "c", "d", "e"], '
                        '"updated_summary": "' + "w " * 100 + '"} trailing {junk}', 40)
    assert p["direction"] == "improving" and p["delta"] == 1 and p["level"] == 100
    assert p["novelty"] == 0.4 and p["materiality"] == 0 and p["horizon"] == "days"
    assert p["drivers"] == ["a", "b {x}", "c", "d"] and len(p["updated_summary"].split()) == 60
    bad = sa.parse_output("I cannot answer that.", 63.0)
    assert bad == dict(direction="stable", delta=0.0, level=63.0, novelty=0.0, materiality=0.0, horizon="days",
                       drivers=[], updated_summary="parse_error")
    q = sa.parse_output('{not json} {"delta": -0.4, "level": "n/a", "drivers": "guidance cut"}', 55)
    assert q["direction"] == "deteriorating" and q["level"] == 55 and q["drivers"] == ["guidance cut"]


def test_dedupe_and_daily_cap():
    rows = [(1, "AAA", "2024-03-04 09:00", "dup", "x", "alpaca", "news", "H"),
            (2, "AAA", "2024-03-04 10:00", "dup copy", "x", "finnhub", "news", "H"),
            (3, "BBB", "2024-03-04 10:00", "dup other sym", "x", "alpaca", "news", "H"),
            (4, "AAA", "2024-03-04 11:00", "short", "a" * 10),
            (5, "AAA", "2024-03-04 12:00", "long", "a" * 900),
            (6, "AAA", "2024-03-04 13:00", "8-K", "a" * 5, "edgar", "8-K"),
            (7, "AAA", "2024-03-09 13:00", "saturday", "a")]  # weekend -> Monday 03-11
    d = sa.select_docs(_docs(rows), 3)
    assert set(d[d.symbol == "BBB"].doc_id) == {3}  # dedupe is per symbol
    assert set(d[(d.symbol == "AAA") & (d.day == pd.Timestamp("2024-03-04"))].doc_id) == {6, 5, 4}  # EDGAR, longest
    assert set(sa.select_docs(_docs(rows), 99).query("symbol == 'AAA'").doc_id) == {1, 4, 5, 6, 7}  # earliest copy wins
    assert d.loc[d.doc_id == 7, "day"].iloc[0] == pd.Timestamp("2024-03-11")
    assert d.published_at.is_monotonic_increasing


def test_resume_skips_existing_and_rebuilds_memory(tmp_path):
    full, be_full, _ = _run(_corpus())
    first, _, _ = _run(_corpus()[lambda x: x.doc_id <= 3])  # day 1 only
    sa.write_events("bundle", tmp_path, first)
    existing = sa.load_events("bundle", tmp_path)
    docs = sa.select_docs(_corpus(), 6)
    process = ~docs.doc_id.isin(existing.doc_id).values
    res, be, _ = _run(_corpus(), existing=existing, process=process)
    assert sorted(res.doc_id) == [4, 5, 6] and [len(b) for b in be.calls] == [2, 1]
    assert be.calls == be_full.calls[1:]  # identical point-in-time context as the uninterrupted run
    cols = ["doc_id", "level", "summary"]
    pd.testing.assert_frame_equal(res[cols].reset_index(drop=True),
                                  full[full.doc_id >= 4][cols].reset_index(drop=True))


def test_bundle_round_trip_and_main(tmp_path):
    _corpus().to_parquet(tmp_path / "documents.parquet", index=False)
    args = ["--backend", "fake", "--source", "bundle", "--sink", "bundle", "--bundle-dir", str(tmp_path)]
    est = sa.main(args + ["--estimate"])
    assert est["docs"] == 6 and est["input_tokens"] > 0 and est["day_batches"] == 3
    out = sa.main(args + ["--until", "2024-03-05"])
    assert sorted(out.doc_id) == [1, 2, 3, 4, 5]
    assert (tmp_path / "embeddings.npy").exists() and (tmp_path / "embeddings_ids.npy").exists()
    ev = sa.load_events("bundle", tmp_path)
    assert list(ev.columns) == sa.EV_COLS and len(ev) == 5 and str(ev.published_at.dt.tz) == "UTC"
    assert ev.doc_id.dtype == np.int64 and all(isinstance(json.loads(x), list) for x in ev.drivers)
    assert ev.level.between(0, 100).all() and ev.delta.between(-1, 1).all()
    out2 = sa.main(args)  # resume: only the new day is scored, nothing is rewritten twice
    assert sorted(out2.doc_id) == [6]
    ev2 = sa.load_events("bundle", tmp_path)
    assert len(ev2) == 6 and not ev2.duplicated(["doc_id", "symbol"]).any()
    assert len(sa.main(args)) == 0  # nothing left
    sa.write_events("bundle", tmp_path, ev2.iloc[:2].assign(summary="rewritten"))  # upsert semantics
    ev3 = sa.load_events("bundle", tmp_path)
    assert len(ev3) == 6 and (ev3.summary == "rewritten").sum() == 2
