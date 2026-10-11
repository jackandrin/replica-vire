"""Offline tests for the v2 ingestion parsers (no network, no database)."""
import json
import math

import pandas as pd
import pyarrow as pa
import pytest

from src import edgar
from src.export_dataset import TABLES, build_query, to_arrow
from src.ingest_docs import content_hash, make_doc
from src.ingest_events import analyst_score, insider_events, parse_finra_line, short_volume_events

FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001214156</rptOwnerCik><rptOwnerName>COOK TIMOTHY D</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>true</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-04-01</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-04-02</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>170.50</value><footnoteId id="F1"/></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-04-03</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>1,000</value></transactionShares>
        <transactionPricePerShare><value>10</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_form4_parse_keeps_only_open_market_p_and_s():
    p = edgar.parse_form4(FORM4)
    assert p["issuer_cik"] == 320193 and p["owner"] == "COOK TIMOTHY D"
    assert p["officer"] is True and p["director"] is True
    assert [(t["row"], t["code"], t["shares"], t["price"]) for t in p["trades"]] == [
        (1, "S", 500.0, 170.5), (2, "P", 1000.0, 10.0)]


def test_form4_insider_events_signed_value_and_times():
    accepted = pd.Timestamp("2024-04-04 21:30", tz="UTC")
    ev = insider_events("AAPL", "0001214156-24-000001", accepted, edgar.parse_form4(FORM4))
    assert [e["value"] for e in ev] == [-85250.0, 10000.0]
    assert ev[0]["key"] == "0001214156-24-000001-1"
    assert ev[0]["event_time"] == pd.Timestamp("2024-04-02 20:00", tz="UTC")  # 16:00 EDT
    assert ev[0]["available_at"] == accepted
    assert json.loads(ev[1]["payload"])["code"] == "P"


def test_form4_namespace_and_missing_price():
    xml = FORM4.replace("<ownershipDocument>", '<ownershipDocument xmlns="http://example.com/x">') \
               .replace("<value>10</value>", "")
    p = edgar.parse_form4(xml.encode())
    assert len(p["trades"]) == 2 and math.isnan(p["trades"][1]["price"])
    ev = insider_events("AAPL", "acc", pd.Timestamp("2024-04-05", tz="UTC"), p)
    assert ev[1]["value"] is None  # unknown price -> NULL, not a fake number


def test_form4_xml_path():
    assert edgar.form4_xml_path("xslF345X05/wk-form4_1712345.xml") == "wk-form4_1712345.xml"
    assert edgar.form4_xml_path("xslF345X03/doc4.xml") == "doc4.xml"
    assert edgar.form4_xml_path("doc4.xml") == "doc4.xml"


def test_finra_line_parsing():
    assert parse_finra_line("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market") is None
    assert parse_finra_line("") is None and parse_finra_line("1234") is None
    x = parse_finra_line("20240102|AAPL|4000|10|10000|B,Q,N")
    assert x["symbol"] == "AAPL" and x["ratio"] == 0.4 and str(x["date"]) == "2024-01-02"
    assert math.isnan(parse_finra_line("20240102|ZZZ|0|0|0|Q")["ratio"])
    assert parse_finra_line("20240102|AAPL|1.5|0|3.0|Q")["ratio"] == 0.5  # fractional volumes


def test_finra_events_universe_and_availability():
    text = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20240105|AAPL|4000|0|10000|Q\n20240105|XYZ|1|0|2|Q\n20240105|SPY|3|0|4|Q\n")
    ev = short_volume_events(text, {"AAPL", "SPY"})
    assert [e["symbol"] for e in ev] == ["AAPL", "SPY"]
    assert ev[0]["event_time"] == pd.Timestamp("2024-01-05 21:00", tz="UTC")    # 16:00 EST
    assert ev[0]["available_at"] == pd.Timestamp("2024-01-06 13:00", tz="UTC")  # next day 08:00 EST
    assert ev[0]["key"] == "20240105" and ev[0]["value"] == 0.4


def test_html_to_text():
    html = ("<html><head><title>x</title><style>p{color:red}</style></head><body>"
            "<div style='display:none'><ix:header>HIDDEN XBRL</ix:header></div>"
            "<p>Revenue&nbsp;rose&amp;fell</p><p>Second</p><script>var a=1</script><td>A</td><td>B</td></body></html>")
    assert edgar.html_to_text(html) == "Revenue rose&fell Second A B"
    assert edgar.html_to_text("plain &amp; simple\n\n text") == "plain & simple text"
    assert edgar.html_to_text("") == ""


def test_mda_extraction_skips_table_of_contents():
    body = "We grew revenue. " * 60
    text = ("Table of Contents Item 2. Management's Discussion and Analysis 15 Item 3. Quantitative 30 "
            "Item 4. Controls 31 PART I Item 1. Financial Statements blah blah "
            f"ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION {body}"
            "Item 3. Quantitative and Qualitative Disclosures About Market Risk more stuff")
    mda = edgar.extract_mda(text, "10-Q")
    assert mda.startswith("ITEM 2. MANAGEMENT") and "Quantitative" not in mda and "We grew revenue" in mda
    k = f"Item 7. Management's Discussion 20 Item 7A. 30 Item 7. Management's Discussion {body} Item 7A. Market risk"
    assert edgar.extract_mda(k, "10-K").startswith("Item 7. Management's Discussion We grew")
    assert edgar.extract_mda("no sections here " * 1000, "10-Q") == ("no sections here " * 1000)[:8000]
    assert len(edgar.extract_mda("x" * 20000, "10-K")) == 8000


def test_8k_item_202_detection():
    assert edgar.has_item("2.02,9.01") and edgar.has_item("9.01, 2.02") and edgar.has_item("2.02")
    assert not edgar.has_item("5.02,9.01") and not edgar.has_item("") and not edgar.has_item(None)
    assert not edgar.has_item("12.02")


def test_press_release_pick():
    names = ["0000320193-24-000081-index.htm", "aapl-20240801.htm", "a8-kex991q3202406292024.htm", "R1.htm"]
    assert edgar.pick_press_release(names) == "a8-kex991q3202406292024.htm"
    assert edgar.pick_press_release(["d8k.htm", "dex991.htm", "dex9910.htm"]) == "dex991.htm"
    assert edgar.pick_press_release(["form8-k.htm", "ex99.htm"]) == "ex99.htm"
    assert edgar.pick_press_release(["form8-k.htm"]) is None


def test_acceptance_is_read_as_eastern():
    assert edgar.parse_acceptance("2024-08-01T16:30:33.000Z") == pd.Timestamp("2024-08-01 20:30:33", tz="UTC")
    assert edgar.parse_acceptance("", "2024-01-05") == pd.Timestamp("2024-01-05 22:30", tz="UTC")


def test_analyst_score():
    assert analyst_score({"strongBuy": 10, "buy": 20, "hold": 10, "sell": 5, "strongSell": 5}) == pytest.approx(0.5)
    assert analyst_score({"strongBuy": 0, "buy": 0, "hold": 4, "sell": 0, "strongSell": 0}) == 0
    assert analyst_score({"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 3}) == -2
    assert math.isnan(analyst_score({}))


def test_documents_hash_and_truncation():
    d = make_doc("AAPL", "alpaca", "news", "2024-01-02T13:00:00Z", "Apple &amp; <b>Co</b>", "x" * 9000, "u")
    assert d["title"] == "Apple & Co" and len(d["body"]) == 8000
    assert d["published_at"] == pd.Timestamp("2024-01-02 13:00", tz="UTC")
    assert content_hash("Hello  World", "Body") == content_hash("hello world", " body ")
    assert content_hash("a", "b") != content_hash("a", "c")


def test_export_arrow_schema_and_query():
    sql, params = build_query("bars_15m", "2020-01-01", None, ["AAPL"])
    assert "ts >= %s" in sql and "symbol = ANY(%s)" in sql and params == ["2020-01-01", ["AAPL"]]
    t = to_arrow([("AAPL", pd.Timestamp("2024-01-02 14:30", tz="UTC"), 1, 2, 0.5, 1.5, 100, None, 3)],
                 TABLES["bars_15m"][2])
    assert t.schema.field("ts").type == pa.timestamp("us", tz="UTC") and t.num_rows == 1
    d = to_arrow([("AAPL", pd.Timestamp("2024-01-02").date(), 1, 2, 0.5, 1.5, 100, None)], TABLES["bars_1d"][2])
    assert d.schema.field("d").type == pa.date32() and d.column("vwap").null_count == 1
