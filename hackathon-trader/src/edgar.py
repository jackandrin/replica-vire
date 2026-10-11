"""SEC EDGAR helper: ticker->CIK, filing lists, filing text (8-K press release, 10-Q/10-K MD&A), Form 4 trades.
Fair-access policy: set SEC_USER_AGENT="Your Name you@example.com" in .env. Requests are kept under 8/s.
python -m src.edgar AAPL [--forms 8-K,10-Q] [--since 2024-01-01]"""
from __future__ import annotations

import argparse
import re
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser

import pandas as pd
import requests

from .common import env, get_logger

log = get_logger("edgar")
ET_TZ = "America/New_York"
MAX_CHARS = 8000
_MIN_INTERVAL = 1 / 8  # <= 8 requests/second (SEC limit is 10)
_lock, _last = threading.Lock(), [0.0]
_session = None


def _ua() -> str:
    ua = env("SEC_USER_AGENT", required=False)
    if not ua or "@" not in ua:
        raise RuntimeError('SEC_USER_AGENT is missing or has no email. Set it in .env, e.g. '
                           'SEC_USER_AGENT="Jane Doe jane@example.com" (required by SEC fair-access policy)')
    return ua


def get(url: str, tries: int = 5) -> requests.Response:
    """Rate-limited GET with retries on 429/5xx/connection errors. Raises HTTPError for other statuses."""
    global _session
    if _session is None:
        hdr = {"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"}  # raises before any request if unset
        _session = requests.Session()
        _session.headers.update(hdr)
    for attempt in range(tries):
        with _lock:
            wait = _last[0] + _MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            r = _session.get(url, timeout=60)
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            log.warning("GET %s failed (%s); retrying", url, e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == tries - 1:
                r.raise_for_status()
            log.warning("GET %s -> %s; backing off", url, r.status_code)
            time.sleep(min(60, 5 * 2 ** attempt))  # SEC throttles for ~10 min after abuse; back off hard
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"unreachable: {url}")


# ---------------------------------------------------------------- ticker -> CIK
@lru_cache(maxsize=1)
def ticker_cik_map() -> dict:
    """{'AAPL': 320193, ...}. Class shares use '-' (BRK-B)."""
    j = get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in j.values()}


def cik_of(ticker: str):
    return ticker_cik_map().get(ticker.upper().replace(".", "-"))


# ---------------------------------------------------------------- filing lists
def parse_acceptance(s, filing_date=None) -> pd.Timestamp:
    """EDGAR acceptanceDateTime -> UTC. The submissions API appends 'Z' but the clock is US/Eastern
    (matches the 'Accepted' time on EDGAR index pages). Reading it as Eastern is also the conservative
    choice: if it were truly UTC we would only delay availability by 4-5 h, never leak."""
    if s:
        t = pd.Timestamp(str(s).rstrip("Z").replace("T", " ")[:19])
        return t.tz_localize(ET_TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    if filing_date:  # very old filings: assume end of the filing day
        return pd.Timestamp(f"{filing_date} 17:30").tz_localize(ET_TZ).tz_convert("UTC")
    return pd.NaT


def utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _cols(block: dict) -> pd.DataFrame:
    keep = ["form", "accessionNumber", "acceptanceDateTime", "filingDate", "primaryDocument", "items"]
    df = pd.DataFrame({k: block.get(k, []) for k in keep if k in block})
    for k in keep:
        if k not in df:
            df[k] = ""
    return df[keep]


def filings(cik: int, forms=None, since=None) -> pd.DataFrame:
    """All filings for a CIK (recent block + older pages) as a DataFrame:
    form, accessionNumber, acceptanceDateTime (UTC), filingDate, primaryDocument, items, cik."""
    j = get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json").json()
    since_d = str(utc(since).date()) if since is not None else None
    parts = [_cols(j.get("filings", {}).get("recent", {}))]
    for f in j.get("filings", {}).get("files", []):
        if since_d and f.get("filingTo", "9999") < since_d:
            continue  # whole page is older than we need
        try:
            parts.append(_cols(get(f"https://data.sec.gov/submissions/{f['name']}").json()))
        except Exception as e:  # noqa: BLE001
            log.warning("CIK %s: older page %s failed: %s", cik, f.get("name"), e)
    df = pd.concat(parts, ignore_index=True)
    if forms:
        df = df[df.form.isin(list(forms))]
    df = df.copy()
    df["items"] = df["items"].fillna("").astype(str)
    df["acceptanceDateTime"] = pd.to_datetime(
        [parse_acceptance(a, d) for a, d in zip(df.acceptanceDateTime, df.filingDate)], utc=True)
    if since is not None:
        df = df[df.acceptanceDateTime >= utc(since)]
    df["cik"] = int(cik)
    return df.sort_values("acceptanceDateTime").reset_index(drop=True)


def doc_url(cik: int, accession: str, name: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{name}"


def has_item(items: str, item: str = "2.02") -> bool:
    """8-K items field looks like '2.02,9.01'. Item 2.02 = Results of Operations (earnings)."""
    return item in [t.strip() for t in re.split(r"[,\s;]+", str(items or "")) if t.strip()]


# ---------------------------------------------------------------- HTML -> text
class _Text(HTMLParser):
    SKIP = {"script", "style", "head", "title", "ix:header"}  # ix:header = hidden inline-XBRL facts
    BLOCK = {"p", "div", "br", "tr", "td", "th", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append(" ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    """HTML (or plain text) -> whitespace-collapsed plain text."""
    if not html:
        return ""
    if "<" in html:
        p = _Text()
        try:
            p.feed(html)
            p.close()
            html = "".join(p.out)
        except Exception:  # noqa: BLE001  malformed markup: crude tag strip
            html = unescape(re.sub(r"<[^>]+>", " ", html))
    else:
        html = unescape(html)
    return re.sub(r"\s+", " ", html).strip()


_SEP = r"\s*[\.\:\-–—]?\s*"
_MDA = {  # form -> (start pattern, end patterns in order of preference)
    "10-Q": (rf"item\s*2{_SEP}management\W{{0,3}}s?\s+discussion",
             [rf"item\s*3{_SEP}quantitative", rf"item\s*4{_SEP}controls", r"item\s*3\b"]),
    "10-K": (rf"item\s*7{_SEP}management\W{{0,3}}s?\s+discussion",
             [r"item\s*7a\b", rf"item\s*8{_SEP}financial\s+statements"]),
}


def extract_mda(text: str, form: str, limit: int = MAX_CHARS) -> str:
    """MD&A section of a 10-Q (Item 2 -> Item 3) or 10-K (Item 7 -> Item 7A). The table of contents also
    matches, so the longest start->end span wins. Falls back to the start of the document."""
    spec = _MDA.get(form.split("/")[0])
    best = ""
    if spec:
        start_pat, end_pats = spec
        for m in re.finditer(start_pat, text, flags=re.I):
            for ep in end_pats:
                e = re.compile(ep, re.I).search(text, m.end())
                if e:
                    if e.start() - m.start() > len(best):
                        best = text[m.start():e.start()]
                    break
    return (best.strip() if len(best) > 500 else text)[:limit]


# ---------------------------------------------------------------- filing documents
_EX991 = re.compile(r"(ex|exhibit)[\s\-_]*99[\-_\.]?0?1(?!\d)", re.I)
_EX99 = re.compile(r"(ex|exhibit)[\s\-_]*99(?!\d)", re.I)


def pick_press_release(names: list):
    """Choose the EX-99.1 press release from a filing directory listing (None if absent)."""
    docs = [n for n in names if n.lower().endswith((".htm", ".html", ".txt")) and not n.lower().endswith("-index.htm")]
    for pat in (_EX991, _EX99):
        hits = [n for n in docs if pat.search(n)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def filing_text(cik: int, accession: str, form: str, primary: str) -> tuple:
    """(text <= 8000 chars, url). 8-K: EX-99.1 press release if present else primary doc; 10-Q/10-K: MD&A."""
    url = doc_url(cik, accession, primary)
    if form.startswith("8-K"):
        try:
            idx = get(doc_url(cik, accession, "index.json")).json()
            ex = pick_press_release([i.get("name", "") for i in idx.get("directory", {}).get("item", [])])
            if ex:
                url = doc_url(cik, accession, ex)
        except Exception as e:  # noqa: BLE001
            log.warning("index.json for %s failed (%s); using primary doc", accession, e)
        return html_to_text(get(url).text)[:MAX_CHARS], url
    return extract_mda(html_to_text(get(url).text), form), url


# ---------------------------------------------------------------- Form 4
def form4_xml_path(primary: str) -> str:
    """'xslF345X05/wk-form4_123.xml' (rendered HTML) -> 'wk-form4_123.xml' (raw XML)."""
    return re.sub(r"^xsl[^/]*/", "", primary or "")


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def _flag(s) -> bool:
    return str(s or "").strip().lower() in ("1", "true", "y", "yes")


def parse_form4(xml_text) -> dict:
    """Form 4 XML -> {'issuer_cik', 'owner', 'officer', 'director', 'trades': [...]} where trades are
    open-market non-derivative purchases (P) and sales (S): {row, date, code, shares, price}."""
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", "replace")
    root = ET.fromstring(xml_text.strip().lstrip("﻿"))
    for el in root.iter():  # drop any namespace
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    def val(el, path):
        x = el.find(path)
        if x is None:
            return None
        v = x.find("value")
        return ((v.text if v is not None else x.text) or "").strip() or None

    owners = root.findall("reportingOwner")
    out = {"issuer_cik": int(_num(val(root, "issuer/issuerCik"))) if val(root, "issuer/issuerCik") else None,
           "owner": "; ".join(filter(None, (val(o, "reportingOwnerId/rptOwnerName") for o in owners))),
           "officer": any(_flag(val(o, "reportingOwnerRelationship/isOfficer")) for o in owners),
           "director": any(_flag(val(o, "reportingOwnerRelationship/isDirector")) for o in owners),
           "trades": []}
    for i, t in enumerate(root.findall("nonDerivativeTable/nonDerivativeTransaction")):
        code = val(t, "transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        out["trades"].append({"row": i, "date": val(t, "transactionDate"), "code": code,
                              "shares": _num(val(t, "transactionAmounts/transactionShares")),
                              "price": _num(val(t, "transactionAmounts/transactionPricePerShare"))})
    return out


def form4_trades(cik: int, accession: str, primary: str) -> dict:
    return parse_form4(get(doc_url(cik, accession, form4_xml_path(primary))).content)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--forms", default="8-K,10-Q,10-K,4")
    ap.add_argument("--since", default=None)
    a = ap.parse_args(argv)
    cik = cik_of(a.ticker)
    df = filings(cik, a.forms.split(","), a.since)
    print(f"{a.ticker} CIK={cik} filings={len(df)}")
    print(df.tail(15).to_string())


if __name__ == "__main__":
    main()
