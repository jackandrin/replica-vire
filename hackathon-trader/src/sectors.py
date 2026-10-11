"""Ticker -> (company, GICS sector). Pure data so the slim web API can import it.
Demo universe: 5 large caps per GICS sector (config.yaml: tickers)."""

GICS_SECTORS = ["Information Technology", "Communication Services", "Consumer Discretionary",
                "Consumer Staples", "Health Care", "Financials", "Energy", "Industrials",
                "Materials", "Utilities", "Real Estate"]

SYMBOLS = {
    "AAPL": ("Apple", "Information Technology"), "MSFT": ("Microsoft", "Information Technology"),
    "NVDA": ("NVIDIA", "Information Technology"), "AVGO": ("Broadcom", "Information Technology"),
    "AMD": ("AMD", "Information Technology"),
    "GOOGL": ("Alphabet", "Communication Services"), "META": ("Meta", "Communication Services"),
    "NFLX": ("Netflix", "Communication Services"), "DIS": ("Disney", "Communication Services"),
    "TMUS": ("T-Mobile US", "Communication Services"),
    "AMZN": ("Amazon", "Consumer Discretionary"), "TSLA": ("Tesla", "Consumer Discretionary"),
    "HD": ("Home Depot", "Consumer Discretionary"), "MCD": ("McDonald's", "Consumer Discretionary"),
    "NKE": ("Nike", "Consumer Discretionary"),
    "WMT": ("Walmart", "Consumer Staples"), "PG": ("Procter & Gamble", "Consumer Staples"),
    "KO": ("Coca-Cola", "Consumer Staples"), "PEP": ("PepsiCo", "Consumer Staples"),
    "COST": ("Costco", "Consumer Staples"),
    "UNH": ("UnitedHealth", "Health Care"), "JNJ": ("Johnson & Johnson", "Health Care"),
    "LLY": ("Eli Lilly", "Health Care"), "PFE": ("Pfizer", "Health Care"), "MRK": ("Merck", "Health Care"),
    "JPM": ("JPMorgan Chase", "Financials"), "V": ("Visa", "Financials"), "BAC": ("Bank of America", "Financials"),
    "GS": ("Goldman Sachs", "Financials"), "MA": ("Mastercard", "Financials"),
    "XOM": ("Exxon Mobil", "Energy"), "CVX": ("Chevron", "Energy"), "COP": ("ConocoPhillips", "Energy"),
    "SLB": ("SLB", "Energy"), "EOG": ("EOG Resources", "Energy"),
    "CAT": ("Caterpillar", "Industrials"), "GE": ("GE Aerospace", "Industrials"),
    "HON": ("Honeywell", "Industrials"), "UPS": ("UPS", "Industrials"), "BA": ("Boeing", "Industrials"),
    "LIN": ("Linde", "Materials"), "SHW": ("Sherwin-Williams", "Materials"),
    "FCX": ("Freeport-McMoRan", "Materials"), "NEM": ("Newmont", "Materials"), "DOW": ("Dow", "Materials"),
    "NEE": ("NextEra Energy", "Utilities"), "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Company", "Utilities"), "D": ("Dominion Energy", "Utilities"),
    "AEP": ("American Electric Power", "Utilities"),
    "PLD": ("Prologis", "Real Estate"), "AMT": ("American Tower", "Real Estate"),
    "EQIX": ("Equinix", "Real Estate"), "O": ("Realty Income", "Real Estate"),
    "SPG": ("Simon Property Group", "Real Estate"),
}


SECTOR_ETF = {"Information Technology": "XLK", "Communication Services": "XLC", "Consumer Discretionary": "XLY",
              "Consumer Staples": "XLP", "Health Care": "XLV", "Financials": "XLF", "Energy": "XLE",
              "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"}
MARKET = "SPY"
VOL_PROXY = "VIXY"  # VIX futures ETF; Alpaca has no index data
CONTEXT_SYMBOLS = [MARKET, VOL_PROXY] + list(SECTOR_ETF.values())  # bars needed, never traded


def sector_etf_of(symbol: str) -> str:
    return SECTOR_ETF.get(sector_of(symbol), MARKET)


def sector_of(symbol: str) -> str:
    return SYMBOLS.get(symbol, (symbol, "Other"))[1]
