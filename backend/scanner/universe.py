"""
A curated, liquid US-stock universe for the scanner to screen.

This is NOT "the only stocks you can analyze" (Stock Analyzer and
Watchlist accept any ticker) — it's specifically the pool the Scanner
searches over, chosen for liquidity and sector spread so a single
batched data pull stays fast and reliable. ~120 names, roughly
mirroring the largest S&P 500 / Nasdaq 100 constituents across every
major sector.
"""

SCANNER_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "ACN",
    "INTC", "IBM", "QCOM", "TXN", "NOW", "INTU", "AMAT", "MU", "PANW", "SNPS",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "EA", "WBD",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "SBUX", "TJX", "ABNB",
    # Consumer Staples
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "CL", "MO", "TGT",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI", "BLK",
    "SCHW", "C", "PGR", "CB", "PYPL",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "AMGN",
    "ISRG", "VRTX", "GILD", "CVS", "MDT",
    # Industrials
    "GE", "RTX", "CAT", "UNP", "HON", "BA", "UPS", "DE", "LMT", "ADP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Materials
    "LIN", "SHW", "FCX", "APD", "ECL",
    # Utilities
    "NEE", "SO", "DUK", "AEP",
    # Real Estate
    "PLD", "AMT", "EQIX", "SPG",
    # ETFs (broad + sector, useful in scans too)
    "SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV",
]
