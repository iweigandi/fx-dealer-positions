from __future__ import annotations

import io
import time
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from fx_analysis import CHART_DIR, DATA_DIR, PALETTE, START_DATE, END_DATE, monthly_fx, set_custom_style

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class FuturesConfig:
    code: str
    yahoo_symbol: str
    spot_series: str
    invert_spot: bool


FUTURES = [
    FuturesConfig("EUR", "6E=F", "DEXUSEU", True),
    FuturesConfig("GBP", "6B=F", "DEXUSUK", True),
    FuturesConfig("JPY", "6J=F", "DEXJPUS", False),
    FuturesConfig("CHF", "6S=F", "DEXSZUS", False),
    FuturesConfig("CAD", "6C=F", "DEXCAUS", False),
    FuturesConfig("AUD", "6A=F", "DEXUSAL", True),
    FuturesConfig("NZD", "6N=F", "DEXUSNZ", True),
    FuturesConfig("MXN", "6M=F", "DEXMXUS", False),
    FuturesConfig("BRL", "6L=F", "DEXBZUS", False),
    FuturesConfig("ZAR", "6Z=F", "DEXSFUS", False),
]


def yahoo_futures_close(symbol: str) -> pd.Series:
    period1 = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = YAHOO_CHART_URL.format(symbol=requests.utils.quote(symbol, safe=""))
    response = requests.get(
        url,
        params={"period1": period1, "period2": period2, "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"No Yahoo futures data for {symbol}: {error}")
    timestamps = result.get("timestamp") or []
    closes = result.get("indicators", {}).get("quote", [{}])[0].get("close") or []
    out = pd.Series(closes, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None), name=symbol)
    out = pd.to_numeric(out, errors="coerce").dropna().sort_index()
    if out.empty:
        raise RuntimeError(f"Yahoo futures data for {symbol} returned no valid closes")
    return out


def build_futures_basis() -> pd.DataFrame:
    rows = []
    for config in FUTURES:
        print(f"Fetching futures basis for {config.code} ({config.yahoo_symbol})")
        close = yahoo_futures_close(config.yahoo_symbol)
        # CME FX futures are quoted in USD per foreign currency unit. The main panel uses
        # foreign currency per USD, so the futures quote is inverted before comparison.
        futures_local_per_usd = (1.0 / close).resample("ME").last().rename("futures_local_per_usd")
        spot = monthly_fx(config.spot_series, config.invert_spot).rename("spot_local_per_usd")
        df = pd.concat([spot, futures_local_per_usd], axis=1).dropna()
        df["futures_log_basis"] = np.log(df["futures_local_per_usd"]) - np.log(df["spot_local_per_usd"])
        df["futures_pct_basis"] = 100.0 * (df["futures_local_per_usd"] / df["spot_local_per_usd"] - 1.0)
        df["ccy"] = config.code
        df["yahoo_symbol"] = config.yahoo_symbol
        rows.append(df.reset_index(names="date"))
        time.sleep(0.25)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ccy"])


def plot_futures_basis(panel: pd.DataFrame) -> None:
    set_custom_style()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    average = panel.groupby("date")["futures_pct_basis"].mean()
    ax.plot(average.index, average, color=PALETTE[0], label="Cross-currency average")
    for i, (ccy, df) in enumerate(panel.groupby("ccy")):
        ax.plot(pd.to_datetime(df["date"]), df["futures_pct_basis"], color=PALETTE[(i + 2) % len(PALETTE)], alpha=0.22, lw=0.8)
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("FX Front-Futures Basis")
    ax.set_ylabel("Futures vs. spot, percent")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    note = (
        "Source: Author's calculations using Yahoo Finance continuous CME FX futures and FRED spot exchange rates. "
        "The series is a front-futures basis, not a constant-maturity forward premium."
    )
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "futures_implied_fx_basis.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    panel = build_futures_basis()
    panel.to_csv(DATA_DIR / "futures_implied_fx_basis.csv", index=False, float_format="%.10g")
    plot_futures_basis(panel)
    print(f"Wrote {DATA_DIR / 'futures_implied_fx_basis.csv'}")
    print(f"Wrote {CHART_DIR / 'futures_implied_fx_basis.png'}")


if __name__ == "__main__":
    main()
