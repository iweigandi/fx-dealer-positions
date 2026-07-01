from __future__ import annotations

import io
import time
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from fx_analysis import (
    CHART_DIR,
    DATA_DIR,
    PALETTE,
    START_DATE,
    END_DATE,
    CURRENCIES,
    monthly_fx,
    monthly_rate,
    set_custom_style,
)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class FuturesConfig:
    code: str
    yahoo_symbol: str
    spot_series: str
    invert_spot: bool


RATE_CONFIGS = {config.code: config for config in CURRENCIES}


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

ADVANCED_ECONOMIES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD"}
FIXED_3M_TAU = 0.25


def yahoo_futures_history(symbol: str) -> pd.DataFrame:
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
    volumes = result.get("indicators", {}).get("quote", [{}])[0].get("volume") or []
    out = pd.DataFrame(
        {"close": closes, "volume": volumes},
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
    )
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["close"]).sort_index()
    if out.empty:
        raise RuntimeError(f"Yahoo futures data for {symbol} returned no valid closes")
    return out




def third_wednesday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    days_until_wednesday = (2 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_until_wednesday + 14)


def next_imm_expiry(date: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(date).normalize()
    for year in [date.year, date.year + 1]:
        for month in [3, 6, 9, 12]:
            expiry = third_wednesday(year, month)
            if expiry > date:
                return expiry
    raise RuntimeError(f"Could not assign IMM expiry for {date}")


def add_futures_implied_basis_proxies(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    us_rate = monthly_rate("TB3MS").rename("us_rate")
    for ccy, df in panel.groupby("ccy", sort=False):
        rate_config = RATE_CONFIGS[ccy]
        foreign_rate = monthly_rate(rate_config.rate_3m_series, rate_config.rate_source).rename("foreign_rate")
        out = df.copy()
        out = out.merge(us_rate.rename_axis("date").reset_index(), on="date", how="left")
        out = out.merge(foreign_rate.rename_axis("date").reset_index(), on="date", how="left")
        maturity_start = pd.to_datetime(out["futures_observation_date"]).fillna(pd.to_datetime(out["date"]))
        out["assumed_expiry_date"] = maturity_start.map(next_imm_expiry)
        out["assumed_tau_years"] = (out["assumed_expiry_date"] - maturity_start).dt.days / 365.0
        out["assumed_tau_days"] = out["assumed_tau_years"] * 365.0
        out["rate_differential_us_minus_foreign"] = out["us_rate"] - out["foreign_rate"]
        out["region_group"] = np.where(out["ccy"].isin(ADVANCED_ECONOMIES), "Advanced economies", "Emerging markets")
        out["futures_implied_basis_next_imm"] = (
            out["futures_log_basis"] - out["rate_differential_us_minus_foreign"] * out["assumed_tau_years"]
        )
        out["futures_implied_basis_fixed_3m"] = (
            out["futures_log_basis"] - out["rate_differential_us_minus_foreign"] * FIXED_3M_TAU
        )
        out["futures_implied_basis_next_imm_bps"] = 10000.0 * out["futures_implied_basis_next_imm"]
        out["futures_implied_basis_fixed_3m_bps"] = 10000.0 * out["futures_implied_basis_fixed_3m"]
        out["basis_maturity_assumption_spread_bps"] = (
            out["futures_implied_basis_next_imm_bps"] - out["futures_implied_basis_fixed_3m_bps"]
        )
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "ccy"])


def build_futures_basis() -> pd.DataFrame:
    rows = []
    for config in FUTURES:
        print(f"Fetching futures basis for {config.code} ({config.yahoo_symbol})")
        history = yahoo_futures_history(config.yahoo_symbol)
        # CME FX futures are quoted in USD per foreign currency unit. The main panel uses
        # foreign currency per USD, so the futures quote is inverted before comparison.
        futures_local_per_usd = (1.0 / history["close"]).resample("ME").last().rename("futures_local_per_usd")
        futures_observation_date = history["close"].dropna().resample("ME").apply(lambda s: s.index.max()).rename("futures_observation_date")
        monthly_volume = history["volume"].resample("ME").median().rename("futures_monthly_median_volume")
        spot = monthly_fx(config.spot_series, config.invert_spot).rename("spot_local_per_usd")
        df = pd.concat([spot, futures_local_per_usd, futures_observation_date, monthly_volume], axis=1).dropna(subset=["spot_local_per_usd", "futures_local_per_usd"])
        df["futures_log_basis"] = np.log(df["futures_local_per_usd"]) - np.log(df["spot_local_per_usd"])
        df["futures_pct_basis"] = 100.0 * (df["futures_local_per_usd"] / df["spot_local_per_usd"] - 1.0)
        df["ccy"] = config.code
        df["yahoo_symbol"] = config.yahoo_symbol
        rows.append(df.reset_index(names="date"))
        time.sleep(0.25)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ccy"])


def plot_futures_implied_basis_proxy(panel: pd.DataFrame) -> None:
    set_custom_style()
    valid = panel.dropna(subset=["futures_implied_basis_next_imm_bps"])
    summary = valid.groupby("date")["futures_implied_basis_next_imm_bps"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    dates = pd.to_datetime(summary.index)
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.fill_between(dates, summary["q25"].to_numpy(), summary["q75"].to_numpy(), color=PALETTE[2], alpha=0.22, label="Interquartile range")
    ax.plot(dates, summary["median"].to_numpy(), color=PALETTE[0], label="Cross-currency median")
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Futures-Implied FX Basis Proxy")
    ax.set_ylabel("Basis points")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    note = (
        "Source: Author's calculations using Yahoo Finance continuous CME FX futures, FRED/BCB spot and short-rate data. "
        "Proxy uses next quarterly IMM expiry as assumed maturity; it is not contract-level CIP."
    )
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "futures_implied_fx_basis_proxy.png", bbox_inches="tight")
    plt.close(fig)


def plot_basis_by_group(panel: pd.DataFrame) -> None:
    set_custom_style()
    valid = panel.dropna(subset=["futures_implied_basis_next_imm_bps"])
    grouped = valid.groupby(["date", "region_group"])["futures_implied_basis_next_imm_bps"].median().unstack()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    if "Advanced economies" in grouped:
        ax.plot(pd.to_datetime(grouped.index), grouped["Advanced economies"], color=PALETTE[0], label="Advanced economies")
    if "Emerging markets" in grouped:
        ax.plot(pd.to_datetime(grouped.index), grouped["Emerging markets"], color=PALETTE[1], label="Emerging markets")
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Futures-Implied FX Basis Proxy by Group")
    ax.set_ylabel("Median basis points")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    note = "Source: Author's calculations using Yahoo Finance continuous CME FX futures, FRED/BCB spot and short-rate data."
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "futures_implied_fx_basis_by_group.png", bbox_inches="tight")
    plt.close(fig)


def plot_maturity_assumption_comparison(panel: pd.DataFrame) -> None:
    set_custom_style()
    valid = panel.dropna(subset=["futures_implied_basis_next_imm_bps", "futures_implied_basis_fixed_3m_bps"])
    comparison = valid.groupby("date")[["futures_implied_basis_next_imm_bps", "futures_implied_basis_fixed_3m_bps"]].median()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    dates = pd.to_datetime(comparison.index)
    ax.plot(dates, comparison["futures_implied_basis_next_imm_bps"], color=PALETTE[0], label="Next IMM maturity")
    ax.plot(dates, comparison["futures_implied_basis_fixed_3m_bps"], color=PALETTE[1], label="Fixed 3-month maturity")
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Maturity Assumption Comparison")
    ax.set_ylabel("Median basis points")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    note = "Source: Author's calculations. Difference between lines shows sensitivity to the assumed maturity of the continuous front futures contract."
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "futures_basis_maturity_comparison.png", bbox_inches="tight")
    plt.close(fig)


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
    panel = add_futures_implied_basis_proxies(build_futures_basis())
    panel.to_csv(DATA_DIR / "futures_implied_fx_basis.csv", index=False, float_format="%.10g")
    plot_futures_basis(panel)
    plot_futures_implied_basis_proxy(panel)
    plot_basis_by_group(panel)
    plot_maturity_assumption_comparison(panel)
    print(f"Wrote {DATA_DIR / 'futures_implied_fx_basis.csv'}")
    print(f"Wrote {CHART_DIR / 'futures_implied_fx_basis.png'}")
    print(f"Wrote {CHART_DIR / 'futures_implied_fx_basis_proxy.png'}")
    print(f"Wrote {CHART_DIR / 'futures_implied_fx_basis_by_group.png'}")
    print(f"Wrote {CHART_DIR / 'futures_basis_maturity_comparison.png'}")


if __name__ == "__main__":
    main()
