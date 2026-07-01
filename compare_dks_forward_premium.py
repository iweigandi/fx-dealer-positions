from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fx_analysis import CHART_DIR, DATA_DIR, PALETTE, set_custom_style

DKS_CACHE = Path(".cache") / "dks_cip_dataset_v4.csv"
PANEL_CURRENCIES = ["AUD", "BRL", "CAD", "CHF", "EUR", "GBP", "JPY", "MXN", "NZD", "ZAR"]
ADVANCED_ECONOMIES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD"}


def load_dks_forward_premium() -> pd.DataFrame:
    if not DKS_CACHE.exists():
        raise FileNotFoundError(f"Missing {DKS_CACHE}. Download cip_dataset_v4.csv from the DKS public data page first.")
    pieces = []
    for chunk in pd.read_csv(DKS_CACHE, usecols=["currency", "tenor", "date", "rho"], chunksize=200000):
        chunk = chunk[(chunk["tenor"].eq("3m")) & (chunk["currency"].isin(PANEL_CURRENCIES))].copy()
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        raise RuntimeError("No DKS 3m rho observations found for panel currencies.")
    out = pd.concat(pieces, ignore_index=True)
    out["raw_date"] = pd.to_datetime(out["date"], format="%d%b%Y", errors="coerce")
    out["dks_forward_premium_3m_pct_ann"] = pd.to_numeric(out["rho"], errors="coerce")
    out = out.dropna(subset=["raw_date", "dks_forward_premium_3m_pct_ann"])
    out = out.rename(columns={"currency": "ccy"})
    out["date"] = out["raw_date"].dt.to_period("M").dt.to_timestamp("M")
    out = out.sort_values(["ccy", "date", "raw_date"]).groupby(["date", "ccy"], as_index=False).tail(1)
    return out[["date", "ccy", "dks_forward_premium_3m_pct_ann"]]


def build_comparison() -> pd.DataFrame:
    futures = pd.read_csv(DATA_DIR / "futures_implied_fx_basis.csv", parse_dates=["date"])
    # DKS rho is an annualized percentage-point forward premium. The closest futures analogue
    # annualizes log(F/S) by the assumed maturity of the front futures contract.
    futures["futures_premium_next_imm_pct_ann"] = 100.0 * futures["futures_log_basis"] / futures["assumed_tau_years"]
    futures["futures_premium_fixed_3m_pct_ann"] = 100.0 * futures["futures_log_basis"] / 0.25
    futures["assumed_tau_days"] = futures["assumed_tau_years"] * 365.0
    futures["near_3m_front_contract"] = futures["assumed_tau_days"].between(60, 95)
    futures["futures_premium_near_3m_pct_ann"] = futures["futures_premium_next_imm_pct_ann"].where(futures["near_3m_front_contract"])
    dks = load_dks_forward_premium()
    out = futures.merge(dks, on=["date", "ccy"], how="inner")
    out["region_group"] = np.where(out["ccy"].isin(ADVANCED_ECONOMIES), "Advanced economies", "Emerging markets")
    out["premium_gap_next_imm_pct_ann"] = out["futures_premium_next_imm_pct_ann"] - out["dks_forward_premium_3m_pct_ann"]
    out["premium_gap_fixed_3m_pct_ann"] = out["futures_premium_fixed_3m_pct_ann"] - out["dks_forward_premium_3m_pct_ann"]
    out["premium_gap_near_3m_pct_ann"] = out["futures_premium_near_3m_pct_ann"] - out["dks_forward_premium_3m_pct_ann"]
    return out.sort_values(["date", "ccy"])


def plot_comparison(df: pd.DataFrame) -> None:
    set_custom_style()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    near = df[df["near_3m_front_contract"]].copy()
    monthly = near.groupby("date")[["futures_premium_near_3m_pct_ann", "dks_forward_premium_3m_pct_ann"]].median()
    ax.plot(monthly.index, monthly["dks_forward_premium_3m_pct_ann"], color=PALETTE[0], label="DKS 3m forward premium")
    ax.plot(monthly.index, monthly["futures_premium_near_3m_pct_ann"], color=PALETTE[1], label="Yahoo futures proxy, near-3m front")
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Near-3M Forward Premium Comparison")
    ax.set_ylabel("Median annualized premium, percent")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    note = (
        "Source: Author's calculations using Du-Keerati-Schreger 3-month rho and Yahoo Finance continuous CME FX futures. "
        "The futures proxy uses only front-futures observations with 60-95 days to assumed IMM expiry and annualizes log(F/S) by actual days to expiry."
    )
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "dks_forward_premium_comparison.png", bbox_inches="tight")
    plt.close(fig)




def plot_group_comparison(df: pd.DataFrame) -> None:
    set_custom_style()
    near = df[df["near_3m_front_contract"]].copy()
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.2), dpi=300, sharex=True)
    for ax, group, color in zip(axes, ["Advanced economies", "Emerging markets"], [PALETTE[0], PALETTE[3]]):
        sub = near[near["region_group"].eq(group)]
        monthly = sub.groupby("date")[["futures_premium_near_3m_pct_ann", "dks_forward_premium_3m_pct_ann"]].median()
        ax.plot(monthly.index, monthly["dks_forward_premium_3m_pct_ann"], color=color, label="DKS 3m rho")
        ax.plot(monthly.index, monthly["futures_premium_near_3m_pct_ann"], color=PALETTE[1], label="Near-3M futures proxy")
        ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
        ax.set_title(group, fontsize=10)
        ax.set_ylabel("Percent")
        ax.legend(loc="upper left")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Forward Premium Comparison by Currency Group", fontsize=12)
    note = "Source: Author's calculations using DKS 3-month rho and Yahoo continuous CME FX futures. Futures series uses only observations with 60-95 days to assumed IMM expiry."
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18, hspace=0.34)
    fig.savefig(CHART_DIR / "dks_forward_premium_group_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_gap(df: pd.DataFrame) -> None:
    set_custom_style()
    near = df[df["near_3m_front_contract"]].copy()
    summary = near.groupby("date")["premium_gap_near_3m_pct_ann"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    dates = pd.to_datetime(summary.index)
    ax.fill_between(dates, summary["q25"].to_numpy(), summary["q75"].to_numpy(), color=PALETTE[2], alpha=0.22, label="Interquartile range")
    ax.plot(dates, summary["median"].to_numpy(), color=PALETTE[0], label="Median gap")
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Near-3M Futures Proxy minus DKS Forward Premium")
    ax.set_ylabel("Annualized percentage-point gap")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    note = "Source: Author's calculations. The comparison uses only near-3M front-futures observations; positive values mean the futures proxy implies a higher annualized premium than DKS rho."
    fig.text(0.10, 0.025, note, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "dks_forward_premium_gap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comparison = build_comparison()
    comparison.to_csv(DATA_DIR / "dks_forward_premium_comparison.csv", index=False, float_format="%.10g")
    plot_comparison(comparison)
    plot_group_comparison(comparison)
    plot_gap(comparison)
    print(f"Wrote {DATA_DIR / 'dks_forward_premium_comparison.csv'}")
    print(f"Wrote {CHART_DIR / 'dks_forward_premium_comparison.png'}")
    print(f"Wrote {CHART_DIR / 'dks_forward_premium_group_comparison.png'}")
    print(f"Wrote {CHART_DIR / 'dks_forward_premium_gap.png'}")
    print(comparison.groupby('ccy')['date'].agg(['min','max','count']).to_string())
    print(comparison[['futures_premium_fixed_3m_pct_ann','futures_premium_near_3m_pct_ann','dks_forward_premium_3m_pct_ann','premium_gap_fixed_3m_pct_ann','premium_gap_near_3m_pct_ann']].describe().round(4).to_string())


if __name__ == "__main__":
    main()
