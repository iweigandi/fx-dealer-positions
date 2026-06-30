from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io
import time
import warnings

import cot_reports as cot
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "chart"
CACHE_DIR = ROOT / ".cache"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

PALETTE = [
    "#00466F",
    "#F38C10",
    "#3297DB",
    "#037E73",
    "#C62828",
    "#FEBD00",
    "#41B01E",
    "#E84C3D",
    "#3D3D3D",
]


@dataclass(frozen=True)
class CurrencyConfig:
    code: str
    fx_series: str
    invert_fx: bool
    cftc_code: str
    label: str
    rate_3m_series: str


CURRENCIES: list[CurrencyConfig] = [
    CurrencyConfig("EUR", "DEXUSEU", True, "099741", "EUR/USD", "IR3TIB01EZM156N"),
    CurrencyConfig("GBP", "DEXUSUK", True, "096742", "GBP/USD", "IR3TIB01GBM156N"),
    CurrencyConfig("JPY", "DEXJPUS", False, "097741", "JPY/USD", "IR3TIB01JPM156N"),
    CurrencyConfig("CHF", "DEXSZUS", False, "092741", "CHF/USD", "IR3TIB01CHM156N"),
    CurrencyConfig("CAD", "DEXCAUS", False, "090741", "CAD/USD", "IR3TIB01CAM156N"),
    CurrencyConfig("AUD", "DEXUSAL", True, "232741", "AUD/USD", "IR3TIB01AUM156N"),
    CurrencyConfig("NZD", "DEXUSNZ", True, "112741", "NZD/USD", "IR3TIB01NZM156N"),
    CurrencyConfig("ZAR", "DEXSFUS", False, "122741", "ZAR/USD", "IR3TIB01ZAM156N"),
    CurrencyConfig("MXN", "DEXMXUS", False, "095741", "MXN/USD", "IR3TIB01MXM156N"),
]

CONTROL_SERIES = {
    "VIXCLS": "VIX",
    "DTWEXBGS": "BBDXY",
    "TB3MS": "TB3MS",
    "GS10": "GS10",
    "BAA10YM": "BAA10YM",
}


def set_custom_style() -> list[str]:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 1.5,
            "figure.figsize": (6, 4.2),
            "figure.dpi": 300,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )
    return PALETTE


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def fred_series(series_id: str, start: str = START_DATE, end: str = END_DATE) -> pd.Series:
    """Fetch a FRED series through the public graph CSV endpoint, without an API key."""
    cache_path = CACHE_DIR / f"fred_{series_id}.csv"
    cache_is_fresh = cache_path.exists() and (time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE_SECONDS)
    if cache_is_fresh:
        raw = pd.read_csv(cache_path)
    else:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        response = requests.get(url, params={"id": series_id}, timeout=45)
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text))
        raw.to_csv(cache_path, index=False)
    if "observation_date" not in raw or series_id not in raw:
        raise ValueError(f"Unexpected FRED response for {series_id}")
    out = raw.rename(columns={"observation_date": "date", series_id: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"].replace(".", np.nan), errors="coerce")
    out = out.dropna(subset=["date", "value"]).set_index("date").sort_index()
    return out.loc[start:end, "value"].rename(series_id)


def monthly_fx(series_id: str, invert: bool) -> pd.Series:
    s = fred_series(series_id).dropna()
    s = 1.0 / s if invert else s
    return s.resample("ME").last().rename("E")


def monthly_rate(series_id: str) -> pd.Series:
    return (fred_series(series_id).dropna() / 100.0).resample("ME").last()


def controls() -> pd.DataFrame:
    pieces = []
    for series_id, label in CONTROL_SERIES.items():
        pieces.append(fred_series(series_id).resample("ME").last().rename(label))
    out = pd.concat(pieces, axis=1)
    out["TERM"] = out["GS10"] - out["TB3MS"]
    out["dVIX"] = out["VIX"].pct_change(fill_method=None)
    out["dDOLL"] = out["BBDXY"].pct_change(fill_method=None)
    out["dTERM"] = out["TERM"].diff()
    out["dBAA10"] = out["BAA10YM"].diff()
    return out[["dVIX", "dDOLL", "TERM", "BAA10YM", "dTERM", "dBAA10"]]


def load_tff() -> pd.DataFrame:
    cache_path = CACHE_DIR / "cftc_tff_financial_futures.csv"
    cache_is_fresh = cache_path.exists() and (time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE_SECONDS)
    if cache_is_fresh:
        df = pd.read_csv(cache_path)
    else:
        df = cot.cot_all(cot_report_type="traders_in_financial_futures_fut", verbose=False)
        df.to_csv(cache_path, index=False)
    return df


def dealer_position(tff: pd.DataFrame, cftc_code: str) -> pd.DataFrame:
    df = tff[
        (tff["CFTC_Contract_Market_Code"].astype(str) == cftc_code)
        & (tff["FutOnly_or_Combined"] == "FutOnly")
    ].copy()
    if df.empty:
        raise ValueError(f"No TFF observations for CFTC code {cftc_code}")
    date_col = "Report_Date_as_YYYY-MM-DD"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index().loc[START_DATE:END_DATE]
    df["dealer_net"] = df["Dealer_Positions_Long_All"] - df["Dealer_Positions_Short_All"]
    df["open_interest"] = df["Open_Interest_All"]
    monthly = df[["dealer_net", "open_interest"]].resample("ME").last().dropna()
    monthly["open_interest_ma12"] = monthly["open_interest"].rolling(12, min_periods=6).mean()
    monthly["fstar"] = 100.0 * monthly["dealer_net"] / monthly["open_interest_ma12"]
    monthly["dfstar"] = monthly["fstar"].diff()
    return monthly[["dealer_net", "open_interest", "open_interest_ma12", "fstar", "dfstar"]]


def build_currency_panel(config: CurrencyConfig, tff: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    fx = monthly_fx(config.fx_series, config.invert_fx)
    foreign_rate = monthly_rate(config.rate_3m_series)
    us_rate = monthly_rate("TB3MS")
    idiff = (foreign_rate - us_rate).rename("idiff")
    pos = dealer_position(tff, config.cftc_code)
    df = pd.concat([fx, idiff, pos], axis=1).join(ctrl, how="left")
    df["dlogE"] = np.log(df["E"]).diff()
    df["didiff"] = df["idiff"].diff()
    df["rx"] = df["idiff"] - df["dlogE"]
    df["rx_lead"] = df["rx"].shift(-1)
    df["dlogE_lead"] = df["dlogE"].shift(-1)
    df["ccy"] = config.code
    df["label"] = config.label
    return df.dropna(subset=["E", "idiff", "fstar", "dfstar", "dlogE", "rx", "rx_lead", "dVIX", "dDOLL", "dTERM", "dBAA10"])


def run_ols(y: pd.Series, x: pd.DataFrame, hac_lags: int = 3) -> sm.regression.linear_model.RegressionResultsWrapper:
    x = sm.add_constant(x, has_constant="add")
    return sm.OLS(y.astype(float), x.astype(float)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def fixed_effect_design(panel: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(panel["ccy"], drop_first=True, prefix="FE", dtype=float)
    return pd.concat([panel[list(cols)].astype(float), dummies], axis=1)


def regression_row(name: str, result: sm.regression.linear_model.RegressionResultsWrapper) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "model": name,
        "nobs": int(result.nobs),
        "r2": float(result.rsquared),
        "adj_r2": float(result.rsquared_adj),
    }
    for key in result.params.index:
        row[f"beta_{key}"] = float(result.params[key])
        row[f"se_{key}"] = float(result.bse[key])
        row[f"p_{key}"] = float(result.pvalues[key])
    return row


def add_note(fig: plt.Figure, text: str, y: float = 0.025) -> None:
    fig.text(0.10, y, text, ha="left", va="bottom", fontsize=6.5, color=PALETTE[8], wrap=True)


def save_cross_section(cross: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.scatter(cross["fstar"], cross["idiff"], color=PALETTE[0], alpha=0.85, s=32)
    for ccy, row in cross.iterrows():
        ax.annotate(ccy, (row["fstar"], row["idiff"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    fit = run_ols(cross["idiff"], cross[["fstar"]], hac_lags=0)
    xline = np.linspace(cross["fstar"].min(), cross["fstar"].max(), 100)
    ax.plot(xline, fit.params["const"] + fit.params["fstar"] * xline, color=PALETTE[1], lw=1.3)
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.axvline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Currency Premia and Dealer Positions")
    ax.set_xlabel("Average dealer net position (% of 12-month average open interest)")
    ax.set_ylabel("Average foreign-US 3-month rate differential")
    add_note(fig, "Source: Author's calculations using FRED and CFTC Traders in Financial Futures data.")
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "cross_section_idiff_fstar.png", bbox_inches="tight")
    plt.close(fig)


def save_panel_scatter(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.scatter(panel["dfstar"], panel["dlogE"], color=PALETTE[0], alpha=0.35, s=14)
    valid = panel[["dfstar", "dlogE"]].dropna()
    fit = run_ols(valid["dlogE"], valid[["dfstar"]])
    xline = np.linspace(valid["dfstar"].quantile(0.01), valid["dfstar"].quantile(0.99), 200)
    ax.plot(xline, fit.params["const"] + fit.params["dfstar"] * xline, color=PALETTE[1], lw=1.3)
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.axvline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Dealer Position Changes and Exchange Rates")
    ax.set_xlabel("Monthly change in dealer net position")
    ax.set_ylabel("Monthly change in log exchange rate")
    add_note(fig, "Source: Author's calculations using FRED and CFTC Traders in Financial Futures data.")
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "panel_scatter.png", bbox_inches="tight")
    plt.close(fig)


def save_predictive_scatter(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.scatter(panel["fstar"], panel["rx_lead"], color=PALETTE[2], alpha=0.35, s=14)
    valid = panel[["fstar", "rx_lead"]].dropna()
    fit = run_ols(valid["rx_lead"], valid[["fstar"]])
    xline = np.linspace(valid["fstar"].quantile(0.01), valid["fstar"].quantile(0.99), 200)
    ax.plot(xline, fit.params["const"] + fit.params["fstar"] * xline, color=PALETTE[1], lw=1.3)
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Dealer Positions and Subsequent Currency Premia")
    ax.set_xlabel("Dealer net position (% of 12-month average open interest)")
    ax.set_ylabel("Next-month currency excess return")
    add_note(fig, "Source: Author's calculations using FRED and CFTC Traders in Financial Futures data.")
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "predictive_regression.png", bbox_inches="tight")
    plt.close(fig)


def save_global_risk(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.scatter(panel["dDOLL"], panel["dfstar"], color=PALETTE[3], alpha=0.35, s=14)
    valid = panel[["dDOLL", "dfstar"]].dropna()
    fit = run_ols(valid["dfstar"], valid[["dDOLL"]])
    xline = np.linspace(valid["dDOLL"].quantile(0.01), valid["dDOLL"].quantile(0.99), 200)
    ax.plot(xline, fit.params["const"] + fit.params["dDOLL"] * xline, color=PALETTE[1], lw=1.3)
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.axvline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("Dollar Risk and Dealer Position Changes")
    ax.set_xlabel("Monthly change in broad dollar index")
    ax.set_ylabel("Monthly change in dealer net position")
    add_note(fig, "Source: Author's calculations using FRED and CFTC Traders in Financial Futures data.")
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "global_risk_factors.png", bbox_inches="tight")
    plt.close(fig)


def save_r2(per_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    per_table.sort_values("r2", ascending=False)["r2"].plot(kind="bar", ax=ax, color=PALETTE[0])
    ax.set_title("Concurrent Exchange-Rate Fit by Currency")
    ax.set_xlabel("")
    ax.set_ylabel("R-squared")
    ax.set_ylim(0, max(0.05, per_table["r2"].max() * 1.15))
    add_note(fig, "Source: Author's calculations using FRED and CFTC Traders in Financial Futures data.")
    plt.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "r2_by_currency.png", bbox_inches="tight")
    plt.close(fig)


def save_fit_charts(per_currency: dict[str, pd.DataFrame], per_results: dict[str, sm.regression.linear_model.RegressionResultsWrapper]) -> None:
    for config in CURRENCIES:
        df = per_currency.get(config.code)
        result = per_results.get(config.code)
        if df is None or result is None:
            continue
        x = sm.add_constant(df[["dfstar", "didiff", "dVIX", "dDOLL", "dTERM", "dBAA10"]], has_constant="add")
        fitted = result.predict(x.astype(float))
        base = np.log(df["E"].iloc[0])
        actual = base + df["dlogE"].cumsum()
        implied = base + fitted.cumsum()
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
        ax.plot(df.index, actual, color=PALETTE[0], label="Observed")
        ax.plot(df.index, implied, color=PALETTE[1], label="Fitted")
        ax.set_title(f"{config.label}: Observed and Fitted Log Exchange Rate")
        ax.set_ylabel("Log exchange rate")
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left")
        add_note(fig, "Source: Author's calculations using FRED and CFTC Traders in Financial Futures data.")
        plt.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.22)
        fig.savefig(CHART_DIR / f"fit_{config.code}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_custom_style()
    tff = load_tff()
    ctrl = controls()

    per_currency: dict[str, pd.DataFrame] = {}
    coverage_rows = []
    panel_parts = []
    for config in CURRENCIES:
        try:
            print(f"Building {config.code}")
            df = build_currency_panel(config, tff, ctrl)
            if df.shape[0] < 24:
                raise ValueError(f"Too few monthly observations after alignment: {df.shape[0]}")
            per_currency[config.code] = df
            panel_parts.append(df.reset_index(names="date"))
            coverage_rows.append(
                {
                    "currency": config.code,
                    "first_date": df.index.min().date().isoformat(),
                    "last_date": df.index.max().date().isoformat(),
                    "observations": int(df.shape[0]),
                    "status": "included",
                    "notes": "",
                }
            )
        except Exception as exc:
            coverage_rows.append(
                {
                    "currency": config.code,
                    "first_date": "",
                    "last_date": "",
                    "observations": 0,
                    "status": "excluded",
                    "notes": str(exc),
                }
            )
            print(f"[WARN] {config.code}: {exc}")

    if not panel_parts:
        raise RuntimeError("No currency panels were produced.")
    panel = pd.concat(panel_parts, ignore_index=True).sort_values(["date", "ccy"])
    panel.to_csv(DATA_DIR / "panel_data.csv", index=False, float_format="%.10g")
    pd.DataFrame(coverage_rows).to_csv(DATA_DIR / "currency_coverage.csv", index=False)

    cross = panel.groupby("ccy")[["idiff", "fstar", "rx", "dlogE"]].mean()
    cross.to_csv(DATA_DIR / "cross_section_means.csv", float_format="%.10g")

    per_rows = []
    per_results: dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}
    for ccy, df in per_currency.items():
        xcols = ["dfstar", "didiff", "dVIX", "dDOLL", "dTERM", "dBAA10"]
        result = run_ols(df["dlogE"], df[xcols])
        per_results[ccy] = result
        per_rows.append(
            {
                "currency": ccy,
                "nobs": int(result.nobs),
                "beta_dfstar": float(result.params.get("dfstar", np.nan)),
                "p_dfstar": float(result.pvalues.get("dfstar", np.nan)),
                "beta_didiff": float(result.params.get("didiff", np.nan)),
                "p_didiff": float(result.pvalues.get("didiff", np.nan)),
                "r2": float(result.rsquared),
            }
        )
    per_table = pd.DataFrame(per_rows).sort_values("r2", ascending=False).set_index("currency")
    per_table.to_csv(DATA_DIR / "per_currency_results.csv", float_format="%.10g")

    risk_x = fixed_effect_design(panel, ["dVIX", "dDOLL"])
    risk = run_ols(panel.loc[risk_x.index, "dfstar"], risk_x)
    pred_x = fixed_effect_design(panel, ["fstar", "idiff"])
    pred = run_ols(panel.loc[pred_x.index, "rx_lead"], pred_x)
    concurrent_x = fixed_effect_design(panel, ["dfstar", "didiff", "dVIX", "dDOLL", "dTERM", "dBAA10"])
    concurrent = run_ols(panel.loc[concurrent_x.index, "dlogE"], concurrent_x)
    pd.DataFrame(
        [
            regression_row("dealer_positions_on_global_risk", risk),
            regression_row("next_month_excess_returns", pred),
            regression_row("concurrent_exchange_rate_changes", concurrent),
        ]
    ).to_csv(DATA_DIR / "panel_regression_summary.csv", index=False, float_format="%.10g")

    save_cross_section(cross)
    save_global_risk(panel)
    save_predictive_scatter(panel)
    save_panel_scatter(panel)
    save_r2(per_table)
    save_fit_charts(per_currency, per_results)

    print(f"Wrote {DATA_DIR / 'panel_data.csv'}")
    print(f"Included currencies: {', '.join(sorted(per_currency))}")
    print(f"Wrote charts to {CHART_DIR}")


if __name__ == "__main__":
    main()
