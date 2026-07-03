from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io
import os
import time
import warnings

import cot_reports as cot
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
import requests
import statsmodels.api as sm

START_DATE = "2010-01-01"
END_DATE = os.environ.get("FX_END_DATE", pd.Timestamp.today().date().isoformat())
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
    rate_source: str = "fred"


CURRENCIES: list[CurrencyConfig] = [
    CurrencyConfig("EUR", "DEXUSEU", True, "099741", "EUR/USD", "IR3TIB01EZM156N"),
    CurrencyConfig("GBP", "DEXUSUK", True, "096742", "GBP/USD", "IUDSOIA"),
    CurrencyConfig("JPY", "DEXJPUS", False, "097741", "JPY/USD", "IR3TIB01JPM156N"),
    CurrencyConfig("CHF", "DEXSZUS", False, "092741", "CHF/USD", "IR3TIB01CHM156N"),
    CurrencyConfig("CAD", "DEXCAUS", False, "090741", "CAD/USD", "IR3TIB01CAM156N"),
    CurrencyConfig("AUD", "DEXUSAL", True, "232741", "AUD/USD", "IR3TIB01AUM156N"),
    CurrencyConfig("NZD", "DEXUSNZ", True, "112741", "NZD/USD", "IR3TIB01NZM156N"),
    CurrencyConfig("ZAR", "DEXSFUS", False, "122741", "ZAR/USD", "IR3TIB01ZAM156N"),
    CurrencyConfig("MXN", "DEXMXUS", False, "095741", "MXN/USD", "IR3TIB01MXM156N"),
    CurrencyConfig("BRL", "DEXBZUS", False, "102741", "BRL/USD", "4389", "bcb_sgs"),
]

CONTROL_SERIES = {
    "VIXCLS": "VIX",
    "DTWEXBGS": "BBDXY",
    "TB3MS": "TB3MS",
    "GS10": "GS10",
    "BAA10YM": "BAA10YM",
}

POSITION_GROUPS = {
    "dealer": ("Dealer_Positions_Long_All", "Dealer_Positions_Short_All"),
    "asset_manager": ("Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All"),
    "leveraged_funds": ("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
    "other_reportable": ("Other_Rept_Positions_Long_All", "Other_Rept_Positions_Short_All"),
    "nonreportable": ("NonRept_Positions_Long_All", "NonRept_Positions_Short_All"),
}

MAJOR_CURRENCIES = ["EUR", "JPY", "GBP", "CAD", "AUD"]
DXY_WEIGHTS = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "CHF": 0.036}


def set_custom_style() -> list[str]:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "font.size": 8,
            "lines.linewidth": 1.3,
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
    cache_path = CACHE_DIR / f"fred_{series_id}.csv"
    cache_is_fresh = cache_path.exists() and (time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE_SECONDS)
    if cache_is_fresh:
        raw = pd.read_csv(cache_path)
    else:
        response = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": series_id},
            timeout=45,
        )
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


def bcb_sgs_series(series_id: str, start: str = START_DATE, end: str = END_DATE) -> pd.Series:
    cache_path = CACHE_DIR / f"bcb_sgs_{series_id}.csv"
    cache_is_fresh = cache_path.exists() and (time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE_SECONDS)
    if cache_is_fresh:
        raw = pd.read_csv(cache_path)
    else:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
        frames = []
        chunk_start = pd.Timestamp(start)
        final = pd.Timestamp(end)
        while chunk_start <= final:
            chunk_end = min(chunk_start + pd.DateOffset(years=9, months=11), final)
            response = requests.get(
                url,
                params={
                    "formato": "json",
                    "dataInicial": chunk_start.strftime("%d/%m/%Y"),
                    "dataFinal": chunk_end.strftime("%d/%m/%Y"),
                },
                timeout=45,
            )
            response.raise_for_status()
            frames.append(pd.DataFrame(response.json()))
            chunk_start = chunk_end + pd.DateOffset(days=1)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raw.to_csv(cache_path, index=False)
    if "data" not in raw or "valor" not in raw:
        raise ValueError(f"Unexpected BCB SGS response for {series_id}")
    out = raw.rename(columns={"data": "date", "valor": "value"})
    out["date"] = pd.to_datetime(out["date"], format="%d/%m/%Y", errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).set_index("date").sort_index()
    return out.loc[start:end, "value"].rename(f"BCB_SGS_{series_id}")


def monthly_fx(series_id: str, invert: bool) -> pd.Series:
    s = fred_series(series_id).dropna()
    s = 1.0 / s if invert else s
    return s.resample("ME").last().rename("E")


def monthly_rate(series_id: str, source: str = "fred") -> pd.Series:
    if source == "bcb_sgs":
        return (bcb_sgs_series(series_id).dropna() / 100.0).resample("ME").last()
    return (fred_series(series_id).dropna() / 100.0).resample("ME").last()


def controls() -> pd.DataFrame:
    pieces = [fred_series(series_id).resample("ME").last().rename(label) for series_id, label in CONTROL_SERIES.items()]
    out = pd.concat(pieces, axis=1)
    out["TERM"] = out["GS10"] - out["TB3MS"]
    out["dVIX"] = np.log(out["VIX"]).diff()
    out["dDOLL"] = np.log(out["BBDXY"]).diff()
    out["dTERM"] = out["TERM"].diff()
    out["dBAA10"] = out["BAA10YM"].diff()
    return out[["dVIX", "dDOLL", "TERM", "BAA10YM", "dTERM", "dBAA10"]]


def load_tff() -> pd.DataFrame:
    cache_path = CACHE_DIR / "cftc_tff_financial_futures.csv"
    cache_is_fresh = cache_path.exists() and (time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE_SECONDS)
    if cache_is_fresh:
        return pd.read_csv(cache_path, low_memory=False)
    df = cot.cot_all(cot_report_type="traders_in_financial_futures_fut", verbose=False)
    df.to_csv(cache_path, index=False)
    return df


def cftc_currency_frame(tff: pd.DataFrame, config: CurrencyConfig) -> pd.DataFrame:
    df = tff[
        (tff["CFTC_Contract_Market_Code"].astype(str) == config.cftc_code)
        & (tff["FutOnly_or_Combined"] == "FutOnly")
    ].copy()
    if df.empty:
        raise ValueError(f"No TFF observations for CFTC code {config.cftc_code}")
    date_col = "Report_Date_as_YYYY-MM-DD"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=[date_col]).set_index(date_col).sort_index().loc[START_DATE:END_DATE]


def dealer_position(tff: pd.DataFrame, config: CurrencyConfig) -> pd.DataFrame:
    df = cftc_currency_frame(tff, config)
    monthly = df[["Dealer_Positions_Long_All", "Dealer_Positions_Short_All", "Open_Interest_All"]].resample("ME").last().dropna()
    monthly["dealer_net"] = monthly["Dealer_Positions_Long_All"] - monthly["Dealer_Positions_Short_All"]
    monthly["open_interest"] = monthly["Open_Interest_All"]
    monthly["open_interest_ma12"] = monthly["open_interest"].rolling(12, min_periods=6).mean()
    monthly["fstar"] = 100.0 * monthly["dealer_net"] / monthly["open_interest_ma12"]
    monthly["dfstar"] = monthly["fstar"].diff()
    return monthly[["dealer_net", "open_interest", "open_interest_ma12", "fstar", "dfstar"]]


def build_currency_panel(config: CurrencyConfig, tff: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    fx = monthly_fx(config.fx_series, config.invert_fx)
    foreign_rate = monthly_rate(config.rate_3m_series, config.rate_source)
    us_rate = monthly_rate("IR3TIB01USM156N")
    idiff = (foreign_rate - us_rate).rename("idiff")
    pos = dealer_position(tff, config)
    df = pd.concat([fx, idiff, pos], axis=1).join(ctrl, how="left")
    df["dlogE"] = np.log(df["E"]).diff()
    df["dlogE_1y"] = np.log(df["E"]).diff(12)
    df["idiff_1y_ago"] = df["idiff"].shift(12)
    df["didiff"] = df["idiff"].diff()
    df["rx"] = df["idiff"].shift(1) / 12.0 - df["dlogE"]
    df["rx_lead"] = df["rx"].shift(-1)
    df["dlogE_lead"] = df["dlogE"].shift(-1)
    df["realized_uip_1y"] = df["idiff_1y_ago"] - df["dlogE_1y"]
    df["ccy"] = config.code
    df["label"] = config.label
    required = [
        "E", "idiff", "fstar", "dfstar", "dlogE", "didiff", "rx", "rx_lead",
        "dVIX", "dDOLL", "dTERM", "dBAA10", "dlogE_1y", "idiff_1y_ago",
    ]
    return df.dropna(subset=required)


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


def participant_decomposition(tff: pd.DataFrame, configs: list[CurrencyConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        df = cftc_currency_frame(tff, config)
        monthly = df.resample("ME").last().dropna(subset=["Open_Interest_All"])
        oi_ma12 = monthly["Open_Interest_All"].rolling(12, min_periods=6).mean()
        out = pd.DataFrame(index=monthly.index)
        out["currency"] = config.code
        out["currency_group"] = config.code if config.code in MAJOR_CURRENCIES else "Other"
        out["open_interest"] = monthly["Open_Interest_All"]
        out["open_interest_ma12"] = oi_ma12
        for group, (long_col, short_col) in POSITION_GROUPS.items():
            out[f"{group}_net"] = monthly[long_col] - monthly[short_col]
            out[f"{group}_net_scaled"] = 100.0 * out[f"{group}_net"] / oi_ma12
        rows.append(out.reset_index(names="date"))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=["dealer_net_scaled"])


def load_expectation_points(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ecb_path = DATA_DIR / "ecb_spf_eur_usd_expectations.csv"
    if ecb_path.exists() and "EUR" in set(panel["ccy"]):
        ecb = pd.read_csv(ecb_path, parse_dates=["survey_date"])
        ecb = ecb[ecb["horizon_years_approx"].round().eq(1)].dropna(subset=["expected_log_change_eur_per_usd"])
        if not ecb.empty:
            exp = ecb.sort_values(["survey_date", "target_year"]).groupby("survey_date").tail(1)
            exp = exp.set_index("survey_date")["expected_log_change_eur_per_usd"].resample("ME").last()
            eur = panel[panel["ccy"].eq("EUR")].set_index("date").join(exp.rename("expected_change"), how="left")
            value = (eur["idiff"] - eur["expected_change"]).dropna().mean()
            if np.isfinite(value):
                rows.append({"ccy": "EUR", "indicator": "Survey UIP", "value": value})
    bcb_path = DATA_DIR / "bcb_focus_brl_usd_expectations.csv"
    if bcb_path.exists() and "BRL" in set(panel["ccy"]):
        bcb = pd.read_csv(bcb_path, parse_dates=["survey_date"])
        bcb = bcb[bcb["horizon_years_approx"].round().eq(1)].dropna(subset=["expected_log_change_brl_per_usd"])
        if not bcb.empty:
            exp = bcb.sort_values(["survey_date", "target_year"]).groupby("survey_date").tail(1)
            exp = exp.set_index("survey_date")["expected_log_change_brl_per_usd"].resample("ME").last()
            brl = panel[panel["ccy"].eq("BRL")].set_index("date").join(exp.rename("expected_change"), how="left")
            value = (brl["idiff"] - brl["expected_change"]).dropna().mean()
            if np.isfinite(value):
                rows.append({"ccy": "BRL", "indicator": "Survey UIP", "value": value})
    return pd.DataFrame(rows)


def build_dxy_proxy(panel: pd.DataFrame, per_currency: dict[str, pd.DataFrame], per_results: dict[str, sm.regression.linear_model.RegressionResultsWrapper]) -> pd.DataFrame:
    available = [ccy for ccy in DXY_WEIGHTS if ccy in per_currency and ccy in per_results]
    if len(available) < 3:
        return pd.DataFrame()
    weights = pd.Series({ccy: DXY_WEIGHTS[ccy] for ccy in available}, dtype=float)
    weights = weights / weights.sum()

    actual_parts = []
    fitted_parts = []
    xcols = ["dfstar", "didiff", "dVIX", "dDOLL", "dTERM", "dBAA10"]
    for ccy in available:
        df = per_currency[ccy].copy()
        x = sm.add_constant(df[xcols], has_constant="add")
        fitted = pd.Series(per_results[ccy].predict(x.astype(float)), index=df.index, name=ccy)
        actual_parts.append(df["dlogE"].rename(ccy))
        fitted_parts.append(fitted)

    actual_change = pd.concat(actual_parts, axis=1).dropna(how="any")
    fitted_change = pd.concat(fitted_parts, axis=1).reindex(actual_change.index).dropna(how="any")
    actual_change = actual_change.reindex(fitted_change.index)
    if actual_change.empty or fitted_change.empty:
        return pd.DataFrame()

    actual_weighted = actual_change.mul(weights, axis=1).sum(axis=1)
    fitted_weighted = fitted_change.mul(weights, axis=1).sum(axis=1)
    out = pd.DataFrame(index=actual_weighted.index)
    out["actual_proxy"] = 100.0 * np.exp(actual_weighted.cumsum() - actual_weighted.cumsum().iloc[0])
    out["model_proxy"] = 100.0 * np.exp(fitted_weighted.cumsum() - fitted_weighted.cumsum().iloc[0])
    out["is_projection"] = False
    out["basket_r2"] = out["actual_proxy"].corr(out["model_proxy"]) ** 2

    return out

def signed_stackplot(ax: plt.Axes, dates: pd.Index, data: pd.DataFrame, colors: list[str], alpha: float = 0.92) -> None:
    x = pd.to_datetime(dates)
    pos_base = np.zeros(len(data))
    neg_base = np.zeros(len(data))
    for i, col in enumerate(data.columns):
        values = data[col].fillna(0.0).to_numpy(dtype=float)
        positive = np.where(values > 0, values, 0.0)
        negative = np.where(values < 0, values, 0.0)
        if np.any(positive):
            ax.fill_between(x, pos_base, pos_base + positive, color=colors[i % len(colors)], alpha=alpha, linewidth=0)
            pos_base += positive
        if np.any(negative):
            ax.fill_between(x, neg_base, neg_base + negative, color=colors[i % len(colors)], alpha=alpha, linewidth=0)
            neg_base += negative


def save_summary_chart(
    panel: pd.DataFrame,
    cross: pd.DataFrame,
    per_table: pd.DataFrame,
    decomp: pd.DataFrame,
    per_currency: dict[str, pd.DataFrame],
    per_results: dict[str, sm.regression.linear_model.RegressionResultsWrapper],
) -> None:
    palette = set_custom_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.65), dpi=300)
    ax1, ax2, ax3, ax4 = axes.ravel()

    dot_size = 30
    ax1.scatter(cross["fstar"], cross["idiff"], color=palette[0], alpha=0.88, s=dot_size, label="Rate differential")
    ax1.scatter(cross["fstar"], cross["realized_uip_1y"], color=palette[2], alpha=0.76, s=dot_size, label="Realized UIP")
    
    texts = []
    
    exp_points = load_expectation_points(panel)
    survey_label_used = False
    for _, row in exp_points.iterrows():
        if row["ccy"] in cross.index:
            ax1.scatter(
                cross.loc[row["ccy"], "fstar"],
                row["value"],
                color=palette[1],
                s=dot_size,
                label="Survey UIP" if not survey_label_used else None,
                zorder=4,
            )
            texts.append(ax1.text(cross.loc[row["ccy"], "fstar"], row["value"], row["ccy"], fontsize=5.8, color=palette[1]))
            survey_label_used = True
            
    for ccy, row in cross.iterrows():
        if not np.isnan(row.get("realized_uip_1y", np.nan)):
            texts.append(ax1.text(row["fstar"], row["realized_uip_1y"], ccy, fontsize=5.8, color=palette[2]))
        texts.append(ax1.text(row["fstar"], row["idiff"], ccy, fontsize=5.8, color=palette[0]))

    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle="-", color='gray', lw=0.4, alpha=0.6))
    
    ax1.axhline(0, color=palette[8], lw=0.6, alpha=0.65)
    ax1.axvline(0, color=palette[8], lw=0.6, alpha=0.65)
    ax1.set_title("Currency Premia and Dealer Positions")
    ax1.set_xlabel("Average dealer net position")
    ax1.set_ylabel("Decimal")
    ax1.legend(loc="upper right", ncol=1, fontsize=5.8, handlelength=1.2)

    dealer = decomp.pivot_table(index="date", columns="currency_group", values="dealer_net_scaled", aggfunc="mean").sort_index()
    ordered_cols = [col for col in MAJOR_CURRENCIES + ["Other"] if col in dealer]
    dealer = dealer[ordered_cols]
    signed_stackplot(ax2, dealer.index, dealer, palette)
    ax2.axhline(0, color=palette[8], lw=0.65, alpha=0.8)
    ax2.set_title("Dealer Net Positions by Currency")
    ax2.set_ylabel("% of 12-month avg. OI")
    ax2.xaxis.set_major_locator(mdates.YearLocator(4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [plt.Line2D([0], [0], color=palette[i % len(palette)], lw=4) for i in range(len(ordered_cols))]
    ax2.legend(handles, ordered_cols, loc="lower right", ncol=3, fontsize=5.6, handlelength=1.3)

    grouped = decomp.groupby("date").mean(numeric_only=True).sort_index()
    nondealer = pd.DataFrame(
        {
            "Asset managers": grouped["asset_manager_net_scaled"],
            "Leveraged funds": grouped["leveraged_funds_net_scaled"],
            "Others": grouped["other_reportable_net_scaled"] + grouped["nonreportable_net_scaled"],
        }
    )
    signed_stackplot(ax3, nondealer.index, nondealer, palette[1:])
    ax3.plot(pd.to_datetime(grouped.index), grouped["dealer_net_scaled"], color=palette[0], lw=1.55, label="Dealers")
    ax3.axhline(0, color=palette[8], lw=0.65, alpha=0.8)
    ax3.set_title("Dealers and Counterpart Positions")
    ax3.set_ylabel("% of 12-month avg. OI")
    ax3.xaxis.set_major_locator(mdates.YearLocator(4))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    nondealer_handles = [plt.Line2D([0], [0], color=palette[i + 1], lw=4) for i in range(nondealer.shape[1])]
    ax3.legend([plt.Line2D([0], [0], color=palette[0], lw=1.6)] + nondealer_handles, ["Dealers"] + list(nondealer.columns), loc="lower right", ncol=1, fontsize=5.3, handlelength=1.2)

    basket = build_dxy_proxy(panel, per_currency, per_results)
    if not basket.empty:
        observed = basket[~basket["is_projection"]]
        ax4.plot(observed.index, observed["actual_proxy"], color=palette[0], label="Dollar basket proxy")
        ax4.plot(basket.index, basket["model_proxy"], color=palette[1], label="Model-implied path")
        r2 = float(observed["basket_r2"].iloc[0])
        ax4.text(0.02, 0.92, f"R$^2$ = {r2:.2f}", transform=ax4.transAxes, fontsize=6.6, color=palette[8])
    ax4.set_title("US Dollar Basket Proxy and Model-Implied Path")
    ax4.set_ylabel("Index")
    ax4.xaxis.set_major_locator(mdates.YearLocator(4))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.legend(loc="lower right", ncol=1, fontsize=5.6, handlelength=1.4)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(axis="both", labelsize=6.6)
        ax.title.set_fontsize(9.0)
        ax.xaxis.label.set_fontsize(7.0)
        ax.yaxis.label.set_fontsize(7.0)
        ax.margins(x=0.02)

    fig.suptitle("FX Premia and Dealer Positions", fontsize=11.5)
    note = (
        "Source: Author's calculations using FRED, Banco Central do Brasil, ECB SPF, BCB Focus, and CFTC Traders in Financial Futures data. "
        "Net positions are long minus short positions scaled by 12-month average open interest. The dollar basket proxy uses public ICE USDX weights for available currencies and renormalizes excluding SEK. "
        "The basket model includes dealer positions, rate differentials, global risk (VIX, broad dollar), and term (10Y-3M) and credit (Baa-10Y) spreads.\n"
        "Data available at https://github.com/iweigandi/fx-dealer-positions"
    )
    fig.text(0.065, 0.035, note, ha="left", va="bottom", fontsize=5.8, color=palette[8], wrap=True)
    plt.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.15, hspace=0.47, wspace=0.18)
    ax1.margins(x=0.08, y=0.08)
    fig.savefig(CHART_DIR / "fx_dealer_positions_summary.png", bbox_inches="tight")
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
                    "rate_source": config.rate_source,
                    "rate_series": config.rate_3m_series,
                    "cftc_code": config.cftc_code,
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
                    "rate_source": config.rate_source,
                    "rate_series": config.rate_3m_series,
                    "cftc_code": config.cftc_code,
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

    cross = panel.groupby("ccy")[["idiff", "fstar", "rx", "dlogE", "realized_uip_1y"]].mean()
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

    decomp = participant_decomposition(tff, [config for config in CURRENCIES if config.code in per_currency])
    decomp.to_csv(DATA_DIR / "cftc_position_decomposition.csv", index=False, float_format="%.10g")
    save_summary_chart(panel, cross, per_table, decomp, per_currency, per_results)

    print(f"Wrote {DATA_DIR / 'panel_data.csv'}")
    print(f"Included currencies: {', '.join(sorted(per_currency))}")
    print(f"Wrote {CHART_DIR / 'fx_dealer_positions_summary.png'}")


if __name__ == "__main__":
    main()





