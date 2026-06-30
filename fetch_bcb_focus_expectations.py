from __future__ import annotations

from pathlib import Path
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from fx_analysis import CHART_DIR, DATA_DIR, PALETTE, fred_series, set_custom_style

BCB_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"


def fetch_bcb_focus_exchange_rate() -> pd.DataFrame:
    rows: list[dict] = []
    page_size = 1000
    skip = 0
    while True:
        params = {
            "$filter": "contains(Indicador,'mbio')",
            "$top": str(page_size),
            "$skip": str(skip),
            "$format": "json",
        }
        response = requests.get(BCB_URL, params=params, timeout=60)
        response.raise_for_status()
        page = response.json().get("value", [])
        rows.extend(page)
        if len(page) < page_size:
            break
        skip += page_size

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("BCB Focus returned no exchange-rate expectations.")
    df = df[df["Indicador"].astype(str).str.contains("mbio", case=False, na=False)].copy()
    if "baseCalculo" in df:
        df = df[pd.to_numeric(df["baseCalculo"], errors="coerce").fillna(0).eq(0)].copy()
    df["survey_date"] = pd.to_datetime(df["Data"], errors="coerce")
    df["target_year"] = pd.to_numeric(df["DataReferencia"], errors="coerce")
    for col in ["Media", "Mediana", "DesvioPadrao", "Minimo", "Maximo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["survey_date", "target_year", "Mediana"])
    df["target_year"] = df["target_year"].astype(int)
    return df.sort_values(["survey_date", "target_year"])


def add_spot(df: pd.DataFrame) -> pd.DataFrame:
    spot = fred_series("DEXBZUS", start="1999-01-01", end="2035-12-31").rename("spot_brl_per_usd")
    out = pd.merge_asof(
        df.sort_values("survey_date"),
        spot.reset_index().rename(columns={"date": "survey_date"}).sort_values("survey_date"),
        on="survey_date",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )
    out = out.rename(
        columns={
            "Media": "expected_brl_per_usd_mean",
            "Mediana": "expected_brl_per_usd_median",
            "DesvioPadrao": "expected_brl_per_usd_std",
            "Minimo": "expected_brl_per_usd_min",
            "Maximo": "expected_brl_per_usd_max",
            "numeroRespondentes": "respondents",
            "baseCalculo": "calculation_basis",
        }
    )
    out["expected_log_change_brl_per_usd"] = np.log(out["expected_brl_per_usd_median"]) - np.log(out["spot_brl_per_usd"])
    out["horizon_years_approx"] = out["target_year"] - out["survey_date"].dt.year
    keep = [
        "survey_date",
        "target_year",
        "expected_brl_per_usd_mean",
        "expected_brl_per_usd_median",
        "expected_brl_per_usd_std",
        "expected_brl_per_usd_min",
        "expected_brl_per_usd_max",
        "respondents",
        "calculation_basis",
        "spot_brl_per_usd",
        "expected_log_change_brl_per_usd",
        "horizon_years_approx",
    ]
    return out[keep].sort_values(["survey_date", "target_year"]).reset_index(drop=True)


def plot_focus(df: pd.DataFrame) -> None:
    set_custom_style()
    latest_target_each_survey = df.dropna(subset=["expected_log_change_brl_per_usd"]).sort_values(
        ["survey_date", "target_year"]
    ).groupby("survey_date").tail(1)
    monthly = latest_target_each_survey.set_index("survey_date")["expected_log_change_brl_per_usd"].resample("ME").last().dropna()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.plot(monthly.index, monthly, color=PALETTE[0], label="Survey expected change")
    ax.axhline(0, color=PALETTE[8], lw=0.8, alpha=0.7)
    ax.set_title("BCB Focus BRL/USD Survey Expectations")
    ax.set_ylabel("Expected log change in BRL per USD")
    ax.legend(loc="upper left")
    fig.text(
        0.10,
        0.025,
        "Source: Author's calculations using Banco Central do Brasil Focus expectations and FRED spot exchange-rate data.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=PALETTE[8],
        wrap=True,
    )
    plt.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    fig.savefig(CHART_DIR / "bcb_focus_brl_usd_expectations.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    focus = add_spot(fetch_bcb_focus_exchange_rate())
    focus.to_csv(DATA_DIR / "bcb_focus_brl_usd_expectations.csv", index=False, float_format="%.10g")
    plot_focus(focus)
    print(f"Wrote {DATA_DIR / 'bcb_focus_brl_usd_expectations.csv'}")
    print(f"Wrote {CHART_DIR / 'bcb_focus_brl_usd_expectations.png'}")


if __name__ == "__main__":
    main()
