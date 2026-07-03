from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from fx_analysis import DATA_DIR, fred_series

ECB_SPF_URL = "https://data-api.ecb.europa.eu/service/data/SPF/Q.U2.ASSU.USD..Q.AVG"


def fetch_ecb_spf_usd_assumptions() -> pd.DataFrame:
    response = requests.get(
        ECB_SPF_URL,
        params={"format": "csvdata"},
        headers={"Accept": "text/csv", "User-Agent": "Mozilla/5.0"},
        timeout=45,
    )
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df = df[df["FCT_SOURCE"].eq("AVG") & df["FCT_BREAKDOWN"].eq("USD")].copy()
    df["survey_quarter"] = pd.PeriodIndex(df["TIME_PERIOD"], freq="Q")
    df["survey_date"] = df["survey_quarter"].dt.to_timestamp(how="end").dt.normalize()
    df["target_year"] = pd.to_numeric(df["FCT_HORIZON"], errors="coerce")
    df["expected_usd_per_eur"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["survey_date", "target_year", "expected_usd_per_eur"])
    df["target_year"] = df["target_year"].astype(int)
    return df[["survey_date", "target_year", "expected_usd_per_eur", "KEY", "TITLE_COMPL"]].sort_values(
        ["survey_date", "target_year"]
    )


def add_spot_and_expected_change(spf: pd.DataFrame) -> pd.DataFrame:
    spot = fred_series("DEXUSEU", start="1999-01-01", end="2035-12-31").resample("QE").last().rename("spot_usd_per_eur")
    out = spf.merge(spot, left_on="survey_date", right_index=True, how="left")
    out["expected_eur_per_usd"] = 1.0 / out["expected_usd_per_eur"]
    out["spot_eur_per_usd"] = 1.0 / out["spot_usd_per_eur"]
    out["expected_log_change_eur_per_usd"] = np.log(out["expected_eur_per_usd"]) - np.log(out["spot_eur_per_usd"])
    out["horizon_years_approx"] = out["target_year"] - out["survey_date"].dt.year
    return out


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = add_spot_and_expected_change(fetch_ecb_spf_usd_assumptions())
    out.to_csv(DATA_DIR / "ecb_spf_eur_usd_expectations.csv", index=False, float_format="%.10g")
    print(f"Wrote {DATA_DIR / 'ecb_spf_eur_usd_expectations.csv'}")


if __name__ == "__main__":
    main()
