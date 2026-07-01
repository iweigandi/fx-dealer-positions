from __future__ import annotations

import numpy as np
import pandas as pd

from compare_dks_forward_premium import build_comparison
from fx_analysis import DATA_DIR

DAY_COUNTS = [360, 365]
EXPIRY_ADJUSTMENTS = {
    "third_wednesday": 0,
    "last_trade_monday": -2,
    "last_trade_friday": -5,
}
MATURITY_WINDOWS = [
    (45, 95),
    (50, 95),
    (55, 95),
    (60, 95),
    (65, 95),
    (70, 95),
    (75, 95),
    (60, 90),
    (60, 85),
    (60, 80),
    (70, 90),
]
VOLUME_THRESHOLDS = [0, 100, 500, 1000, 2500, 5000, 10000, 20000, 50000]
SAMPLES = {
    "all": lambda df: pd.Series(True, index=df.index),
    "advanced": lambda df: df["region_group"].eq("Advanced economies"),
    "emerging": lambda df: df["region_group"].eq("Emerging markets"),
}


def evaluate() -> pd.DataFrame:
    base = build_comparison()
    rows = []
    for day_count in DAY_COUNTS:
        for expiry_name, expiry_adjustment in EXPIRY_ADJUSTMENTS.items():
            tau_days = (base["assumed_tau_days"] + expiry_adjustment).clip(lower=1)
            tau = tau_days / day_count
            premium = 100.0 * base["futures_log_basis"] / tau
            for lower, upper in MATURITY_WINDOWS:
                maturity_mask = tau_days.between(lower, upper)
                for volume_threshold in VOLUME_THRESHOLDS:
                    volume_mask = base["futures_monthly_median_volume"].fillna(0).ge(volume_threshold)
                    mask0 = maturity_mask & volume_mask
                    for sample_name, sample_fn in SAMPLES.items():
                        mask = mask0 & sample_fn(base)
                        df = base.loc[mask].copy()
                        if df.shape[0] < 30 or df["ccy"].nunique() < 3:
                            continue
                        p = premium.loc[mask]
                        y = df["dks_forward_premium_3m_pct_ann"]
                        gap = p - y
                        rows.append(
                            {
                                "sample": sample_name,
                                "day_count": day_count,
                                "expiry_assumption": expiry_name,
                                "expiry_adjustment_days": expiry_adjustment,
                                "maturity_lower_days": lower,
                                "maturity_upper_days": upper,
                                "volume_threshold": volume_threshold,
                                "observations": int(df.shape[0]),
                                "currencies": int(df["ccy"].nunique()),
                                "first_date": df["date"].min().date().isoformat(),
                                "last_date": df["date"].max().date().isoformat(),
                                "corr": float(p.corr(y)),
                                "rmse": float(np.sqrt(np.mean(gap**2))),
                                "mean_gap": float(gap.mean()),
                                "median_gap": float(gap.median()),
                                "std_gap": float(gap.std()),
                                "abs_median_gap": float(abs(gap.median())),
                            }
                        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["sample", "rmse", "abs_median_gap", "observations"], ascending=[True, True, True, False])
    return out


def main() -> None:
    out = evaluate()
    out.to_csv(DATA_DIR / "dks_forward_premium_proxy_grid.csv", index=False, float_format="%.10g")
    print(f"Wrote {DATA_DIR / 'dks_forward_premium_proxy_grid.csv'}")
    for sample in ["all", "advanced", "emerging"]:
        print(f"\nTop parameter sets: {sample}")
        cols = [
            "sample", "day_count", "expiry_assumption", "maturity_lower_days", "maturity_upper_days",
            "volume_threshold", "observations", "currencies", "corr", "rmse", "mean_gap", "median_gap", "std_gap"
        ]
        print(out[out["sample"].eq(sample)].head(10)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
