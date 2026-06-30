# FX Currency Premia and Dealer Positions

This project builds a monthly panel linking currency excess returns, short-rate differentials, and dealer positioning in FX futures markets.

The empirical object is the relation between:

- the foreign-US 3-month interest-rate differential,
- currency excess returns against the US dollar,
- dealer net positions from CFTC Traders in Financial Futures reports, scaled by a 12-month average of open interest,
- global risk controls including the VIX, broad dollar index, US term spread, and Baa-Treasury spread.

The sample currently includes AUD, CAD, CHF, EUR, GBP, JPY, MXN, NZD, and ZAR. RUB and BRL are excluded because the previous script used unavailable FRED 3-month rate series for those currencies.

## Outputs

Data:

- `data/panel_data.csv`: monthly currency panel.
- `data/currency_coverage.csv`: included currencies and sample coverage.
- `data/cross_section_means.csv`: currency-level sample means.
- `data/per_currency_results.csv`: per-currency concurrent regressions.
- `data/panel_regression_summary.csv`: panel fixed-effect regression summaries.

Charts:

- `chart/cross_section_idiff_fstar.png`
- `chart/global_risk_factors.png`
- `chart/predictive_regression.png`
- `chart/panel_scatter.png`
- `chart/r2_by_currency.png`
- `chart/fit_*.png`

## Data Sources

- FRED public CSV endpoint for exchange rates, interest rates, VIX, dollar index, and US spread controls.
- CFTC Traders in Financial Futures reports for dealer positions and open interest.

No FRED API key is required.

## Method

For each currency, the script constructs a monthly exchange-rate series against the US dollar and computes the foreign-US 3-month interest-rate differential. Currency excess returns are measured as the interest differential minus the monthly log exchange-rate change. Dealer positioning is measured as dealer long minus short positions, scaled by a 12-month moving average of open interest.

The script estimates cross-sectional, predictive, concurrent per-currency, and panel fixed-effect regressions with HAC standard errors.

## Replication

```bash
pip install -r requirements.txt
python fx_analysis.py
```

The GitHub Action is configured to run monthly and refresh data and charts.
