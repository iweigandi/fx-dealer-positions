# FX Currency Premia and Dealer Positions

This project builds a monthly panel linking currency excess returns, short-rate differentials, and dealer positioning in FX futures markets.

The empirical object is the relation between:

- the foreign-US 3-month interest-rate differential,
- currency excess returns against the US dollar,
- dealer net positions from CFTC Traders in Financial Futures reports, scaled by a 12-month average of open interest,
- global risk controls including the VIX, broad dollar index, US term spread, and Baa-Treasury spread.

The sample currently includes AUD, BRL, CAD, CHF, EUR, GBP, JPY, MXN, NZD, and ZAR. Most short-rate differentials use public FRED/OECD 3-month money-market rates. BRL uses Banco Central do Brasil SGS series 4389 as a public CDI short-rate proxy, because a comparable BRL 3-month series is not available through the same FRED/OECD source.

## Outputs

Data:

- `data/panel_data.csv`: monthly currency panel.
- `data/currency_coverage.csv`: included currencies and sample coverage.
- `data/cross_section_means.csv`: currency-level sample means.
- `data/per_currency_results.csv`: per-currency concurrent regressions.
- `data/panel_regression_summary.csv`: panel fixed-effect regression summaries.
- `data/cftc_position_decomposition.csv`: dealer, asset-manager, leveraged-fund, other-reportable, and non-reportable net positions by currency.
- `data/ecb_spf_eur_usd_expectations.csv`: supplementary ECB SPF survey expectations for EUR/USD.
- `data/bcb_focus_brl_usd_expectations.csv`: supplementary Banco Central do Brasil Focus survey expectations for BRL/USD.
- `data/futures_implied_fx_basis.csv`: front-futures basis computed from continuous CME FX futures and spot exchange rates.

Charts:

- `chart/cross_section_idiff_fstar.png`
- `chart/global_risk_factors.png`
- `chart/predictive_regression.png`
- `chart/panel_scatter.png`
- `chart/r2_by_currency.png`
- `chart/fit_*.png`
- `chart/position_decomposition.png`
- `chart/dealer_vs_leveraged_funds.png`
- `chart/ecb_spf_eur_usd_expectations.png`
- `chart/bcb_focus_brl_usd_expectations.png`
- `chart/futures_implied_fx_basis.png`

## Data Sources

- FRED public CSV endpoint for exchange rates, most short rates, VIX, dollar index, and US spread controls.
- Banco Central do Brasil SGS series 4389 for the BRL CDI short-rate proxy.
- CFTC Traders in Financial Futures reports for dealer positions, open interest, and participant-type decompositions.
- ECB Survey of Professional Forecasters for a supplementary EUR/USD survey-expectations series.
- Banco Central do Brasil Focus survey for supplementary BRL/USD exchange-rate expectations.
- Yahoo Finance historical chart data for continuous CME FX futures used to construct a front-futures basis approximation.

No FRED API key is required.

## Method

For each currency, the script constructs a monthly exchange-rate series against the US dollar and computes the foreign-US 3-month interest-rate differential. Currency excess returns are measured as the interest differential minus the monthly log exchange-rate change. Dealer positioning is measured as dealer long minus short positions, scaled by a 12-month moving average of open interest. The project also reports comparable net-position measures for asset managers, leveraged funds, other reportables, and non-reportables.

The script estimates cross-sectional, predictive, concurrent per-currency, and panel fixed-effect regressions with HAC standard errors. Supplementary scripts collect ECB SPF average USD/EUR assumptions and BCB Focus BRL/USD expectations. These public survey data are not identical to the Consensus Economics 3-month expectations used in IMF WP/25/153, but they provide transparent survey-expectations checks for two currency pairs. A separate futures-basis script computes a market-implied approximation from continuous front CME FX futures. This series is useful for comparison, but it is not a constant-maturity OTC forward premium.

## Replication

```bash
pip install -r requirements.txt
python fx_analysis.py
python fetch_ecb_spf_expectations.py
python fetch_bcb_focus_expectations.py
python fetch_futures_implied_basis.py
```

The GitHub Action is configured to run monthly and refresh data and charts.
