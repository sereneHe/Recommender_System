# Industry EU Data

## Data Overview

This directory contains European industry production index data collected from OECD and FRED and transformed into wide tables for modeling.

All the seires ids are in the file:
`data/industry_eu/raw/download_summary.csv`

- summarizes the downloaded OECD and FRED series for each country.
- `OECD_series_id`: OECD SDMX key, e.g. `AUT.M.PRVM.IX.BTE.Y._Z._Z.N`
- `Fred_series_id`: FRED identifier, e.g. `AUTPRINTO01GPSAM`
- `Country_code`: local country code used in this repository

## Generate The Data

Run from the repository root:

```bash
cd /Users/xiaoyuhe/Recommender_Pavel
source .venv/bin/activate
python3 -m data_industry
```

Useful variants:

```bash
python3 -m data_industry --source oecd
python3 -m data_industry --source fred
python3 -m data_industry --activity BTE
python3 -m data_industry --activity _T
```

Generated files are written to:

- `data/industry_eu/processed/`

## Processed Tables

| File | Source | Coverage | Frequency | Rows | Country Columns |
| --- | --- | --- | --- | ---: | ---: |
| `industry_eu_oecd_ExcludingConstruction_9country_monthly_817x9.csv` | OECD | 9-country | Monthly | 817 | 9 |
| `industry_eu_oecd_ExcludingConstruction_9country_quarterly_273x9.csv` | OECD | 9-country | Quarterly | 273 | 9 |
| `industry_eu_oecd_ExcludingConstruction_16country_monthly_337x16.csv` | OECD | 16-country | Monthly | 337 | 16 |
| `industry_eu_oecd_ExcludingConstruction_16country_quarterly_113x16.csv` | OECD | 16-country | Quarterly | 113 | 16 |
| `industry_eu_oecd_ExcludingConstruction_allcountry_monthly_89x25.csv` | OECD | All-country | Monthly | 89 | 25 |
| `industry_eu_oecd_ExcludingConstruction_allcountry_quarterly_30x25.csv` | OECD | All-country | Quarterly | 30 | 25 |
| `industry_eu_fred_ExcludingConstruction_9country_monthly_780x9.csv` | FRED | 9-country | Monthly | 780 | 9 |
| `industry_eu_fred_ExcludingConstruction_9country_quarterly_260x9.csv` | FRED | 9-country | Quarterly | 260 | 9 |
| `industry_eu_fred_ExcludingConstruction_16country_monthly_300x16.csv` | FRED | 16-country | Monthly | 300 | 16 |
| `industry_eu_fred_ExcludingConstruction_16country_quarterly_100x16.csv` | FRED | 16-country | Quarterly | 100 | 16 |
| `industry_eu_fred_ExcludingConstruction_allcountry_monthly_206x25.csv` | FRED | All-country | Monthly | 206 | 25 |
| `industry_eu_fred_ExcludingConstruction_allcountry_quarterly_69x25.csv` | FRED | All-country | Quarterly | 69 | 25 |
