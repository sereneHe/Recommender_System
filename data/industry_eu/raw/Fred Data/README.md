# FRED Data

This folder stores FRED-exported industry-related CSV files kept as local reference data.

## Current File Groups

The files currently fall into two frequency groups:

- `monthly`
  Most country-level files such as `AUTPRINTO01GPSAM.csv`
- `quarterly`
  Aggregate series such as `PRINTO01EZQ661S.csv`

## Recommended Reading Order

Use `fred_manifest.csv` first.
It summarizes:

- `file_name`
- `series_id`
- `country_or_region`
- `suffix_family`
- `inferred_frequency`
- `n_obs`
- `start_date`
- `end_date`

## Naming Notes

Observed suffix families in this folder:

- `GPSAM`
- `GYSAM`
- `Q657S`
- `Q661S`

These files are kept here as downloaded FRED series ids.
They are not renamed to OECD-style names in this folder.

## Practical Use

- Use `fred_manifest.csv` to identify monthly vs quarterly files.
- Compare these files with OECD raw data only after aligning:
  - country or region
  - frequency
  - index vs rate interpretation
- Do not compare a monthly FRED series directly with a quarterly OECD series without resampling first.
