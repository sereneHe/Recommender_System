from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path('/Users/xiaoyuhe/Recommender_Pavel/data/industry_eu')
RAW_DIR = ROOT / 'raw'
PROCESSED_DIR = ROOT / 'processed'
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = {'User-Agent': 'Mozilla/5.0'}
OECD_BASE = 'https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_INDSERV'

QUARTERLY_COUNTRIES = {
    'AUT': 'Austria',
    'BEL': 'Belgium',
    'DEU': 'Germany',
    'ESP': 'Spain',
    'EST': 'Estonia',
    'FIN': 'Finland',
    'FRA': 'France',
    'GRC': 'Greece',
    'IRL': 'Ireland',
    'ITA': 'Italy',
    'LTU': 'Lithuania',
    'LUX': 'Luxembourg',
    'LVA': 'Latvia',
    'NLD': 'Netherlands',
    'PRT': 'Portugal',
    'SVK': 'Slovakia',
    'SVN': 'Slovenia',
}

MONTHLY_16_CODES = [
    'DEU', 'BEL', 'ITA', 'LUX', 'PRT', 'AUT', 'FRA', 'NLD',
    'FIN', 'GRC', 'ESP', 'IRL', 'SVK', 'SVN', 'EST', 'LTU',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fetch and build OECD industry tables.')
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--sleep', type=float, default=0.2)
    return parser.parse_args()


def oecd_url(country_code: str, freq: str) -> tuple[str, str]:
    series_id = f'{country_code}.{freq}.PRVM.IX.BTE.Y._Z._Z.N'
    url = (
        f'{OECD_BASE}/{quote(series_id, safe=".")}'
        '?dimensionAtObservation=AllDimensions&format=csvfilewithlabels'
    )
    return series_id, url


def fetch_oecd_series(country_code: str, country: str, freq: str, retries: int = 5, timeout: int = 120) -> pd.DataFrame:
    series_id, url = oecd_url(country_code, freq)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=USER_AGENT)
            with urlopen(req, timeout=timeout) as response:
                text = response.read().decode('utf-8')
            df = pd.read_csv(StringIO(text))
            if 'TIME_PERIOD' not in df.columns or 'OBS_VALUE' not in df.columns:
                raise RuntimeError(f'Unexpected OECD CSV format for {country_code}: {list(df.columns)}')
            out = df[['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'date', 'OBS_VALUE': 'value'})
            if freq == 'Q':
                out['date'] = pd.PeriodIndex(out['date'], freq='Q').to_timestamp(how='start')
            elif freq == 'M':
                out['date'] = pd.PeriodIndex(out['date'], freq='M').to_timestamp(how='start')
            else:
                out['date'] = pd.to_datetime(out['date'])
            out['value'] = pd.to_numeric(out['value'], errors='coerce')
            out = out.dropna(subset=['value']).sort_values('date').reset_index(drop=True)
            out.insert(1, 'country_code', country_code)
            out.insert(2, 'country', country)
            out.insert(3, 'series_id', series_id)
            out.insert(4, 'frequency', freq)
            out.insert(5, 'source', 'OECD')
            return out
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f'Failed to fetch OECD series for {country_code} {freq}') from last_exc


def write_quarterly_raw(retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> None:
    summaries = []
    all_frames = []

    for code, country in QUARTERLY_COUNTRIES.items():
        out_path = RAW_DIR / f'{code}_{country.replace(" ", "_")}.csv'
        df = fetch_oecd_series(code, country, 'Q', retries=retries, timeout=timeout)
        df.to_csv(out_path, index=False)
        all_frames.append(df)
        summaries.append(
            {
                'country_code': code,
                'country': country,
                'series_id': df['series_id'].iloc[0],
                'status': 'downloaded_oecd',
                'frequency': 'Q',
                'n_obs': len(df),
                'start_date': df['date'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
                'latest_value': float(df['value'].iloc[-1]),
                'file': out_path.name,
            }
        )
        time.sleep(sleep_s)

    pd.DataFrame(summaries).sort_values('country_code').to_csv(RAW_DIR / 'summary.csv', index=False)
    pd.concat(all_frames, ignore_index=True).to_csv(RAW_DIR / 'all_countries_long.csv', index=False)


def build_16country_monthly_processed(retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    frames = []
    for code in MONTHLY_16_CODES:
        country = QUARTERLY_COUNTRIES[code]
        df = fetch_oecd_series(code, country, 'M', retries=retries, timeout=timeout)
        df = df[(df['date'] >= '1998-01-01') & (df['date'] <= '2023-12-01')][['date', 'value']].rename(columns={'value': code})
        frames.append(df)
        time.sleep(sleep_s)

    wide = frames[0]
    for df in frames[1:]:
        wide = wide.merge(df, on='date', how='inner')

    wide = wide.sort_values('date').reset_index(drop=True)
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    out_path = PROCESSED_DIR / 'industry_eu_16country_monthly_1998_2023.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def load_data(
    data_path,
    frequency=None,
    target=None,
    features=None,
    start_date=None,
    end_date=None,
    impute="none",
    dropna_selected=True,
):
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if frequency is not None and "frequency" in df.columns:
        df = df[df["frequency"] == frequency].copy()

    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)].copy()
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)].copy()

    selected = []
    if target is not None:
        selected.append(target)
    if features is not None:
        selected.extend(features)
    selected = list(dict.fromkeys(selected))

    keep_cols = ["date"] + [c for c in selected if c in df.columns]
    df = df[keep_cols].sort_values("date").reset_index(drop=True)

    if impute == "interpolate_ffill_bfill":
        value_cols = [c for c in df.columns if c != "date"]
        df[value_cols] = (
            df[value_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    elif impute not in (None, "none"):
        raise ValueError(f"Unsupported impute mode: {impute}")

    if dropna_selected and selected:
        df = df.dropna(subset=[c for c in selected if c in df.columns])

    return df.set_index("date")


def main() -> None:
    args = parse_args()
    write_quarterly_raw(retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
    monthly_path = build_16country_monthly_processed(retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
    print(f'Wrote raw quarterly country tables to: {RAW_DIR}')
    print(f'Wrote processed monthly table to: {monthly_path}')


if __name__ == '__main__':
    main()
