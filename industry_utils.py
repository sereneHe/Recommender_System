from __future__ import annotations

import argparse
import subprocess
import time
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent
ROOT = REPO_ROOT / 'data' / 'industry_eu'
RAW_DIR = ROOT / 'raw'
RAW_OECD_DIR = RAW_DIR / 'OECD Data'
RAW_FRED_DIR = RAW_DIR / 'Fred Data'
PROCESSED_DIR = ROOT / 'processed'
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_OECD_DIR.mkdir(parents=True, exist_ok=True)
RAW_FRED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = {'User-Agent': 'Mozilla/5.0'}
OECD_BASE = 'https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_INDSERV'
FRED_CSV_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='

QUARTERLY_COUNTRIES = {
    'AUT': 'Austria',
    'BEL': 'Belgium',
    'CHE': 'Switzerland',
    'CZE': 'Czech_Republic',
    'DEU': 'Germany',
    'DNK': 'Denmark',
    'EA19': 'Euro_Area_19',
    'ESP': 'Spain',
    'EST': 'Estonia',
    'FIN': 'Finland',
    'FRA': 'France',
    'GBR': 'United_Kingdom',
    'GRC': 'Greece',
    'HUN': 'Hungary',
    'IRL': 'Ireland',
    'ISL': 'Iceland',
    'ITA': 'Italy',
    'LTU': 'Lithuania',
    'LUX': 'Luxembourg',
    'LVA': 'Latvia',
    'NLD': 'Netherlands',
    'NOR': 'Norway',
    'POL': 'Poland',
    'PRT': 'Portugal',
    'SVK': 'Slovakia',
    'SVN': 'Slovenia',
    'SWE': 'Sweden',
}

MONTHLY_16_CODES = [
    'DEU', 'BEL', 'ITA', 'LUX', 'PRT', 'AUT', 'FRA', 'NLD',
    'FIN', 'GRC', 'ESP', 'IRL', 'SVK', 'SVN', 'EST', 'LTU',
]

MONTHLY_9_CODES = [
    'DEU', 'AUT', 'BEL', 'PRT', 'ITA', 'LUX', 'FRA', 'NLD', 'FIN',
]
FRED_ALL_MONTHLY_CODES = [
    'AUT', 'BEL', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR',
    'GRC', 'HUN', 'IRL', 'ISL', 'ITA', 'LTU', 'LUX', 'LVA', 'NLD', 'NOR',
    'POL', 'PRT', 'SVK', 'SVN', 'SWE',
]

MONTHLY_16_START = '1998-01-01'
MONTHLY_16_END = '2023-12-01'
MONTHLY_9_START = '1958-01-01'
MONTHLY_9_END = '2023-10-01'
ACTIVITY_LABELS = {
    'BTE': 'ExcludingConstruction',
    '_T': 'total_economy',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fetch and build OECD industry tables.')
    subparsers = parser.add_subparsers(dest='command')

    parser.add_argument('--source', choices=['oecd', 'fred'], default='oecd')
    parser.add_argument('--retries', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--sleep', type=float, default=0.4)
    parser.add_argument('--activity', choices=sorted(ACTIVITY_LABELS), default='BTE')

    problem_parser = subparsers.add_parser(
        'generate-problems',
        help='Generate Hydra problem YAML files from a processed industry table.',
    )
    problem_parser.add_argument(
        '--csv',
        required=True,
        help='Processed CSV path relative to repo root or absolute path.',
    )
    problem_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where problem YAML files will be written.',
    )
    problem_parser.add_argument(
        '--name',
        default='industry_eu',
        help='Problem name stored in each YAML.',
    )
    problem_parser.add_argument(
        '--frequency',
        default='M',
        help='Problem frequency stored in each YAML.',
    )
    problem_parser.add_argument(
        '--impute',
        default='none',
        help='Problem imputation mode.',
    )
    problem_parser.add_argument(
        '--dropna-selected',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Value for dropna_selected in generated YAML files.',
    )
    return parser.parse_args()


def _activity_slug(activity: str) -> str:
    return ACTIVITY_LABELS.get(activity, activity.strip('_').lower())


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _problem_yaml_path(output_dir: Path, name: str, target: str) -> Path:
    return output_dir / f'{name}_{target.lower()}.yaml'


def generate_problem_configs(
    csv_path: str | Path,
    output_dir: str | Path,
    name: str = 'industry_eu',
    frequency: str = 'M',
    impute: str = 'none',
    dropna_selected: bool = True,
) -> list[Path]:
    csv_path = _resolve_repo_path(str(csv_path))
    output_dir = _resolve_repo_path(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, nrows=1)
    countries = [column for column in df.columns if column != 'date']
    csv_ref = str(csv_path) if csv_path.is_absolute() else str(csv_path)

    written = []
    for target in countries:
        features = [country for country in countries if country != target]
        data = {
            'name': name,
            'data_path': str(csv_path.relative_to(REPO_ROOT)) if csv_path.is_relative_to(REPO_ROOT) else csv_ref,
            'frequency': frequency,
            'impute': impute,
            'dropna_selected': dropna_selected,
            'target': target,
            'features': features,
        }
        yaml_path = _problem_yaml_path(output_dir, name, target)
        yaml_path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding='utf-8',
        )
        written.append(yaml_path)
    return written


def oecd_url(country_code: str, freq: str, activity: str = 'BTE') -> tuple[str, str]:
    series_id = f'{country_code}.{freq}.PRVM.IX.{activity}.Y._Z._Z.N'
    url = (
        f'{OECD_BASE}/{quote(series_id, safe=".")}'
        '?dimensionAtObservation=AllDimensions&format=csvfilewithlabels'
    )
    return series_id, url


def fetch_oecd_series(country_code: str, country: str, freq: str, activity: str = 'BTE', retries: int = 5, timeout: int = 120) -> pd.DataFrame:
    series_id, url = oecd_url(country_code, freq, activity=activity)
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
    raise RuntimeError(f'Failed to fetch OECD series for {country_code} {freq} {activity}') from last_exc


def _quarterly_raw_path(code: str, country: str, activity: str = 'BTE') -> Path:
    suffix = f'_{_activity_slug(activity)}'
    return RAW_OECD_DIR / f'{code}_{country.replace(" ", "_")}{suffix}.csv'


def _monthly_raw_path(code: str, country: str, activity: str = 'BTE') -> Path:
    if activity == 'BTE':
        return RAW_OECD_DIR / f'{code}_{country.replace(" ", "_")}_OECD.csv'
    suffix = f'_{_activity_slug(activity)}'
    return RAW_OECD_DIR / f'{code}_{country.replace(" ", "_")}{suffix}_OECD.csv'


def _existing_monthly_raw_path(code: str, country: str, activity: str = 'BTE') -> Path:
    preferred = _monthly_raw_path(code, country, activity=activity)
    if preferred.exists():
        return preferred
    legacy_candidates = [
        RAW_OECD_DIR / f'{code}_OECD.csv',
        RAW_DIR / f'{code}_OECD.csv',
        RAW_OECD_DIR / f'{code}_{country.replace(" ", "_")}_monthly.csv',
        RAW_DIR / f'{code}_OECD.csv',
        RAW_DIR / f'{code}_{country.replace(" ", "_")}_monthly.csv',
        RAW_OECD_DIR / f'{code}_{country.replace(" ", "_")}_{_activity_slug(activity)}_OECD.csv',
        RAW_DIR / f'{code}_{country.replace(" ", "_")}_{_activity_slug(activity)}_OECD.csv',
        RAW_OECD_DIR / f'{code}_{country.replace(" ", "_")}_{_activity_slug(activity)}_monthly.csv',
        RAW_DIR / f'{code}_{country.replace(" ", "_")}_{_activity_slug(activity)}_monthly.csv',
    ]
    for legacy in legacy_candidates:
        if legacy.exists():
            return legacy
    return preferred


def _fred_manifest() -> pd.DataFrame:
    manifest_path = RAW_FRED_DIR / 'fred_manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing FRED manifest: {manifest_path}')
    return pd.read_csv(manifest_path)


def _fred_manifest_row(code: str) -> pd.Series:
    manifest = _fred_manifest()
    matches = manifest[manifest['Fred_country_code'] == code]
    if matches.empty:
        file_matches = sorted(RAW_FRED_DIR.glob(f'{code}*.csv'))
        if not file_matches:
            raise FileNotFoundError(f'Missing FRED manifest entry for {code}')
        stem = file_matches[0].stem
        return pd.Series(
            {
                'Fred_country_code': code,
                'Fred_series_id': stem,
                'Fred_suffix_family': stem[-5:] if len(stem) >= 5 else '',
                'Fred_frequency': 'monthly',
                'Fred_file_name': file_matches[0].name,
            }
        )
    monthly_matches = matches[matches['Fred_frequency'].str.lower() == 'monthly']
    if not monthly_matches.empty:
        return monthly_matches.iloc[0]
    return matches.iloc[0]


def _fred_download_url(series_id: str) -> str:
    return f'{FRED_CSV_BASE}{quote(series_id)}'


def fetch_fred_series(code: str, country: str, series_id: str, retries: int = 5, timeout: int = 120) -> pd.DataFrame:
    url = _fred_download_url(series_id)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ['curl', '-L', '--silent', '--show-error', '--max-time', str(timeout), url],
                check=True,
                capture_output=True,
                text=True,
            )
            text = result.stdout
            df = pd.read_csv(StringIO(text))
            if 'observation_date' not in df.columns or series_id not in df.columns:
                raise RuntimeError(f'Unexpected FRED CSV format for {series_id}: {list(df.columns)}')
            out = df.rename(columns={'observation_date': 'date', series_id: 'value'})
            out['date'] = pd.to_datetime(out['date'])
            out['value'] = pd.to_numeric(out['value'], errors='coerce')
            out = out.dropna(subset=['value']).sort_values('date').reset_index(drop=True)
            out.insert(1, 'country_code', code)
            out.insert(2, 'country', country)
            out.insert(3, 'series_id', series_id)
            out.insert(4, 'frequency', 'M')
            out.insert(5, 'source', 'FRED')
            return out
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f'Failed to fetch FRED series for {code} ({series_id})') from last_exc


def _fred_raw_path(code: str) -> Path:
    matches = sorted(RAW_FRED_DIR.glob(f'{code}*.csv'))
    if not matches:
        raise FileNotFoundError(f'Missing FRED raw file for {code} in {RAW_FRED_DIR}')
    return matches[0]


def _load_fred_series(code: str, retries: int = 5, timeout: int = 120) -> pd.DataFrame:
    country = QUARTERLY_COUNTRIES[code]
    metadata = _fred_manifest_row(code)
    raw_path = RAW_FRED_DIR / metadata['Fred_file_name']
    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=['observation_date'])
        value_col = [c for c in df.columns if c != 'observation_date'][0]
        out = df.rename(columns={'observation_date': 'date', value_col: 'value'})
        out['value'] = pd.to_numeric(out['value'], errors='coerce')
        out = out.dropna(subset=['value']).sort_values('date').reset_index(drop=True)
        out.insert(1, 'country_code', code)
        out.insert(2, 'country', country)
        out.insert(3, 'series_id', value_col)
        out.insert(4, 'frequency', 'M')
        out.insert(5, 'source', 'FRED')
        return out

    out = fetch_fred_series(code, country, metadata['Fred_series_id'], retries=retries, timeout=timeout)
    raw_df = out[['date', 'value']].rename(columns={'date': 'observation_date', 'value': metadata['Fred_series_id']})
    raw_df['observation_date'] = raw_df['observation_date'].dt.strftime('%Y-%m-%d')
    raw_df.to_csv(raw_path, index=False)
    return out


def _build_monthly_processed_from_source(codes: list[str], source: str, start_date: str | None, end_date: str | None, activity: str, retries: int, timeout: int, sleep_s: float) -> pd.DataFrame:
    frames = []
    for code in codes:
        country = QUARTERLY_COUNTRIES[code]
        if source == 'oecd':
            raw_path = _existing_monthly_raw_path(code, country, activity=activity)
            if raw_path.exists():
                df = pd.read_csv(raw_path, parse_dates=['date'])
            else:
                df = fetch_oecd_series(code, country, 'M', activity=activity, retries=retries, timeout=timeout)
                df.to_csv(_monthly_raw_path(code, country, activity=activity), index=False)
        else:
            if activity != 'BTE':
                raise ValueError('FRED source currently supports only activity=BTE')
            df = _load_fred_series(code, retries=retries, timeout=timeout)
        if start_date is not None:
            df = df[df['date'] >= start_date]
        if end_date is not None:
            df = df[df['date'] <= end_date]
        df = df[['date', 'value']].rename(columns={'value': code})
        frames.append(df)
        time.sleep(sleep_s)

    wide = frames[0]
    for df in frames[1:]:
        wide = wide.merge(df, on='date', how='inner')

    value_cols = [c for c in wide.columns if c != 'date']
    wide = wide.dropna(subset=value_cols)
    return wide.sort_values('date').reset_index(drop=True)


def write_quarterly_raw(activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> None:
    summaries = []
    all_frames = []

    for code, country in QUARTERLY_COUNTRIES.items():
        out_path = _quarterly_raw_path(code, country, activity=activity)
        if out_path.exists():
            df = pd.read_csv(out_path, parse_dates=['date'])
            status = 'existing_raw'
        else:
            df = fetch_oecd_series(code, country, 'Q', activity=activity, retries=retries, timeout=timeout)
            df.to_csv(out_path, index=False)
            status = 'downloaded_oecd'
        all_frames.append(df)
        summaries.append(
            {
                'country_code': code,
                'country': country,
                'series_id': df['series_id'].iloc[0],
                'status': status,
                'frequency': 'Q',
                'n_obs': len(df),
                'start_date': df['date'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
                'latest_value': float(df['value'].iloc[-1]),
                'file': out_path.name,
                'activity': activity,
            }
        )
        time.sleep(sleep_s)

    suffix = f'_{_activity_slug(activity)}'
    pd.DataFrame(summaries).sort_values('country_code').to_csv(RAW_DIR / f'summary{suffix}.csv', index=False)
    pd.concat(all_frames, ignore_index=True).to_csv(RAW_DIR / f'all_countries_long{suffix}.csv', index=False)


def write_monthly_summary(activity: str = 'BTE') -> Path:
    summaries = []
    suffix = f'_{_activity_slug(activity)}'

    for code, country in QUARTERLY_COUNTRIES.items():
        raw_path = _existing_monthly_raw_path(code, country, activity=activity)
        if not raw_path.exists():
            continue
        df = pd.read_csv(raw_path, parse_dates=['date'])
        summaries.append(
            {
                'country_code': code,
                'country': country,
                'series_id': df['series_id'].iloc[0],
                'status': 'existing_raw',
                'frequency': 'M',
                'n_obs': len(df),
                'start_date': df['date'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
                'latest_value': float(df['value'].iloc[-1]),
                'file': raw_path.name,
                'activity': activity,
            }
        )

    out_path = RAW_DIR / f'summary_monthly{suffix}.csv'
    pd.DataFrame(summaries).sort_values('country_code').to_csv(out_path, index=False)
    return out_path


def build_16country_monthly_processed(activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    return build_16country_monthly_processed_from_source('oecd', activity=activity, retries=retries, timeout=timeout, sleep_s=sleep_s)


def build_16country_monthly_processed_from_source(source: str, activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    wide = _build_monthly_processed_from_source(MONTHLY_16_CODES, source, None, None, activity, retries, timeout, sleep_s)
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    prefix = f'industry_eu_{source}_{_activity_slug(activity)}'
    out_path = PROCESSED_DIR / f'{prefix}_16country_monthly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_9country_monthly_processed(activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    return build_9country_monthly_processed_from_source('oecd', activity=activity, retries=retries, timeout=timeout, sleep_s=sleep_s)


def build_9country_monthly_processed_from_source(source: str, activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    wide = _build_monthly_processed_from_source(MONTHLY_9_CODES, source, None, None, activity, retries, timeout, sleep_s)
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    prefix = f'industry_eu_{source}_{_activity_slug(activity)}'
    out_path = PROCESSED_DIR / f'{prefix}_9country_monthly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def _load_quarterly_wide(codes: list[str], activity: str = 'BTE') -> pd.DataFrame:
    frames = []
    for code in codes:
        country = QUARTERLY_COUNTRIES[code]
        raw_path = _quarterly_raw_path(code, country, activity=activity)
        if not raw_path.exists():
            raise FileNotFoundError(f'Missing quarterly raw file for {code}: {raw_path}')
        df = pd.read_csv(raw_path, parse_dates=['date'])
        df = df[['date', 'value']].rename(columns={'value': code})
        frames.append(df)

    wide = frames[0]
    for df in frames[1:]:
        wide = wide.merge(df, on='date', how='inner')

    return wide.sort_values('date').reset_index(drop=True)


def build_16country_quarterly_processed(activity: str = 'BTE') -> Path:
    try:
        wide = _load_quarterly_wide(MONTHLY_16_CODES, activity=activity)
    except FileNotFoundError:
        wide = _build_monthly_processed_from_source(MONTHLY_16_CODES, 'oecd', None, None, activity, retries=0, timeout=0, sleep_s=0.0)
        wide = wide.set_index('date').resample('QS').mean().dropna().reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    prefix = f'industry_eu_oecd_{_activity_slug(activity)}'
    out_path = PROCESSED_DIR / f'{prefix}_16country_quarterly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_16country_quarterly_processed_from_fred(activity: str = 'BTE') -> Path:
    if activity != 'BTE':
        raise ValueError('FRED source currently supports only activity=BTE')
    wide = _build_monthly_processed_from_source(MONTHLY_16_CODES, 'fred', None, None, activity, retries=0, timeout=0, sleep_s=0.0)
    wide = wide.set_index('date').resample('QS').mean().dropna().reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    out_path = PROCESSED_DIR / f'industry_eu_fred_{_activity_slug(activity)}_16country_quarterly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_9country_quarterly_processed(activity: str = 'BTE') -> Path:
    try:
        wide = _load_quarterly_wide(MONTHLY_9_CODES, activity=activity)
    except FileNotFoundError:
        wide = _build_monthly_processed_from_source(MONTHLY_9_CODES, 'oecd', None, None, activity, retries=0, timeout=0, sleep_s=0.0)
        wide = wide.set_index('date').resample('QS').mean().dropna().reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    prefix = f'industry_eu_oecd_{_activity_slug(activity)}'
    out_path = PROCESSED_DIR / f'{prefix}_9country_quarterly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_9country_quarterly_processed_from_fred(activity: str = 'BTE') -> Path:
    if activity != 'BTE':
        raise ValueError('FRED source currently supports only activity=BTE')
    wide = _build_monthly_processed_from_source(MONTHLY_9_CODES, 'fred', None, None, activity, retries=0, timeout=0, sleep_s=0.0)
    wide = wide.set_index('date').resample('QS').mean().dropna().reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    out_path = PROCESSED_DIR / f'industry_eu_fred_{_activity_slug(activity)}_9country_quarterly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_allcountry_monthly_processed_from_fred(activity: str = 'BTE') -> Path:
    if activity != 'BTE':
        raise ValueError('FRED source currently supports only activity=BTE')
    wide = _build_monthly_processed_from_source(FRED_ALL_MONTHLY_CODES, 'fred', None, None, activity, retries=0, timeout=0, sleep_s=0.0)
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    out_path = PROCESSED_DIR / f'industry_eu_fred_{_activity_slug(activity)}_allcountry_monthly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_allcountry_quarterly_processed_from_fred(activity: str = 'BTE') -> Path:
    if activity != 'BTE':
        raise ValueError('FRED source currently supports only activity=BTE')
    wide = _build_monthly_processed_from_source(FRED_ALL_MONTHLY_CODES, 'fred', None, None, activity, retries=0, timeout=0, sleep_s=0.0)
    wide = wide.set_index('date').resample('QS').mean()
    value_cols = list(wide.columns)
    wide = wide.dropna(subset=value_cols).reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    out_path = PROCESSED_DIR / f'industry_eu_fred_{_activity_slug(activity)}_allcountry_quarterly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def _available_oecd_monthly_codes(activity: str = 'BTE') -> list[str]:
    codes = []
    for code, country in QUARTERLY_COUNTRIES.items():
        if _existing_monthly_raw_path(code, country, activity=activity).exists():
            codes.append(code)
    return codes


def build_allcountry_monthly_processed_from_oecd(activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    codes = _available_oecd_monthly_codes(activity=activity)
    if not codes:
        raise FileNotFoundError('No OECD monthly raw files available to build all-country table')
    wide = _build_monthly_processed_from_source(codes, 'oecd', None, None, activity, retries, timeout, sleep_s)
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    out_path = PROCESSED_DIR / f'industry_eu_oecd_{_activity_slug(activity)}_allcountry_monthly_{n_rows}x{n_cols}.csv'
    wide.to_csv(out_path, index=False)
    return out_path


def build_allcountry_quarterly_processed_from_oecd(activity: str = 'BTE', retries: int = 5, timeout: int = 120, sleep_s: float = 0.2) -> Path:
    codes = _available_oecd_monthly_codes(activity=activity)
    if not codes:
        raise FileNotFoundError('No OECD monthly raw files available to build all-country quarterly table')
    wide = _build_monthly_processed_from_source(codes, 'oecd', None, None, activity, retries, timeout, sleep_s)
    wide = wide.set_index('date').resample('QS').mean().dropna().reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1
    out_path = PROCESSED_DIR / f'industry_eu_oecd_{_activity_slug(activity)}_allcountry_quarterly_{n_rows}x{n_cols}.csv'
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
    if args.command == 'generate-problems':
        written = generate_problem_configs(
            csv_path=args.csv,
            output_dir=args.output_dir,
            name=args.name,
            frequency=args.frequency,
            impute=args.impute,
            dropna_selected=args.dropna_selected,
        )
        for path in written:
            print(path)
        return

    if args.source == 'oecd':
        write_quarterly_raw(activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        monthly_summary_path = write_monthly_summary(activity=args.activity)
        monthly_16_path = build_16country_monthly_processed_from_source('oecd', activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        monthly_9_path = build_9country_monthly_processed_from_source('oecd', activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        quarterly_16_path = build_16country_quarterly_processed(activity=args.activity)
        quarterly_9_path = build_9country_quarterly_processed(activity=args.activity)
        all_monthly_path = build_allcountry_monthly_processed_from_oecd(activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        all_quarterly_path = build_allcountry_quarterly_processed_from_oecd(activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        print(f'Wrote monthly summary table to: {monthly_summary_path}')
    else:
        monthly_16_path = build_16country_monthly_processed_from_source('fred', activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        monthly_9_path = build_9country_monthly_processed_from_source('fred', activity=args.activity, retries=args.retries, timeout=args.timeout, sleep_s=args.sleep)
        quarterly_16_path = build_16country_quarterly_processed_from_fred(activity=args.activity)
        quarterly_9_path = build_9country_quarterly_processed_from_fred(activity=args.activity)
        all_monthly_path = build_allcountry_monthly_processed_from_fred(activity=args.activity)
        all_quarterly_path = build_allcountry_quarterly_processed_from_fred(activity=args.activity)
    print(f'Source: {args.source}')
    print(f'Activity: {args.activity} ({_activity_slug(args.activity)})')
    print(f'Wrote processed 16-country monthly table to: {monthly_16_path}')
    print(f'Wrote processed 9-country monthly table to: {monthly_9_path}')
    print(f'Wrote processed 16-country quarterly table to: {quarterly_16_path}')
    print(f'Wrote processed 9-country quarterly table to: {quarterly_9_path}')
    print(f'Wrote processed all-country monthly table to: {all_monthly_path}')
    print(f'Wrote processed all-country quarterly table to: {all_quarterly_path}')


if __name__ == '__main__':
    main()
