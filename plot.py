from __future__ import annotations

import argparse
import ast
import colorsys
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageChops

ROOT = Path('/Users/xiaoyuhe/Recommender_Pavel')
REPORTS_DIR = ROOT / 'reports'
DEFAULT_DATA = ROOT / 'data/industry_eu_master/industry_eu_recommender_master.csv'
DEFAULT_GRID_PNG_OUTPUT = REPORTS_DIR / 'europe_industry_network_grid.png'
DEFAULT_GRID_9_OUTPUT = REPORTS_DIR / 'europe_industry_network_9countries_3x3.png'
DEFAULT_GRID_16_OUTPUT = REPORTS_DIR / 'europe_industry_network_16countries_4x4.png'
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 8
PNG_DPI = 300
GRID_CANVAS_WIDTH = 2880
GRID_CANVAS_HEIGHT = 3600
TARGET_COLOR = '#A84A33'
BASE_BLUE = '#5E8FD6'

COUNTRY_META = {
    'AUT': {'name': 'Austria', 'iso3': 'AUT', 'lon': 14.3, 'lat': 47.6},
    'BEL': {'name': 'Belgium', 'iso3': 'BEL', 'lon': 4.7, 'lat': 50.7},
    'DEU': {'name': 'Germany', 'iso3': 'DEU', 'lon': 10.4, 'lat': 51.1},
    'ESP': {'name': 'Spain', 'iso3': 'ESP', 'lon': -3.5, 'lat': 40.2},
    'EST': {'name': 'Estonia', 'iso3': 'EST', 'lon': 25.3, 'lat': 58.7},
    'FIN': {'name': 'Finland', 'iso3': 'FIN', 'lon': 25.3, 'lat': 64.5},
    'FRA': {'name': 'France', 'iso3': 'FRA', 'lon': 2.3, 'lat': 46.5},
    'GRC': {'name': 'Greece', 'iso3': 'GRC', 'lon': 22.9, 'lat': 39.1},
    'IRL': {'name': 'Ireland', 'iso3': 'IRL', 'lon': -8.0, 'lat': 53.3},
    'ITA': {'name': 'Italy', 'iso3': 'ITA', 'lon': 12.6, 'lat': 42.8},
    'LTU': {'name': 'Lithuania', 'iso3': 'LTU', 'lon': 23.9, 'lat': 55.3},
    'LUX': {'name': 'Luxembourg', 'iso3': 'LUX', 'lon': 6.2, 'lat': 49.8},
    'NLD': {'name': 'Netherlands', 'iso3': 'NLD', 'lon': 5.4, 'lat': 52.3},
    'PRT': {'name': 'Portugal', 'iso3': 'PRT', 'lon': -8.0, 'lat': 39.6},
    'SVK': {'name': 'Slovakia', 'iso3': 'SVK', 'lon': 19.5, 'lat': 48.7},
    'SVN': {'name': 'Slovenia', 'iso3': 'SVN', 'lon': 14.9, 'lat': 46.2},
}

LABEL_OFFSETS = {
    'AUT': (0.3, 0.9),
    'BEL': (-0.8, 0.7),
    'DEU': (0.2, 1.0),
    'ESP': (-0.2, 0.9),
    'EST': (0.6, 0.9),
    'FIN': (0.8, 1.2),
    'FRA': (-0.4, 0.9),
    'GRC': (0.9, -0.4),
    'IRL': (-1.0, 0.7),
    'ITA': (0.7, 0.6),
    'LTU': (0.7, 0.8),
    'LUX': (0.8, 0.2),
    'NLD': (0.8, 0.7),
    'PRT': (-0.8, 0.5),
    'SVK': (0.9, 0.5),
    'SVN': (0.9, -0.2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot Europe industrial production overview grid with W matrix edges.')
    parser.add_argument('--data-path', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--frequency', choices=['Q', 'M'], default='Q')
    parser.add_argument('--w-matrix', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--selected-features', type=Path, default=None)
    parser.add_argument('--edge-threshold', type=float, default=0.05)
    parser.add_argument('--top-k-edges', type=int, default=20)
    parser.add_argument('--grid-output', type=Path, default=DEFAULT_GRID_PNG_OUTPUT)
    parser.add_argument('--grid-industry-targets', action='store_true')
    parser.add_argument('--multirun-dir', type=Path, default=ROOT / 'multirun/2026-03-27/20-22-28')
    parser.add_argument('--grid-9-output', type=Path, default=DEFAULT_GRID_9_OUTPUT)
    parser.add_argument('--grid-16-output', type=Path, default=DEFAULT_GRID_16_OUTPUT)
    return parser.parse_args()


def latest_values(df: pd.DataFrame, frequency: str) -> pd.Series:
    sub = df.copy()
    if 'frequency' in sub.columns:
        sub = sub[sub['frequency'] == frequency].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    values = {}
    for code in COUNTRY_META:
        if code not in sub.columns:
            continue
        s = sub[['date', code]].dropna()
        if s.empty:
            continue
        values[code] = float(s.iloc[-1][code])
    return pd.Series(values).sort_index()


def autodiscover_w_artifacts() -> tuple[Path, Path | None, Path | None]:
    candidates = []
    for w_path in ROOT.glob('mlruns/*/*/artifacts/W_est.csv'):
        art = w_path.parent
        cfg_path = art / 'config.yaml'
        if not cfg_path.exists():
            continue
        txt = cfg_path.read_text()
        if 'name: industry_eu' not in txt:
            continue
        candidates.append(w_path)
    if not candidates:
        raise FileNotFoundError('No industry_eu W_est.csv found under mlruns.')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    w_path = candidates[0]
    art = w_path.parent
    cfg_path = art / 'config.yaml'
    sel_path = art / 'selected_features.txt'
    return w_path, cfg_path if cfg_path.exists() else None, sel_path if sel_path.exists() else None


def parse_selected_features(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    txt = path.read_text().strip()
    if not txt:
        return []
    try:
        parsed = ast.literal_eval(txt)
        return [str(x) for x in parsed]
    except Exception:
        txt = txt.strip('[]')
        return [part.strip().strip("'").strip('"') for part in txt.split(',') if part.strip()]


def parse_target_from_config(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    lines = path.read_text().splitlines()
    in_problem = False
    for line in lines:
        if line.startswith('problem:'):
            in_problem = True
            continue
        if in_problem and line.startswith('experiment:'):
            break
        if in_problem and line.strip().startswith('target:'):
            return line.split(':', 1)[1].strip().strip('"').strip("'")
    return None


def infer_labels(w: np.ndarray, selected: list[str], target: str | None) -> list[str]:
    if selected and target and w.shape[0] == len(selected) + 1:
        return selected + [target]
    if w.shape[0] == len(COUNTRY_META):
        return list(COUNTRY_META.keys())
    raise ValueError(
        f'Cannot infer labels for W matrix shape {w.shape}. Provide matching selected_features/config or a 16x16 matrix.'
    )


def build_edge_table(w: np.ndarray, labels: list[str], threshold: float, top_k: int) -> pd.DataFrame:
    rows = []
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if i == j:
                continue
            weight = float(w[i, j])
            abs_weight = abs(weight)
            if abs_weight < threshold:
                continue
            rows.append({'source': src, 'target': dst, 'weight': weight, 'abs_weight': abs_weight})
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges = edges.sort_values('abs_weight', ascending=False).head(top_k).reset_index(drop=True)
    return edges


def labels_from_problem_config(cfg: dict, w: np.ndarray) -> list[str]:
    labels = [cfg['problem']['target'], *cfg['problem']['features']]
    if len(labels) != w.shape[0]:
        raise ValueError(
            f"Config labels length {len(labels)} does not match W shape {w.shape} for target {cfg['problem']['target']}"
        )
    return labels


def parse_best_features_from_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors='ignore')
    match = re.search(r"Best features Index\(\[(.*?)\]", text, re.S)
    if not match:
        return []
    inside = match.group(1)
    return [part.strip().strip("'").strip('"') for part in inside.split(',') if part.strip()]


def add_edge_traces(fig: go.Figure, edges: pd.DataFrame) -> None:
    if edges.empty:
        return
    max_w = edges['abs_weight'].max()
    for _, row in edges.iterrows():
        src_meta = COUNTRY_META.get(row['source'])
        dst_meta = COUNTRY_META.get(row['target'])
        if src_meta is None or dst_meta is None:
            continue
        color = 'rgba(72, 72, 72, 0.72)'
        width = 2.2 + 6.2 * (row['abs_weight'] / max_w)
        sx, sy = src_meta['lon'], src_meta['lat']
        tx, ty = dst_meta['lon'], dst_meta['lat']
        dx, dy = tx - sx, ty - sy
        norm = (dx ** 2 + dy ** 2) ** 0.5
        if norm == 0:
            continue
        px, py = -dy / norm, dx / norm
        curve_strength = min(1.2, 0.14 * norm)
        cx = (sx + tx) / 2 + px * curve_strength
        cy = (sy + ty) / 2 + py * curve_strength
        ts = np.linspace(0.0, 1.0, 30)
        lons = (1 - ts) ** 2 * sx + 2 * (1 - ts) * ts * cx + ts ** 2 * tx
        lats = (1 - ts) ** 2 * sy + 2 * (1 - ts) * ts * cy + ts ** 2 * ty
        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=width, color=color),
                opacity=0.92,
                hoverinfo='text',
                text=f"{row['source']} → {row['target']}<br>weight={row['weight']:.4f}",
                showlegend=False,
            )
        )
        end_dx = lons[-1] - lons[-3]
        end_dy = lats[-1] - lats[-3]
        end_norm = (end_dx ** 2 + end_dy ** 2) ** 0.5
        if end_norm == 0:
            continue
        ux, uy = end_dx / end_norm, end_dy / end_norm
        apx, apy = -uy, ux
        arrow_len = 0.52
        arrow_w = 0.16
        bx = tx - ux * arrow_len
        by = ty - uy * arrow_len
        left_x = bx + apx * arrow_w
        left_y = by + apy * arrow_w
        right_x = bx - apx * arrow_w
        right_y = by - apy * arrow_w
        fig.add_trace(
            go.Scattergeo(
                lon=[left_x, tx, right_x, left_x],
                lat=[left_y, ty, right_y, left_y],
                mode='lines',
                fill='toself',
                fillcolor='rgba(90, 90, 90, 0.88)',
                line=dict(width=0.8, color='rgba(72, 72, 72, 0.96)'),
                opacity=1.0,
                hoverinfo='skip',
                showlegend=False,
            )
        )


def add_edge_legend(fig: go.Figure) -> None:
    legend_specs = [
        ("Low flow", "rgba(72, 72, 72, 0.72)", 3.0),
        ("Medium flow", "rgba(72, 72, 72, 0.72)", 5.4),
        ("High flow", "rgba(72, 72, 72, 0.72)", 8.0),
    ]
    for name, color, width in legend_specs:
        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode='lines',
                line=dict(width=width, color=color),
                name=name,
                showlegend=True,
                hoverinfo='skip',
            )
        )


def discrete_colorscale(codes: list[str]) -> list[list[float | str]]:
    if len(codes) == 1:
        color = COUNTRY_COLORS[codes[0]]
        return [[0.0, color], [1.0, color]]

    scale = []
    n = len(codes)
    for idx, code in enumerate(codes):
        start = idx / n
        end = (idx + 1) / n
        color = COUNTRY_COLORS[code]
        scale.append([start, color])
        scale.append([end, color])
    return scale


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    return '#{:02X}{:02X}{:02X}'.format(
        *(max(0, min(255, round(channel * 255))) for channel in rgb)
    )


def shade_color(hex_color: str, intensity: float) -> str:
    r, g, b = hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Match the reference's cool blue look: lighter for lower values,
    # richer blue for higher values.
    lightness = 0.92 - 0.34 * intensity
    saturation = min(1.0, max(0.35, s * (0.95 + 0.10 * intensity)))
    shaded = colorsys.hls_to_rgb(h, lightness, saturation)
    return rgb01_to_hex(shaded)


def value_shaded_colorscale(codes: list[str], colors: list[str]) -> list[list[float | str]]:
    if len(codes) == 1:
        color = colors[0]
        return [[0.0, color], [1.0, color]]

    scale = []
    n = len(codes)
    for idx, color in enumerate(colors):
        start = idx / n
        end = (idx + 1) / n
        scale.append([start, color])
        scale.append([end, color])
    return scale


def build_plot_df(values: pd.Series, target: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    plot_df = pd.DataFrame({
        'code': list(values.index),
        'value': list(values.values),
    })
    plot_df = plot_df.sort_values('code').reset_index(drop=True)
    plot_df['name'] = plot_df['code'].map(lambda c: COUNTRY_META[c]['name'])
    plot_df['iso3'] = plot_df['code'].map(lambda c: COUNTRY_META[c]['iso3'])
    plot_df['lon'] = plot_df['code'].map(lambda c: COUNTRY_META[c]['lon'])
    plot_df['lat'] = plot_df['code'].map(lambda c: COUNTRY_META[c]['lat'])
    plot_df['country_idx'] = np.arange(len(plot_df))
    vmin, vmax = plot_df['value'].min(), plot_df['value'].max()
    size = np.full(len(plot_df), 16.0)
    intensity = (plot_df['value'] - vmin) / (vmax - vmin + 1e-9)
    plot_df['country_color'] = [shade_color(BASE_BLUE, float(level)) for level in intensity]
    if target in set(plot_df['code']):
        plot_df.loc[plot_df['code'] == target, 'country_color'] = TARGET_COLOR
    plot_df['is_target'] = plot_df['code'].eq(target)
    return plot_df, size


def build_figure(
    plot_df: pd.DataFrame,
    size: np.ndarray,
    edges: pd.DataFrame,
    title: str,
    subtitle: str,
    show_legend: bool = True,
    width: int = FIG_WIDTH_IN * PNG_DPI,
    height: int = FIG_HEIGHT_IN * PNG_DPI,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=plot_df['iso3'],
            z=plot_df['country_idx'],
            text=plot_df['name'],
            locationmode='ISO-3',
            colorscale=value_shaded_colorscale(
                plot_df['code'].tolist(),
                plot_df['country_color'].tolist(),
            ),
            zmin=0,
            zmax=max(len(plot_df) - 1, 1),
            marker_line_color='white',
            marker_line_width=0.8,
            showscale=False,
            hovertemplate='%{text}<extra></extra>',
        )
    )

    add_edge_traces(fig, edges)
    if show_legend:
        add_edge_legend(fig)

    fig.add_trace(
        go.Scattergeo(
            lon=plot_df.loc[~plot_df['is_target'], 'lon'],
            lat=plot_df.loc[~plot_df['is_target'], 'lat'],
            customdata=np.stack(
                [
                    plot_df.loc[~plot_df['is_target'], 'name'],
                    plot_df.loc[~plot_df['is_target'], 'value'],
                ],
                axis=1,
            ),
            mode='markers',
            marker=dict(
                size=size[~plot_df['is_target']],
                color='#7A7A7A',
                opacity=0.92,
                line=dict(width=1.8, color='#3F3F3F'),
            ),
            hovertemplate='%{customdata[0]}<br>industrial production=%{customdata[1]:.2f}<extra></extra>',
            name='Countries',
            showlegend=False,
        )
    )

    if plot_df['is_target'].any():
        target_df = plot_df.loc[plot_df['is_target']]
        fig.add_trace(
            go.Scattergeo(
                lon=target_df['lon'],
                lat=target_df['lat'],
                customdata=np.stack([target_df['name'], target_df['value']], axis=1),
                mode='markers',
                marker=dict(
                    size=size[plot_df['is_target']] + 2,
                    color='#7A7A7A',
                    opacity=0.96,
                    line=dict(width=2.6, color=TARGET_COLOR),
                ),
                hovertemplate='%{customdata[0]}<br>industrial production=%{customdata[1]:.2f}<extra></extra>',
                name='Target',
                showlegend=False,
            )
        )

    label_lon = []
    label_lat = []
    label_text = []
    for _, row in plot_df.iterrows():
        dx, dy = LABEL_OFFSETS.get(row['code'], (0.0, 0.8))
        label_lon.append(row['lon'] + dx)
        label_lat.append(row['lat'] + dy)
        label_text.append(row['name'])

    fig.add_trace(
        go.Scattergeo(
            lon=label_lon,
            lat=label_lat,
            text=label_text,
            mode='text',
            textfont=dict(size=24 if width > 1500 else 15, color='#111111'),
            hoverinfo='skip',
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=''),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        geo=dict(
            scope='europe',
            showcountries=True,
            countrycolor='white',
            showland=True,
            landcolor='rgb(245,245,245)',
            showocean=True,
            oceancolor='rgb(242,248,255)',
            showframe=True,
            framecolor='rgba(80,80,80,0.75)',
            framewidth=1.0,
            lataxis=dict(
                range=[34, 76],
                showgrid=True,
                gridcolor='rgba(120,120,120,0.18)',
                gridwidth=0.6,
                dtick=10,
                tick0=40,
            ),
            lonaxis=dict(
                range=[-15.75, 33],
                showgrid=False,
                dtick=10,
                tick0=0,
            ),
            domain=dict(x=[0.0, 1.0], y=[0.0, 0.95]),
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=2, b=0),
    )
    return fig


def crop_white_margins(image: Image.Image, border: int = 0) -> Image.Image:
    bg = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - border)
    top = max(0, top - border)
    right = min(image.width, right + border)
    bottom = min(image.height, bottom + border)
    return image.crop((left, top, right, bottom))


def fit_image_to_canvas(image: Image.Image, canvas_width: int, canvas_height: int) -> Image.Image:
    return ImageOps.fit(image, (canvas_width, canvas_height), method=Image.Resampling.LANCZOS)


def write_grid_overview(
    values: pd.Series,
    current_target: str | None,
    current_edges: pd.DataFrame,
    output: Path,
) -> None:
    panel_width = 900
    panel_height = 650
    cols = 4
    rows = 4
    codes = sorted(COUNTRY_META.keys())

    with tempfile.TemporaryDirectory(prefix='industry-grid-') as tmpdir:
        tmpdir_path = Path(tmpdir)
        panel_paths: list[Path] = []
        for code in codes:
            plot_df, size = build_plot_df(values, code)
            panel_edges = current_edges if code == current_target else pd.DataFrame(columns=['source', 'target', 'weight', 'abs_weight'])
            fig = build_figure(
                plot_df,
                size,
                panel_edges,
                title=COUNTRY_META[code]['name'],
                subtitle='',
                show_legend=False,
                width=panel_width,
                height=panel_height,
            )
            panel_path = tmpdir_path / f'{code}.png'
            fig.write_image(str(panel_path), width=panel_width, height=panel_height, scale=1)
            panel_paths.append(panel_path)

        canvas = Image.new('RGB', (cols * panel_width, rows * panel_height), 'white')
        draw = ImageDraw.Draw(canvas)
        for idx, (code, panel_path) in enumerate(zip(codes, panel_paths)):
            img = crop_white_margins(Image.open(panel_path).convert('RGB'))
            img = ImageOps.contain(img, (panel_width, panel_height), method=Image.Resampling.LANCZOS)
            row, col = divmod(idx, cols)
            x0 = col * panel_width
            y0 = row * panel_height
            canvas.paste(img, (x0, y0))
            draw.rectangle([x0, y0, x0 + panel_width - 1, y0 + panel_height - 1], outline='#D9D9D9', width=2)

        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, dpi=(PNG_DPI, PNG_DPI))
        print(f'Wrote {output} as 4x4 overview grid')


def discover_multirun_jobs(multirun_dir: Path) -> list[dict]:
    jobs = []
    for job_dir in sorted(
        [p for p in multirun_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    ):
        cfg_path = job_dir / 'config.yaml'
        w_path = job_dir / 'W_est.csv'
        log_path = job_dir / 'run_experiments.log'
        if not cfg_path.exists() or not w_path.exists() or not log_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get('problem', {}).get('name') != 'industry_eu':
            continue
        log_text = log_path.read_text(errors='ignore')
        if 'Experiment Finished' not in log_text:
            continue
        w = np.loadtxt(w_path, delimiter=',')
        try:
            labels = labels_from_problem_config(cfg, w)
        except ValueError:
            best_features = parse_best_features_from_log(log_path)
            labels = [*best_features, cfg['problem']['target']]
            if len(labels) != w.shape[0]:
                continue
        jobs.append(
            {
                'job_dir': job_dir,
                'cfg': cfg,
                'w': w,
                'labels': labels,
                'size': len(labels),
                'problem_size': len(cfg['problem']['features']) + 1,
                'target': cfg['problem']['target'],
            }
        )
    return jobs


def write_panel_image(fig: go.Figure, output: Path, width: int, height: int) -> None:
    fig_json = output.with_suffix('.json')
    svg_path = output.with_suffix('.svg')
    fig_json.write_text(fig.to_json())
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib, plotly.io as pio; "
                    f"fig=pio.from_json(pathlib.Path(r'{fig_json}').read_text()); "
                    f"fig.write_image(r'{svg_path}', format='svg', width={width}, height={height}, scale=1)"
                ),
            ],
            check=True,
        )
        subprocess.run(
            [
                "/usr/local/bin/rsvg-convert",
                "-f",
                "png",
                "-w",
                str(width),
                "-h",
                str(height),
                "-o",
                str(output),
                str(svg_path),
            ],
            check=True,
        )
    finally:
        fig_json.unlink(missing_ok=True)
        svg_path.unlink(missing_ok=True)


def write_target_grid(
    jobs: list[dict],
    cols: int,
    rows: int,
    output: Path,
    threshold: float,
    top_k: int,
) -> None:
    if not jobs:
        print(f'No completed jobs available for {output.name}')
        return

    jobs = sorted(jobs, key=lambda item: item['target'])
    first_cfg = jobs[0]['cfg']
    values = latest_values(pd.read_csv(ROOT / first_cfg['problem']['data_path']), first_cfg['problem']['frequency'])
    panel_width = GRID_CANVAS_WIDTH // cols
    panel_height = GRID_CANVAS_HEIGHT // rows

    with tempfile.TemporaryDirectory(prefix='industry-target-grid-') as tmpdir:
        tmpdir_path = Path(tmpdir)
        panel_paths: list[Path] = []
        for job in jobs:
            edges = build_edge_table(job['w'], job['labels'], threshold, top_k)
            plot_df, size = build_plot_df(values, job['target'])
            fig = build_figure(
                plot_df,
                size,
                edges,
                title=COUNTRY_META[job['target']]['name'],
                subtitle='',
                show_legend=False,
                width=panel_width,
                height=panel_height,
            )
            panel_path = tmpdir_path / f"{job['target']}.png"
            write_panel_image(fig, panel_path, panel_width, panel_height)
            panel_paths.append(panel_path)

        canvas = Image.new('RGB', (GRID_CANVAS_WIDTH, GRID_CANVAS_HEIGHT), 'white')
        draw = ImageDraw.Draw(canvas)
        for idx, panel_path in enumerate(panel_paths):
            img = crop_white_margins(Image.open(panel_path).convert('RGB'))
            img = ImageOps.contain(img, (panel_width, panel_height), method=Image.Resampling.LANCZOS)
            row, col = divmod(idx, cols)
            x0 = col * panel_width
            y0 = row * panel_height
            canvas.paste(img, (x0, y0))
            draw.rectangle([x0, y0, x0 + panel_width - 1, y0 + panel_height - 1], outline='#D9D9D9', width=2)

        output.parent.mkdir(parents=True, exist_ok=True)
        canvas = crop_white_margins(canvas, border=0)
        canvas = fit_image_to_canvas(canvas, GRID_CANVAS_WIDTH, GRID_CANVAS_HEIGHT)
        canvas.save(output, dpi=(PNG_DPI, PNG_DPI))
        print(f'Wrote {output}')


def main() -> None:
    args = parse_args()
    if args.grid_industry_targets:
        jobs = discover_multirun_jobs(args.multirun_dir)
        jobs9 = [job for job in jobs if job['problem_size'] == 9]
        jobs16 = [job for job in jobs if job['problem_size'] == 16]
        write_target_grid(jobs9, cols=3, rows=3, output=args.grid_9_output, threshold=args.edge_threshold, top_k=args.top_k_edges)
        write_target_grid(jobs16, cols=4, rows=4, output=args.grid_16_output, threshold=args.edge_threshold, top_k=args.top_k_edges)
        return

    df = pd.read_csv(args.data_path)
    values = latest_values(df, args.frequency)

    if args.w_matrix is None:
        w_path, cfg_path, sel_path = autodiscover_w_artifacts()
    else:
        w_path, cfg_path, sel_path = args.w_matrix, args.config, args.selected_features

    w = np.loadtxt(w_path, delimiter=',')
    selected = parse_selected_features(sel_path)
    target = parse_target_from_config(cfg_path)
    labels = infer_labels(w, selected, target)
    edges = build_edge_table(w, labels, args.edge_threshold, args.top_k_edges)

    write_grid_overview(values, target, edges, args.grid_output)
    print(f'Using W matrix: {w_path}')
    print(f'Using labels: {labels}')
    if edges.empty:
        print('No edges passed the threshold; lower --edge-threshold if needed.')


if __name__ == '__main__':
    main()
