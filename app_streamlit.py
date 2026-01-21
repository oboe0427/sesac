# app_streamlit.py
from __future__ import annotations

import os
import re
import platform
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb, LinearSegmentedColormap

import plotly.graph_objects as go
from plotly.colors import qualitative

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# 0) 경로 유틸 (로컬/클라우드 공통)
# =========================
ROOT = Path(__file__).resolve().parent
BASE_CANDIDATES = [
    ROOT,                      # repo root
    ROOT / "team_project_1",    # repo 안에 team_project_1/가 있는 구조 대응
    ROOT.parent / "team_project_1",  # 상위에 team_project_1/가 있는 구조 대응
]


def pick_existing_path(*parts: str, fallback_base: Path = ROOT) -> str:
    """가능한 base들에서 parts 경로가 존재하는 첫 번째 것을 반환. 없으면 fallback_base/parts 반환."""
    for base in BASE_CANDIDATES:
        p = base.joinpath(*parts)
        if p.exists():
            return str(p)
    return str(fallback_base.joinpath(*parts))


# =========================
# 1) 기본 설정(필요시 수정)
# =========================
# ✅ shp 대신 geojson을 기본으로 사용
DEFAULT_GEO_PATH = pick_existing_path("geodata", "seoul_gu.geojson")
DEFAULT_CSV_PATH = pick_existing_path("data", "00_merged_final_re.csv")
DEFAULT_BURDEN_TS_PATH = pick_existing_path("data", "01_combined.csv")

CSV_KEY = "자치구"

SEOUL_GU = [
    "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구","노원구",
    "은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구","관악구",
    "서초구","강남구","송파구","강동구"
]

BG_COLOR = "#e6e6e6"

# 3x3 팔레트
BIVAR_PALETTE_3x3 = [
    ["#e8e8e8", "#e4acac", "#c85a5a"],
    ["#b0d5df", "#ad9ea5", "#985356"],
    ["#64acbe", "#627f8c", "#574249"],
]

# 4x4 팔레트
BIVAR_PALETTE_4x4 = [
    ["#e8e8e8", "#dfb0d6", "#be64ac", "#8c3b78"],
    ["#ace4e4", "#a5add3", "#8c62aa", "#3b4994"],
    ["#5ac8c8", "#5698b9", "#5a5aa6", "#2f2f7f"],
    ["#1f968b", "#1d6ea6", "#2c3784", "#0d1b4c"],
]


def short_gu_name(s: str) -> str:
    """'강남구' -> '강남', '구로구' -> '구로' (맨 뒤 '구'만 제거)"""
    return re.sub(r"구$", "", str(s).strip())


def normalize_gu(s: str) -> str:
    s = str(s).strip()
    s = s.replace("서울특별시", "").replace("서울시", "").strip()
    s = re.sub(r"\s+", "", s)
    return s


# =========================
# 2) 한글 폰트
# =========================
def set_korean_font():
    """
    Streamlit Cloud(리눅스)에서도 한글 안 깨지게:
    - repo에 포함한 fonts/NotoSansKR-Regular.ttf 를 최우선으로 등록/사용
    - 없으면 OS별 기본 폰트 후보로 fallback
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "fonts", "NotoSansKR-Regular.ttf")

    # ✅ 1) 폰트 파일이 있으면 무조건 이걸 사용 (가장 확실)
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = font_name
        mpl.rcParams["axes.unicode_minus"] = False
        return

    # ✅ 2) fallback (로컬/다른 환경 대비)
    sysname = platform.system().lower()
    if "darwin" in sysname:
        candidates = ["AppleGothic", "Apple SD Gothic Neo"]
    elif "windows" in sysname:
        candidates = ["Malgun Gothic"]
    else:
        candidates = ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]

    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            mpl.rcParams["font.family"] = c
            break

    mpl.rcParams["axes.unicode_minus"] = False


# =========================
# 3) 캐시 로더
# =========================
@st.cache_resource
def load_seoul_gu(path: str) -> gpd.GeoDataFrame:
    """
    ✅ geojson/shp 모두 지원.
    - geojson이 이미 구 단위면: '자치구' 또는 'SIG_KOR_NM'에서 구 이름을 만들어 사용
    - shp(TN_SIGNGU_BNDRY 등)면: ADZONE_NM + LEGLCD_SE 기반 기존 로직 유지
    """
    gdf = gpd.read_file(path)

    # ---- (A) 이미 '자치구' 컬럼이 있는 경우 (가장 이상적)
    if "자치구" in gdf.columns:
        gdf["자치구"] = gdf["자치구"].apply(normalize_gu)
        seoul_gu = gdf[gdf["자치구"].isin(SEOUL_GU)].copy()
        seoul_gu = seoul_gu[seoul_gu.geometry.notna() & (~seoul_gu.geometry.is_empty)].copy()
        seoul_gu = seoul_gu.dissolve(by="자치구", as_index=False)
        return seoul_gu

    # ---- (B) SIG_KOR_NM 같은 컬럼에 "서울특별시 강남구" 형태가 있는 경우
    if "SIG_KOR_NM" in gdf.columns:
        gdf["자치구"] = (
            gdf["SIG_KOR_NM"].astype(str)
            .str.replace("서울특별시", "", regex=False)
            .str.replace("서울시", "", regex=False)
            .str.strip()
            .str.split()
            .str[-1]
        ).apply(normalize_gu)

        seoul_gu = gdf[gdf["자치구"].isin(SEOUL_GU)].copy()
        seoul_gu = seoul_gu[seoul_gu.geometry.notna() & (~seoul_gu.geometry.is_empty)].copy()
        seoul_gu = seoul_gu.dissolve(by="자치구", as_index=False)
        return seoul_gu

    # ---- (C) shp(TN_SIGNGU_BNDRY) 전용 로직
    name_col = "ADZONE_NM"
    leg_col  = "LEGLCD_SE"
    if (name_col in gdf.columns) and (leg_col in gdf.columns):
        gdf["자치구"] = (
            gdf[name_col].astype(str)
            .str.replace("서울특별시", "", regex=False)
            .str.replace("서울시", "", regex=False)
            .str.strip()
            .str.split()
            .str[-1]
        ).apply(normalize_gu)

        leg = (
            gdf[leg_col].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
        )
        sido_cd = leg.str[:2]

        seoul = gdf[(sido_cd == "11") & (gdf["자치구"].isin(SEOUL_GU))].copy()
        seoul = seoul[seoul.geometry.notna() & (~seoul.geometry.is_empty)].copy()

        seoul_gu = seoul.dissolve(by="자치구", as_index=False)
        return seoul_gu

    raise KeyError(
        "경계 파일에서 구 이름을 만들 수 없습니다.\n"
        "지원 형태:\n"
        " - geojson: '자치구' 컬럼 또는 'SIG_KOR_NM' 컬럼 필요\n"
        " - shp: 'ADZONE_NM', 'LEGLCD_SE' 컬럼 필요\n"
        f"현재 컬럼: {list(gdf.columns)}"
    )


@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(csv_path)


# =========================
# 4) 이변량 지도
# =========================
def safe_qcut(s: pd.Series, q: int) -> pd.Series:
    s = s.copy()
    if s.notna().sum() < q:
        return pd.Series([pd.NA] * len(s), index=s.index)
    try:
        return pd.qcut(s, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=q, labels=False)


def pick_bivar_color(xb, yb, palette, n_bins: int) -> str:
    if pd.isna(xb) or pd.isna(yb):
        return "#cccccc"
    xb = int(xb); yb = int(yb)
    xb = max(0, min(n_bins - 1, xb))
    yb = max(0, min(n_bins - 1, yb))
    return palette[yb][xb]


def make_bivariate_fig(
    seoul_gu: gpd.GeoDataFrame,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int,
    palette: list[list[str]],
    label_fontsize: int,
    figsize: tuple[float, float],
    legend_box: tuple[float, float, float, float],
) -> tuple[plt.Figure, gpd.GeoDataFrame, str]:

    for col in [CSV_KEY, x_col, y_col]:
        if col not in df.columns:
            raise KeyError(f"CSV에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    work = df.copy()
    work[CSV_KEY] = work[CSV_KEY].apply(normalize_gu)
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

    m = seoul_gu.merge(work[[CSV_KEY, x_col, y_col]], left_on="자치구", right_on=CSV_KEY, how="left")

    miss_x = m[x_col].isna().mean()
    miss_y = m[y_col].isna().mean()
    info = f"[INFO] merge missing rate - {x_col}: {miss_x:.1%}, {y_col}: {miss_y:.1%}"

    m["x_bin"] = safe_qcut(m[x_col], n_bins)
    m["y_bin"] = safe_qcut(m[y_col], n_bins)
    m["bivar_color"] = [
        pick_bivar_color(xb, yb, palette, n_bins)
        for xb, yb in zip(m["x_bin"], m["y_bin"])
    ]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    m.plot(ax=ax, color=m["bivar_color"], edgecolor="black", linewidth=0.8)
    ax.set_axis_off()
    ax.set_title(f"Seoul Bivariate Choropleth: {x_col} (x) vs {y_col} (y)")

    for _, row in m.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        pt = row.geometry.representative_point()
        label = re.sub(r"구$", "", row["자치구"])
        ax.text(
            pt.x, pt.y, label,
            fontsize=label_fontsize,
            ha="center", va="center",
            color="black", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

    leg_ax = ax.inset_axes(list(legend_box))
    img = np.array([[to_rgb(c) for c in row] for row in palette])
    leg_ax.imshow(img, origin="lower")

    leg_ax.set_xticks([0, n_bins - 1])
    leg_ax.set_yticks([0, n_bins - 1])
    leg_ax.set_xticklabels(["low", "high"])
    leg_ax.set_yticklabels(["low", "high"])
    leg_ax.set_xlabel(x_col, fontsize=8)
    leg_ax.set_ylabel(y_col, fontsize=8)

    leg_ax.set_xticks(np.arange(-.5, n_bins, 1), minor=True)
    leg_ax.set_yticks(np.arange(-.5, n_bins, 1), minor=True)
    leg_ax.grid(which="minor", color="white", linewidth=1.0)
    leg_ax.tick_params(which="minor", bottom=False, left=False)

    return fig, m, info


# =========================
# 5) 단일 지도
# =========================
def make_univariate_fig(
    seoul_gu: gpd.GeoDataFrame,
    df: pd.DataFrame,
    val_col: str,
    label_fontsize: int,
    figsize: tuple[float, float],
    min_color: str,
    mid_color: str,
    max_color: str,
    cbar_shrink: float,
    cbar_aspect: int,
    cbar_pad: float,
) -> tuple[plt.Figure, gpd.GeoDataFrame, str]:

    for col in [CSV_KEY, val_col]:
        if col not in df.columns:
            raise KeyError(f"CSV에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    work = df.copy()
    work[CSV_KEY] = work[CSV_KEY].apply(normalize_gu)
    work[val_col] = pd.to_numeric(work[val_col], errors="coerce")

    m = seoul_gu.merge(work[[CSV_KEY, val_col]], left_on="자치구", right_on=CSV_KEY, how="left")
    missing_rate = m[val_col].isna().mean()
    info = f"[INFO] merge missing rate - {val_col}: {missing_rate:.1%}"

    cmap = LinearSegmentedColormap.from_list("pastel_gwr", [min_color, mid_color, max_color])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    m.plot(
        column=val_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        linewidth=0.8,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
        legend_kwds={
            "shrink": cbar_shrink,
            "aspect": cbar_aspect,
            "pad": cbar_pad,
            "label": val_col
        }
    )

    ax.set_axis_off()
    ax.set_title(f"Seoul Choropleth: {val_col}")

    for _, row in m.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        pt = row.geometry.representative_point()
        ax.text(
            pt.x, pt.y,
            short_gu_name(row["자치구"]),
            fontsize=label_fontsize,
            ha="center", va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    return fig, m, info


# =========================
# 6) 노년부양비 Plotly
# =========================
@st.cache_data
def load_burden_ts(csv_path: str) -> pd.DataFrame:
    return load_csv(csv_path)


def make_burden_timeseries_fig(df: pd.DataFrame) -> tuple[go.Figure, str]:
    year_cols = [c for c in df.columns if str(c).isdigit()]
    if not year_cols:
        raise ValueError("연도 컬럼(숫자 문자열)이 없습니다. 예: '2015', '2016' ...")

    years = np.array(sorted(int(c) for c in year_cols))
    year_cols_sorted = [str(y) for y in years]

    solid_mask = (years >= 2015) & (years <= 2024)
    dash_mask  = (years >= 2025) & (years <= 2034)

    x_solid = years[solid_mask]
    x_dash  = years[dash_mask]

    name_col = df.columns[0]

    fig = go.Figure()
    color_cycle = cycle(qualitative.Alphabet)
    color_map = {}

    hover_tpl = (
        "<span style='color:%{customdata[1]}; font-weight:800'></span> "
        "<span style='color:%{customdata[1]}; font-weight:600'>%{customdata[0]}</span>"
        ": %{y:.1f}<extra></extra>"
    )

    for _, row in df.iterrows():
        gu = str(row[name_col])
        color = next(color_cycle)
        color_map[gu] = color

        y = pd.to_numeric(row[year_cols_sorted], errors="coerce").to_numpy()
        y_solid = y[solid_mask]
        y_dash  = y[dash_mask]

        fig.add_trace(go.Scatter(
            x=x_solid, y=y_solid,
            mode="lines",
            name=f"{gu} (실제)",
            showlegend=True,
            legendgroup=gu,
            line=dict(width=2, color=color),
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=x_dash, y=y_dash,
            mode="lines",
            name=f"{gu} (추계)",
            showlegend=True,
            legendgroup=gu,
            line=dict(width=2, dash="dash", color=color),
            hoverinfo="skip"
        ))

        if len(x_solid) > 0 and len(x_dash) > 0:
            fig.add_trace(go.Scatter(
                x=[x_solid[-1], x_dash[0]],
                y=[y_solid[-1], y_dash[0]],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(width=2, dash="dash", color=color)
            ))

    val_mat = df[year_cols_sorted].apply(pd.to_numeric, errors="coerce").to_numpy()
    gus = df[name_col].astype(str).to_numpy()

    for yi, year in enumerate(years):
        items = []
        for gi, gu in enumerate(gus):
            yv = val_mat[gi, yi]
            if yv is None or (isinstance(yv, float) and np.isnan(yv)):
                continue
            items.append((gu, float(yv)))

        items.sort(key=lambda x: x[1], reverse=True)

        for gu, yv in items:
            c = color_map.get(gu, "#000000")
            fig.add_trace(go.Scatter(
                x=[year, year],
                y=[yv, yv],
                mode="lines",
                line=dict(color=c, width=3),
                showlegend=False,
                legendgroup=gu,
                customdata=[[gu, c], [gu, c]],
                hovertemplate=hover_tpl,
                hoveron="points",
            ))

    fig.update_layout(
        title=dict(
            text="노년부양비(2015–2034): 2015–2024 실제 인구, 2025–2034 추계 인구",
            font=dict(size=18)
        ),
        xaxis=dict(title="연도", tickmode="array", tickvals=years, tickangle=45),
        yaxis=dict(title=None),
        hovermode="x unified",
        legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle", groupclick="togglegroup"),
        margin=dict(l=40, r=260, t=80, b=80),
        height=1000,
        autosize=True
    )

    info = f"[INFO] rows={len(df)}, years={years.min()}-{years.max()}, name_col='{name_col}'"
    return fig, info


# =========================
# 7) 다중회귀(OLS + HC3) + Partial Regression Plot
# =========================
def _clean_for_reg(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    if CSV_KEY not in df.columns:
        raise KeyError(f"CSV에 '{CSV_KEY}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    needed = [CSV_KEY, y_col] + x_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다. 누락: {missing}\n현재: {list(df.columns)}")

    work = df[needed].copy()
    work[CSV_KEY] = work[CSV_KEY].astype(str).apply(normalize_gu)

    for c in [y_col] + x_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[y_col] + x_cols).copy()
    return work


def fit_ols_hc3(work: pd.DataFrame, y_col: str, x_cols: list[str]):
    y = work[y_col].astype(float)
    X = sm.add_constant(work[x_cols].astype(float))
    res = sm.OLS(y, X).fit(cov_type="HC3")
    return res


def summarize_hc3(res, y_name: str) -> pd.DataFrame:
    ci = res.conf_int()
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "se_hc3": res.bse.values,
        "pvalue": res.pvalues.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })
    out.insert(0, "y", y_name)
    out["n"] = int(res.nobs)
    out["r2"] = float(res.rsquared)
    out["adj_r2"] = float(res.rsquared_adj)
    return out


def compute_vif(work: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    X = sm.add_constant(work[x_cols].astype(float))
    rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        rows.append((name, float(variance_inflation_factor(X.values, i))))
    return pd.DataFrame(rows, columns=["variable", "VIF"])


def build_color_mask(gu_series: pd.Series, red_list: list[str], green_list: list[str]) -> np.ndarray:
    gu = gu_series.astype(str)
    colors = np.array(["blue"] * len(gu), dtype=object)
    red_set = set(red_list)
    green_set = set(green_list)

    is_red = gu.isin(red_set).to_numpy()
    is_green = gu.isin(green_set).to_numpy() & (~is_red)

    colors[is_green] = "green"
    colors[is_red] = "red"
    return colors


def plot_with_labels_partial(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    fontsize: int = 8,
    point_size: float = 22,
    point_alpha: float = 0.9,
):
    colors = np.asarray(colors, dtype=object)
    is_red = colors == "red"
    is_green = colors == "green"
    is_base = ~(is_red | is_green)

    ax.scatter(x[is_base], y[is_base], s=point_size, alpha=point_alpha)
    if is_green.any():
        ax.scatter(x[is_green], y[is_green], s=point_size * 1.3, alpha=1.0, color="green", zorder=3)
    if is_red.any():
        ax.scatter(x[is_red], y[is_red], s=point_size * 1.3, alpha=1.0, color="red", zorder=4)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    y_offsets = [0, 6, -6, 10, -10, 14, -14, 18, -18, 24, -24, 30, -30]
    placed = []

    for xi, yi, lab, c in zip(x, y, labels, colors):
        txt_color = c if c in ("red", "green") else None
        ann = ax.annotate(
            short_gu_name(lab),
            (xi, yi),
            xytext=(6, y_offsets[0]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=txt_color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
            zorder=5 if c == "red" else (4 if c == "green" else 2),
        )

        fig.canvas.draw()
        bb = ann.get_window_extent(renderer=renderer).expanded(1.02, 1.10)
        k = 0
        while any(bb.overlaps(p) for p in placed) and k < len(y_offsets) - 1:
            k += 1
            ann.set_position((6, y_offsets[k]))
            fig.canvas.draw()
            bb = ann.get_window_extent(renderer=renderer).expanded(1.02, 1.10)

        placed.append(bb)


def make_partial_reg_fig(
    work: pd.DataFrame,
    y_col: str,
    x_focus: str,
    x_cols: list[str],
    highlight_red: list[str],
    highlight_green: list[str],
    figsize=(9, 7),
    label_fontsize=8,
) -> tuple[plt.Figure, dict]:

    if x_focus not in x_cols:
        raise ValueError("x_focus는 선택된 X 목록에 있어야 합니다.")

    others = [c for c in x_cols if c != x_focus]

    if len(others) > 0:
        Xo = sm.add_constant(work[others].astype(float))
        y_resid = sm.OLS(work[y_col].astype(float), Xo).fit().resid.to_numpy()
        x_resid = sm.OLS(work[x_focus].astype(float), Xo).fit().resid.to_numpy()
    else:
        y_resid = (work[y_col].astype(float) - np.nanmean(work[y_col].astype(float))).to_numpy()
        x_resid = (work[x_focus].astype(float) - np.nanmean(work[x_focus].astype(float))).to_numpy()

    res = fit_ols_hc3(work, y_col, x_cols)
    b = float(res.params.get(x_focus, np.nan))
    p = float(res.pvalues.get(x_focus, np.nan))
    r2 = float(res.rsquared)
    n = int(res.nobs)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = work[CSV_KEY].astype(str).to_numpy()
    colors = build_color_mask(work[CSV_KEY], highlight_red, highlight_green)

    plot_with_labels_partial(
        ax=ax,
        x=x_resid,
        y=y_resid,
        labels=labels,
        colors=colors,
        fontsize=label_fontsize,
    )

    xr = np.linspace(np.nanmin(x_resid), np.nanmax(x_resid), 200)
    yr = b * xr
    ax.plot(xr, yr, linewidth=2, label=f"Partial slope β={b:.6f}")

    ax.axhline(0, linewidth=1, alpha=0.4)
    ax.axvline(0, linewidth=1, alpha=0.4)
    ax.legend(loc="best", frameon=True)

    ax.set_xlabel(f"{x_focus} (residualized on other X)")
    ax.set_ylabel(f"{y_col} (residualized on other X)")
    ax.set_title(f"Partial Regression (HC3): {y_col} ~ {x_focus} | controls={others if others else 'None'}")

    box_txt = f"β({x_focus}) = {b:.6f}\np = {p:.4g}\nR²(full) = {r2:.3f}\nN = {n}"
    ax.text(
        0.98, 0.02, box_txt,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        zorder=10
    )

    meta = {"beta": b, "p": p, "r2": r2, "n": n}
    return fig, meta


# =========================
# 8) Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="Seoul 시각화 (단일/이변량/노년부양비/다중회귀)", layout="wide")
    set_korean_font()

    st.title("Seoul 시각화 (단일 / 이변량 / 노년부양비 / 다중회귀)")

    with st.sidebar:
        st.header("설정")

        mode = st.radio(
            "모드",
            ["단일(단변량)", "이변량", "노년부양비", "다중회귀"],
            index=1
        )

        # 공통 UI
        label_fontsize = st.slider("라벨 크기", min_value=6, max_value=14, value=8, step=1)
        fig_w = st.slider("그림 너비", 6, 16, 10, 1)
        fig_h = st.slider("그림 높이", 5, 12, 8, 1)

        if mode in ["단일(단변량)", "이변량"]:
            geo_path = st.text_input("GEO_PATH (GeoJSON/SHP)", value=DEFAULT_GEO_PATH)
            csv_path = st.text_input("CSV_PATH", value=DEFAULT_CSV_PATH)

        elif mode == "노년부양비":
            burden_ts_path = st.text_input("노년부양비 CSV_PATH", value=DEFAULT_BURDEN_TS_PATH)

        else:  # 다중회귀
            csv_path = st.text_input("CSV_PATH (회귀용)", value=DEFAULT_CSV_PATH)

        # 모드별 옵션
        if mode == "이변량":
            st.divider()
            n_bins = st.selectbox("분할 개수 (N_BINS)", options=[3, 4], index=0)
            st.caption("Legend 박스 위치 [x0, y0, w, h]")
            legend_x0 = st.slider("legend x0", 0.0, 0.4, 0.03, 0.01)
            legend_y0 = st.slider("legend y0", 0.0, 0.95, 0.72, 0.01)
            legend_w  = st.slider("legend w", 0.10, 0.35, 0.22, 0.01)
            legend_h  = st.slider("legend h", 0.10, 0.35, 0.22, 0.01)

        if mode == "단일(단변량)":
            st.divider()
            st.caption("컬러맵(최저 → 중간 → 최고)")
            min_color = st.color_picker("MIN_COLOR", "#46E851")
            mid_color = st.color_picker("MID_COLOR", "#ffffff")
            max_color = st.color_picker("MAX_COLOR", "#E85E46")

            st.caption("Colorbar 옵션")
            cbar_shrink = st.slider("CBar_SHRINK", 0.40, 1.00, 0.90, 0.02)
            cbar_aspect = st.slider("CBar_ASPECT", 10, 60, 30, 1)
            cbar_pad = st.slider("CBar_PAD", 0.00, 0.10, 0.02, 0.005)

    # =========================
    # 노년부양비 모드
    # =========================
    if mode == "노년부양비":
        try:
            df_ts = load_burden_ts(burden_ts_path)
            fig, info = make_burden_timeseries_fig(df_ts)
        except Exception as e:
            st.error(f"노년부양비 차트 생성 실패: {e}")
            st.stop()

        st.caption(info)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("데이터(미리보기)"):
            st.dataframe(df_ts, use_container_width=True)

        return

    # =========================
    # 다중회귀 모드 (회귀1 vs 회귀2 비교)
    # =========================
    if mode == "다중회귀":
        try:
            df_all = load_csv(csv_path)
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")
            st.stop()

        numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        if CSV_KEY in numeric_cols:
            numeric_cols.remove(CSV_KEY)

        candidate_cols = numeric_cols if len(numeric_cols) > 0 else [c for c in df_all.columns if c != CSV_KEY]
        candidate_cols = candidate_cols[::-1]  # BurdenIndex ... 2015 순서로 뒤집기

        if len(candidate_cols) < 2:
            st.error("회귀에 사용할 숫자형 컬럼이 부족합니다.")
            st.stop()

        st.subheader("다중회귀 비교 (OLS + HC3) : 회귀1 vs 회귀2")

        gu_list = (
            sorted(df_all[CSV_KEY].astype(str).dropna().unique().tolist())
            if CSV_KEY in df_all.columns else []
        )

        def safe_selectbox(label: str, options: list, key: str, default=None):
            if not options:
                st.warning(f"{label}: 선택 가능한 옵션이 없습니다.")
                return None
            if key in st.session_state and st.session_state[key] not in options:
                st.session_state[key] = options[0]
            if key not in st.session_state and default is not None and default in options:
                st.session_state[key] = default
            return st.selectbox(label, options=options, key=key)

        def run_reg_panel(prefix: str, defaults: dict):
            st.markdown(f"### {prefix.upper()} 설정")

            y_col = safe_selectbox(
                "Y (종속변수)",
                options=candidate_cols,
                key=f"{prefix}_y",
                default=defaults.get("y"),
            )
            if y_col is None:
                return None

            st.caption("X는 최대 3개까지 선택 (x2/x3는 '(없음)' 가능)")
            none_opt = ["(없음)"]

            x_pool = [c for c in candidate_cols if c != y_col]
            if len(x_pool) < 1:
                st.error("X 후보가 부족합니다. (Y로 선택한 컬럼 제외 후 1개 이상 필요)")
                return None

            x1 = safe_selectbox(
                "X1 (필수)",
                options=x_pool,
                key=f"{prefix}_x1",
                default=defaults.get("x1"),
            )
            if x1 is None:
                return None

            x2_options = none_opt + [c for c in x_pool if c != x1]
            x2 = safe_selectbox(
                "X2 (선택)",
                options=x2_options,
                key=f"{prefix}_x2",
                default=defaults.get("x2", "(없음)"),
            )
            if x2 is None:
                return None

            x3_candidates = [c for c in x_pool if c not in {x1} and (x2 == "(없음)" or c != x2)]
            x3_options = none_opt + x3_candidates
            x3 = safe_selectbox(
                "X3 (선택)",
                options=x3_options,
                key=f"{prefix}_x3",
                default=defaults.get("x3", "(없음)"),
            )
            if x3 is None:
                return None

            x_cols = [x1]
            if x2 != "(없음)":
                x_cols.append(x2)
            if x3 != "(없음)":
                x_cols.append(x3)

            x_focus_default = defaults.get("x_focus")
            if x_focus_default not in x_cols:
                x_focus_default = x_cols[0]

            x_focus = safe_selectbox(
                "Partial regression으로 볼 X",
                options=x_cols,
                key=f"{prefix}_x_focus",
                default=x_focus_default,
            )
            if x_focus is None:
                return None

            with st.expander("구 강조(선택) - red/green", expanded=False):
                highlight_red = st.multiselect("RED 강조", options=gu_list, default=[], key=f"{prefix}_hi_red")
                highlight_green = st.multiselect("GREEN 강조", options=gu_list, default=[], key=f"{prefix}_hi_green")

            try:
                work = _clean_for_reg(df_all, y_col=y_col, x_cols=x_cols)
                res = fit_ols_hc3(work, y_col=y_col, x_cols=x_cols)
                coef_df = summarize_hc3(res, y_name=y_col)

                fig, _meta = make_partial_reg_fig(
                    work=work,
                    y_col=y_col,
                    x_focus=x_focus,
                    x_cols=x_cols,
                    highlight_red=highlight_red,
                    highlight_green=highlight_green,
                    figsize=(fig_w, fig_h),
                    label_fontsize=label_fontsize,
                )

                vif_df = (
                    compute_vif(work, x_cols)
                    if len(x_cols) >= 2
                    else pd.DataFrame({"variable": x_cols, "VIF": [np.nan] * len(x_cols)})
                )

            except Exception as e:
                st.error(f"{prefix} 회귀/시각화 실패: {e}")
                return None

            return {"res": res, "coef_df": coef_df, "vif_df": vif_df, "fig": fig}

        reg1_defaults = {
            "y": "Care",
            "x1": "BurdenIndex",
            "x2": "사회복지예산(1224.9)",
            "x3": "인구밀도(15857)",
            "x_focus": "BurdenIndex",
        }

        reg2_defaults = {
            "y": "Leisure",
            "x1": "BurdenIndex",
            "x2": "사회복지예산(1224.9)",
            "x3": "인구밀도(15857)",
            "x_focus": "BurdenIndex",
        }

        left, right = st.columns(2, gap="large")

        with left:
            out1 = run_reg_panel("reg1", defaults=reg1_defaults)
            if out1 is not None:
                r = out1["res"]
                st.caption(f"[REG1] N={int(r.nobs)}, R²={r.rsquared:.3f}, Adj.R²={r.rsquared_adj:.3f}")
                st.pyplot(out1["fig"], clear_figure=True)

                with st.expander("REG1 회귀 결과(HC3)", expanded=False):
                    st.dataframe(out1["coef_df"], use_container_width=True)
                with st.expander("REG1 VIF (선택된 X)", expanded=False):
                    st.dataframe(out1["vif_df"], use_container_width=True)

        with right:
            out2 = run_reg_panel("reg2", defaults=reg2_defaults)
            if out2 is not None:
                r = out2["res"]
                st.caption(f"[REG2] N={int(r.nobs)}, R²={r.rsquared:.3f}, Adj.R²={r.rsquared_adj:.3f}")
                st.pyplot(out2["fig"], clear_figure=True)

                with st.expander("REG2 회귀 결과(HC3)", expanded=False):
                    st.dataframe(out2["coef_df"], use_container_width=True)
                with st.expander("REG2 VIF (선택된 X)", expanded=False):
                    st.dataframe(out2["vif_df"], use_container_width=True)

        return

    # =========================
    # 지도 모드(단일/이변량)
    # =========================
    try:
        seoul_gu = load_seoul_gu(geo_path)
        df = load_csv(csv_path)
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        st.stop()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if CSV_KEY in numeric_cols:
        numeric_cols.remove(CSV_KEY)
    candidate_cols = numeric_cols if len(numeric_cols) > 0 else [c for c in df.columns if c != CSV_KEY]
    candidate_cols = candidate_cols[::-1]

    def safe_selectbox(label: str, options: list, key: str, default=None):
        if not options:
            st.warning(f"{label}: 선택 가능한 옵션이 없습니다.")
            return None
        if key in st.session_state and st.session_state[key] not in options:
            st.session_state[key] = options[0]
        if key not in st.session_state and default is not None and default in options:
            st.session_state[key] = default
        return st.selectbox(label, options=options, key=key)

    # ---- 이변량(left/right)
    if mode == "이변량":
        palette = BIVAR_PALETTE_3x3 if n_bins == 3 else BIVAR_PALETTE_4x4

        def run_bivar_panel(prefix: str, defaults: dict):
            st.markdown(f"### {prefix.upper()} (이변량)")

            x_col = safe_selectbox(
                "X 컬럼 (가로축)",
                options=candidate_cols,
                key=f"{prefix}_bivar_x",
                default=defaults.get("x"),
            )
            if x_col is None:
                return

            y_col = safe_selectbox(
                "Y 컬럼 (세로축)",
                options=candidate_cols,
                key=f"{prefix}_bivar_y",
                default=defaults.get("y"),
            )
            if y_col is None:
                return

            if x_col == y_col:
                st.warning("X와 Y가 같습니다. 다른 컬럼을 선택해 주세요.")

            try:
                fig, merged_gdf, info = make_bivariate_fig(
                    seoul_gu=seoul_gu,
                    df=df,
                    x_col=x_col,
                    y_col=y_col,
                    n_bins=n_bins,
                    palette=palette,
                    label_fontsize=label_fontsize,
                    figsize=(fig_w, fig_h),
                    legend_box=(legend_x0, legend_y0, legend_w, legend_h),
                )
            except Exception as e:
                st.error(f"{prefix} 지도 생성 실패: {e}")
                return

            st.caption(info)
            st.pyplot(fig, clear_figure=True)

            with st.expander(f"{prefix.upper()} 분류 결과(표)", expanded=False):
                show_cols = ["자치구", x_col, y_col, "x_bin", "y_bin", "bivar_color"]
                out_df = merged_gdf[show_cols].copy()
                st.dataframe(out_df, use_container_width=True)

        bivar_left_defaults = {
            "x": "BurdenIndex" if "BurdenIndex" in candidate_cols else (candidate_cols[0] if candidate_cols else None),
            "y": "Care" if "Care" in candidate_cols else (candidate_cols[0] if candidate_cols else None),
        }
        bivar_right_defaults = {
            "x": "BurdenIndex" if "BurdenIndex" in candidate_cols else (candidate_cols[0] if candidate_cols else None),
            "y": "Leisure" if "Leisure" in candidate_cols else (candidate_cols[0] if candidate_cols else None),
        }

        left, right = st.columns(2, gap="large")
        with left:
            run_bivar_panel("left", bivar_left_defaults)
        with right:
            run_bivar_panel("right", bivar_right_defaults)

        return

    # ---- 단일(단변량) left/right
    else:
        def run_uni_panel(prefix: str, defaults: dict):
            st.markdown(f"### {prefix.upper()} (단일)")

            val_col = safe_selectbox(
                "값 컬럼 (VAL_COL)",
                options=candidate_cols,
                key=f"{prefix}_uni_val",
                default=defaults.get("val"),
            )
            if val_col is None:
                return

            try:
                fig, merged_gdf, info = make_univariate_fig(
                    seoul_gu=seoul_gu,
                    df=df,
                    val_col=val_col,
                    label_fontsize=label_fontsize,
                    figsize=(fig_w, fig_h),
                    min_color=min_color,
                    mid_color=mid_color,
                    max_color=max_color,
                    cbar_shrink=cbar_shrink,
                    cbar_aspect=int(cbar_aspect),
                    cbar_pad=float(cbar_pad),
                )
            except Exception as e:
                st.error(f"{prefix} 지도 생성 실패: {e}")
                return

            st.caption(info)
            st.pyplot(fig, clear_figure=True)

            with st.expander(f"{prefix.upper()} 결과(표)", expanded=False):
                out_df = merged_gdf[["자치구", val_col]].copy()
                st.dataframe(out_df, use_container_width=True)

        uni_left_defaults = {
            "val": "BurdenIndex" if "BurdenIndex" in candidate_cols else (candidate_cols[0] if candidate_cols else None)
        }
        uni_right_defaults = {
            "val": "Care" if "Care" in candidate_cols else (candidate_cols[0] if candidate_cols else None)
        }

        left, right = st.columns(2, gap="large")
        with left:
            run_uni_panel("left", uni_left_defaults)
        with right:
            run_uni_panel("right", uni_right_defaults)

        return


if __name__ == "__main__":
    main()
