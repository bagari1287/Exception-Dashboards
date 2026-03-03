# app.py
# =========================================================
# PETS - WES Ops Dashboard (Upload-ready)
#
# CHANGES (your latest requests):
# 1) KEEP carrier brand colours (EVRI/DPD/Parcel Force) + KEEP Area colours (do not change)
# 2) For ALL OTHER charts: use severity colour order (high -> low)
#    Red -> Orange -> Yellow -> Light Green -> Green -> Grey
# 3) Add Executive Dashboard (PowerBI-style) tab:
#    - Donut chart for Shift % (Last 24h)
#    - Replace "Last 24h: exception type share %" chart with:
#        ✅ Most affected tote (Source/Destination) + top source tote + top destination tote
#    - Replace "Top lanes by boxes" with:
#        ✅ Ship lane % share (derived) (carrier brand colours preserved)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="PETS WES Ops Dashboard", page_icon="🐾", layout="wide")

# =========================================================
# Theme selector + Plotly template
# =========================================================
THEME = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"], index=0)
px.defaults.template = "plotly_white" if THEME == "Light" else "plotly_dark"

# =========================================================
# CSS (background + chart cards + metric cards + no overflow)
# =========================================================
LIGHT_CSS = """
<style>
  :root{
    --bg0:#F6F8FF; --bg1:#EEF2FF;
    --stroke:rgba(10,20,60,0.10);
    --text:rgba(10,15,30,0.92); --muted:rgba(10,15,30,0.65);
  }
  .stApp{
    background:
      radial-gradient(1100px 600px at 15% 10%, rgba(47,126,247,0.16), transparent 55%),
      radial-gradient(900px 550px at 85% 20%, rgba(25,179,122,0.14), transparent 55%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
  }
  .block-container {padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1750px;}
  h1 {font-size: 32px !important;}
  h2 {font-size: 24px !important;}
  h3 {font-size: 18px !important;}
  h1,h2,h3,{color: var(--text); !important}
  
div {
  color: green;
}

.st-emotion-cache-1c87b1z {
color: rgb(33 31 31);
}

.st-emotion-cache-p7mjpa {
    width: calc(16.6667% - 1rem);
    flex: 1 1 calc(16.6667% - 1rem);
    padding: 1rem 1.2rem;
    border-radius: 14px;
    color: #fff;
    display: flex;
    flex-direction: column;
    justify-content: space-between;

    /* --- 3D Magic Starts Here --- */
    background: linear-gradient(145deg, #ffffff, #e0e0e0);
    box-shadow:
        8px 8px 16px rgba(0, 0, 0, 0.25),      /* deep outer shadow */
        -4px -4px 12px rgba(255, 255, 255, .8), /* top-left highlight */
        inset 2px 2px 6px rgba(0,0,0,0.15),     /* inner depth */
        inset -2px -2px 6px rgba(255,255,255,0.7); /* inner highlight */

    transform: translateY(0);
    transition: all 0.25s ease;
}

/* 3D Hover lift effect */
.st-emotion-cache-p7mjpa:hover {
    transform: translateY(-6px);
    box-shadow:
        12px 12px 24px rgba(0,0,0,0.3),
        -4px -4px 16px rgba(255,255,255,0.9),
        inset 1px 1px 3px rgba(0,0,0,0.1),
        inset -1px -1px 3px rgba(255,255,255,0.6);
}

  .hero{
    display:flex; align-items:center; justify-content:space-between; gap:18px;
    padding: 14px 16px; border:1px solid var(--stroke);
    background: rgba(255,255,255,0.80);
    border-radius: 18px; box-shadow: 0 18px 40px rgba(10,20,60,0.10);
    margin-bottom: 12px;
  }
  .subtitle{opacity:0.78; font-size: 14px; margin-top:-6px; color: var(--muted) !important;}
  .pill{
    display:inline-flex; align-items:center; gap:8px;
    padding: 7px 12px; border-radius: 999px;
    border: 1px solid rgba(10,20,60,0.12);
    background: rgba(255,255,255,0.75);
    font-size: 12.5px; opacity: 0.95;
  }

  .hr {height:1px; background: rgba(10,20,60,0.10); margin: 14px 0;}
  .section {font-size: 20px; font-weight: 900; margin: 8px 0 10px 0;}

  [data-testid="stPlotlyChart"]{
    border-radius: 16px;
    border: 1px solid rgba(10,20,60,0.10);
    background: rgba(255,255,255,0.92);
    box-shadow: 0 16px 44px rgba(10,20,60,0.08);
    padding: 6px;
  }

  div[data-testid="metric-container"]{
    border: 1px solid rgba(10,20,60,0.10);
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.78));
    border-radius: 18px;
    padding: 14px 14px;
    box-shadow: 0 18px 44px rgba(10,20,60,0.10);
    height: 100%;
    overflow: hidden;
  }
  div[data-testid="metric-container"] [data-testid="stMetricLabel"]{
    font-size: 0.90rem;
    line-height: 1.15rem;
    white-space: normal;
    overflow-wrap: anywhere;
    opacity: 0.85;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-size: 1.55rem;
    line-height: 1.9rem;
    white-space: nowrap;
    text-overflow: ellipsis;
    overflow: hidden;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"]{
    font-size: 0.85rem;
    white-space: normal;
    overflow-wrap: anywhere;
    opacity: 0.85;
  }
</style>
"""

DARK_CSS = """
<style>
  :root{
    --bg0:#070A12; --bg1:#0B1020;
    --stroke:rgba(255,255,255,0.10);
    --text:rgba(255,255,255,0.92); --muted:rgba(255,255,255,0.72);
  }
  .stApp{
    background:
      radial-gradient(1200px 600px at 20% 10%, rgba(78,168,255,0.20), transparent 55%),
      radial-gradient(900px 550px at 80% 20%, rgba(46,229,157,0.18), transparent 55%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
  }
  .block-container {padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1750px;}
  h1 {font-size: 32px !important;}
  h2 {font-size: 24px !important;}
  h3 {font-size: 18px !important;}
  h1,h2,h3,p,li,span,div {color: var(--text) !important;}

  .hero{
    display:flex; align-items:center; justify-content:space-between; gap:18px;
    padding: 14px 16px; border:1px solid var(--stroke);
    background: rgba(255,255,255,0.06);
    border-radius: 18px; box-shadow: 0 18px 40px rgba(0,0,0,0.40);
    margin-bottom: 12px;
  }
  .subtitle{opacity:0.78; font-size: 14px; margin-top:-6px; color: var(--muted) !important;}
  .pill{
    display:inline-flex; align-items:center; gap:8px;
    padding: 7px 12px; border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    font-size: 12.5px; opacity: 0.92;
  }

  .hr {height:1px; background: rgba(255,255,255,0.10); margin: 14px 0;}
  .section {font-size: 20px; font-weight: 900; margin: 8px 0 10px 0;}

  [data-testid="stPlotlyChart"]{
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(10,16,32,0.55);
    box-shadow: 0 16px 44px rgba(0,0,0,0.42);
    padding: 6px;
  }

  div[data-testid="metric-container"]{
    border: 1px solid rgba(255,255,255,0.10);
    background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
    border-radius: 18px;
    padding: 14px 14px;
    box-shadow: 0 18px 44px rgba(0,0,0,0.45);
    height: 100%;
    overflow: hidden;
  }
  div[data-testid="metric-container"] [data-testid="stMetricLabel"]{
    font-size: 0.90rem;
    line-height: 1.15rem;
    white-space: normal;
    overflow-wrap: anywhere;
    opacity: 0.88;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-size: 1.55rem;
    line-height: 1.9rem;
    white-space: nowrap;
    text-overflow: ellipsis;
    overflow: hidden;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"]{
    font-size: 0.85rem;
    white-space: normal;
    overflow-wrap: anywhere;
    opacity: 0.88;
  }
</style>
"""

st.markdown(LIGHT_CSS if THEME == "Light" else DARK_CSS, unsafe_allow_html=True)

# =========================================================
# Helpers (robust)
# =========================================================
def norm_text(x: str) -> str:
    s = str(x) if x is not None else ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9% ]", "", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\xa0", " ")).strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    nmap = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        key = norm_text(cand)
        if key in nmap:
            return nmap[key]
    df_norm = [(norm_text(c), c) for c in df.columns]
    for cand in candidates:
        ck = norm_text(cand)
        for dn, real in df_norm:
            if ck and ck in dn:
                return real
    return None

def to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.replace("\xa0", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")

def normalize_percent_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mx = s.max(skipna=True)
    if pd.notna(mx) and mx <= 1:
        return s * 100
    return s

def safe_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def shift_from_hour(h: int) -> str:
    if 6 <= h < 14:
        return "Shift A (06-14)"
    if 14 <= h < 22:
        return "Shift B (14-22)"
    return "Shift C (22-06)"

def area_from_dp(dp: str) -> str:
    s = (dp or "").upper()
    if "PACK" in s or "PACKOUT" in s:
        return "PACK"
    if "DECANT" in s:
        return "DECANT"
    if "WAXFER" in s:
        return "WAXFER"
    if "GTP" in s:
        return "GTP"
    if "RETAIL" in s:
        return "RETAIL"
    if "SHUTTLE" in s or "KNAPP" in s:
        return "SHUTTLE"
    return "OTHER"

def safe_percent_str(x):
    return f"{x:.2f}%" if pd.notna(x) else "—"

def fig_style(fig, title=None, subtitle=None, x_title=None, y_title=None):
    if THEME == "Light":
        font_color = "#0F172A"
        grid_color = "rgba(15, 23, 42, 0.12)"
        axis_color = "rgba(15, 23, 42, 0.45)"
        hover_bg = "white"
        hover_font = "#0F172A"
    else:
        font_color = "rgba(255,255,255,0.92)"
        grid_color = "rgba(255,255,255,0.14)"
        axis_color = "rgba(255,255,255,0.55)"
        hover_bg = "rgba(10,16,32,0.92)"
        hover_font = "white"

    if title and subtitle:
        fig.update_layout(
            title=dict(
                text=f"{title}<br><span style='font-size:13px; opacity:0.75'>{subtitle}</span>",
                x=0.01, xanchor="left",
            )
        )
    elif title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"))

    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)

    fig.update_layout(
        margin=dict(l=14, r=14, t=78, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", size=13, color=font_color),
        title_font=dict(size=22, color=font_color),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=font_color)),
        hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font)),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, linecolor=axis_color,
                     tickfont=dict(size=12, color=font_color), title_font=dict(size=16, color=font_color))
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, linecolor=axis_color,
                     tickfont=dict(size=12, color=font_color), title_font=dict(size=16, color=font_color))
    return fig

def percent_col_lookup(df: pd.DataFrame, wanted: str) -> str | None:
    w = norm_text(wanted).replace(" ", "")
    for c in df.columns:
        if c in ["Decision Point", "Area"]:
            continue
        cc = norm_text(c).replace(" ", "").replace("%", "")
        if w == cc:
            return c
    for c in df.columns:
        if c in ["Decision Point", "Area"]:
            continue
        cc = norm_text(c).replace(" ", "").replace("%", "")
        if w and w in cc:
            return c
    return None

def rank_scanners_by_exception_pct(exc_pct_df: pd.DataFrame, exc_col: str, ignore_blanks: bool, top_n: int):
    if exc_col not in exc_pct_df.columns:
        return pd.DataFrame(columns=["Decision Point", "Rate %"]), pd.DataFrame(columns=["Decision Point", "Rate %"])
    r = exc_pct_df[["Decision Point", exc_col]].copy()
    r = r.rename(columns={exc_col: "Rate %"})
    r["Rate %"] = normalize_percent_series(r["Rate %"])
    if ignore_blanks:
        r = r.dropna(subset=["Rate %"])
    else:
        r["Rate %"] = r["Rate %"].fillna(0)
    r = r.sort_values("Rate %", ascending=False)
    return r.head(top_n).copy(), r.copy()

# =========================================================
# Severity colours (use for ALL charts except:
# - Carrier brand colours (EVRI/DPD/Parcel Force)
# - Area colours (Area categories)
# =========================================================
SEVERITY = {
    "RED": "#EF4444",
    "ORANGE": "#F97316",
    "YELLOW": "#F59E0B",
    "LIGHT_GREEN": "#A3E635",
    "GREEN": "#22C55E",
    "GREY": "#94A3B8",
}

def severity_bucket_color(rank: int, n: int) -> str:
    """
    Map rank (0=highest) to severity colours.
    Uses buckets by percentile so it scales with any chart length.
    """
    if n <= 1:
        return SEVERITY["RED"]
    p = rank / (n - 1)  # 0..1
    if p <= 0.10:
        return SEVERITY["RED"]
    if p <= 0.30:
        return SEVERITY["ORANGE"]
    if p <= 0.55:
        return SEVERITY["YELLOW"]
    if p <= 0.75:
        return SEVERITY["LIGHT_GREEN"]
    if p <= 0.92:
        return SEVERITY["GREEN"]
    return SEVERITY["GREY"]

def apply_severity_bar_colors(fig, ordered_values: list[float]):
    """Apply severity colours to a single-trace bar chart, based on descending order."""
    n = len(ordered_values)
    colors = [severity_bucket_color(i, n) for i in range(n)]
    fig.update_traces(marker=dict(color=colors))
    return fig

# =========================================================
# Brand carrier colours + normalization (KEEP)
# =========================================================
BRAND_COLORS = {
    "EVRI": "#007BC4",
    "DPD": "#DC0032",
    "PARCEL FORCE": "#ED2929",
    "PARCELFORCE": "#ED2929",
    "PARCEL FORCE WORLDWIDE": "#ED2929",
}

def canonical_carrier(x: str) -> str:
    s = (str(x) if x is not None else "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    if s in ["PARCELFORCE", "PARCEL FORCE WORLDWIDE"]:
        return "PARCEL FORCE"
    if s.startswith("DPD"):
        return "DPD"
    if s.startswith("EVRI"):
        return "EVRI"
    if s.startswith("PARCEL FORCE"):
        return "PARCEL FORCE"
    return s

# =========================================================
# Upload Excel
# =========================================================
st.sidebar.header("📤 Upload")
uploaded = st.sidebar.file_uploader("Upload your Excel workbook", type=["xlsx", "xlsm", "xls"])

file_label = uploaded.name if uploaded else "No file uploaded"
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1 style="margin:0;">🐾 WES Ops Dashboard</h1>
        <div class="subtitle">Core Scanner KPIs + Executive PowerBI-style dashboard</div>
      </div>
      <div class="pill">📄 File: <b style="margin-left:6px;">{file_label}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not uploaded:
    st.info("Upload your Excel file from the sidebar to load the dashboard.")
    st.stop()

file_bytes = uploaded.getvalue()

@st.cache_data(show_spinner=False)
def get_sheet_map(file_bytes_: bytes) -> dict:
    xls = pd.ExcelFile(BytesIO(file_bytes_))
    return {s.lower().strip(): s for s in xls.sheet_names}

@st.cache_data(show_spinner=False)
def read_sheet(file_bytes_: bytes, sheet_name_exact: str) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(file_bytes_), sheet_name=sheet_name_exact)
    return normalize_columns(df)

sheet_map = get_sheet_map(file_bytes)
SEQ = ["total", "exception pct", "overall pct", "shipping lanes", "carrier", "scan", "errors", "mission error"]
missing = [s for s in SEQ if s not in sheet_map]
if missing:
    st.error(f"Your workbook is missing these required sheets: {missing}")
    st.write("Sheets found:", list(sheet_map.values()))
    st.stop()

df_total = read_sheet(file_bytes, sheet_map["total"])
df_exc_pct = read_sheet(file_bytes, sheet_map["exception pct"])
df_overall_raw = read_sheet(file_bytes, sheet_map["overall pct"])
df_ship = read_sheet(file_bytes, sheet_map["shipping lanes"])
df_carrier = read_sheet(file_bytes, sheet_map["carrier"])
df_scan_raw = read_sheet(file_bytes, sheet_map["scan"])
df_errors = read_sheet(file_bytes, sheet_map["errors"])
df_mission = read_sheet(file_bytes, sheet_map["mission error"])

# =========================================================
# total (numeric)
# =========================================================
df_total_num = df_total.copy()
for c in df_total_num.columns:
    df_total_num[c] = to_num(df_total_num[c])

# =========================================================
# exception pct
# =========================================================
col_dp_exc = find_col(df_exc_pct, ["Decision Point", "DecisionPoint", "Scanner", "DP"])
if col_dp_exc is None:
    st.error("exception pct sheet: could not find 'Decision Point' column.")
    st.write("Columns:", df_exc_pct.columns.tolist())
    st.stop()

df_exc_pct = df_exc_pct.rename(columns={col_dp_exc: "Decision Point"})
df_exc_pct["Decision Point"] = df_exc_pct["Decision Point"].astype(str).str.strip()

pct_cols = [c for c in df_exc_pct.columns if c != "Decision Point"]
for c in pct_cols:
    df_exc_pct[c] = normalize_percent_series(to_num(df_exc_pct[c]))

df_exc_pct["Area"] = df_exc_pct["Decision Point"].apply(area_from_dp)

# =========================================================
# overall pct
# =========================================================
df_overall = df_overall_raw.copy()

col_dp = find_col(df_overall, ["Decision Point", "DecisionPoint", "Scanner", "DP"])
col_scans = find_col(df_overall, ["Total Scans by Scanner", "Total Scans", "Total Container Scan", "Total Container Scans", "Container Scans"])
col_excs = find_col(df_overall, ["Total Container Exceptions", "Total Exceptions", "Total Container Exception", "Container Exceptions", "Exceptions"])
col_rate = find_col(df_overall, ["Exception Rate %", "Exception Rate", "Exception Rate (%)", "Exception %"])
col_read = find_col(df_overall, ["Scanner Read %", "Scanner Read", "Read %", "Read Rate %", "Read Rate"])

need = []
if col_dp is None: need.append("Decision Point")
if col_scans is None: need.append("Total Scans")
if col_excs is None: need.append("Total Exceptions")
if need:
    st.error(f"overall pct sheet: could not detect columns: {need}")
    st.write("Columns:", df_overall.columns.tolist())
    st.stop()

rename_map = {col_dp: "Decision Point", col_scans: "Total Scans", col_excs: "Total Exceptions"}
if col_rate: rename_map[col_rate] = "Exception Rate %"
if col_read: rename_map[col_read] = "Scanner Read %"
df_overall = df_overall.rename(columns=rename_map)

df_overall["Decision Point"] = df_overall["Decision Point"].astype(str).str.strip()
df_overall["Total Scans"] = to_num(df_overall["Total Scans"])
df_overall["Total Exceptions"] = to_num(df_overall["Total Exceptions"])

if "Exception Rate %" in df_overall.columns:
    df_overall["Exception Rate %"] = normalize_percent_series(to_num(df_overall["Exception Rate %"]))
else:
    df_overall["Exception Rate %"] = np.where(
        df_overall["Total Scans"] > 0,
        (df_overall["Total Exceptions"] / df_overall["Total Scans"]) * 100,
        np.nan
    )

if "Scanner Read %" in df_overall.columns:
    df_overall["Scanner Read %"] = normalize_percent_series(to_num(df_overall["Scanner Read %"]))
else:
    df_overall["Scanner Read %"] = np.nan

df_overall["Area"] = df_overall["Decision Point"].apply(area_from_dp)

# =========================================================
# shipping lanes (counts + % contribution)
# =========================================================
col_lane = find_col(df_ship, ["SHIP LANE", "Ship Lane", "Lane", "Shipping Lane"])
col_car = find_col(df_ship, ["CARRIER", "Carrier", "Service", "Carrier Service"])
col_total = find_col(df_ship, ["TOTAL BOXES", "Total Boxes", "Boxes", "Total"])
col_bph = find_col(df_ship, ["BOXES per HOUR", "Boxes per Hour", "Boxes/Hr", "BPH"])
col_pct = find_col(df_ship, ["PERCENTAGE", "Percentage", "%", "Percent"])

if col_lane: df_ship = df_ship.rename(columns={col_lane: "Lane"})
else: df_ship["Lane"] = "UNKNOWN"
if col_car: df_ship = df_ship.rename(columns={col_car: "Carrier"})
else: df_ship["Carrier"] = "UNKNOWN"
if col_total: df_ship = df_ship.rename(columns={col_total: "Total Boxes"})
else: df_ship["Total Boxes"] = np.nan
if col_bph: df_ship = df_ship.rename(columns={col_bph: "Boxes per Hour"})
else: df_ship["Boxes per Hour"] = np.nan
if col_pct: df_ship = df_ship.rename(columns={col_pct: "Percentage"})
else: df_ship["Percentage"] = np.nan

df_ship["Lane"] = df_ship["Lane"].astype(str).str.strip()
df_ship["Carrier"] = df_ship["Carrier"].astype(str).str.strip().apply(canonical_carrier)
df_ship["Total Boxes"] = to_num(df_ship["Total Boxes"])
df_ship["Boxes per Hour"] = to_num(df_ship["Boxes per Hour"])
df_ship["Percentage"] = normalize_percent_series(to_num(df_ship["Percentage"]))

total_boxes_all = float(np.nansum(df_ship["Total Boxes"].values)) if "Total Boxes" in df_ship.columns else 0.0
df_ship["% of Total Boxes (Derived)"] = (df_ship["Total Boxes"] / total_boxes_all) * 100 if total_boxes_all > 0 else np.nan

# =========================================================
# carrier (best-effort numeric)
# =========================================================
col_car2 = find_col(df_carrier, ["CARRIER Service", "Carrier", "CARRIER", "Service"])
if col_car2:
    df_carrier = df_carrier.rename(columns={col_car2: "Carrier"})
else:
    df_carrier["Carrier"] = "UNKNOWN"
df_carrier["Carrier"] = df_carrier["Carrier"].astype(str).str.strip().apply(canonical_carrier)

car_total = find_col(df_carrier, ["TOTAL BOXES", "Total Boxes", "Boxes"])
car_bph = find_col(df_carrier, ["Boxes per Hour", "BOXES per HOUR", "BPH"])
car_pct = find_col(df_carrier, ["Percentage", "PERCENTAGE", "%"])

if car_total and car_total != "TOTAL BOXES":
    df_carrier = df_carrier.rename(columns={car_total: "TOTAL BOXES"})
if "TOTAL BOXES" in df_carrier.columns:
    df_carrier["TOTAL BOXES"] = to_num(df_carrier["TOTAL BOXES"])

if car_bph and car_bph != "Boxes per Hour":
    df_carrier = df_carrier.rename(columns={car_bph: "Boxes per Hour"})
if "Boxes per Hour" in df_carrier.columns:
    df_carrier["Boxes per Hour"] = to_num(df_carrier["Boxes per Hour"])

if car_pct and car_pct != "Percentage":
    df_carrier = df_carrier.rename(columns={car_pct: "Percentage"})
if "Percentage" in df_carrier.columns:
    df_carrier["Percentage"] = normalize_percent_series(to_num(df_carrier["Percentage"]))

# =========================================================
# scan
# =========================================================
df_scan = df_scan_raw.copy()
col_dp_scan = find_col(df_scan, ["Decision Point", "DecisionPoint", "Scanner", "DP"])
col_exc_type = find_col(df_scan, ["Exception Type", "Exception", "Error Type", "ExceptionType"])
col_updated = find_col(df_scan, ["Updated", "Updated Time", "Timestamp", "Time", "UpdatedTime"])

if col_dp_scan is None or col_exc_type is None or col_updated is None:
    st.error("scan sheet: missing key columns (Decision Point / Exception Type / Updated).")
    st.write("Columns:", df_scan.columns.tolist())
    st.stop()

df_scan = df_scan.rename(columns={col_dp_scan: "Decision Point", col_exc_type: "Exception Type", col_updated: "Updated"})
df_scan["Decision Point"] = df_scan["Decision Point"].astype(str).str.strip()
df_scan["Exception Type"] = df_scan["Exception Type"].astype(str).str.strip().str.upper()
df_scan["Updated"] = safe_dt(df_scan["Updated"])
df_scan = df_scan.dropna(subset=["Updated", "Decision Point", "Exception Type"]).copy()

df_scan["Date"] = df_scan["Updated"].dt.date
df_scan["Hour"] = df_scan["Updated"].dt.hour
df_scan["Shift"] = df_scan["Hour"].apply(lambda x: shift_from_hour(int(x)) if pd.notna(x) else "Unknown")
df_scan["Area"] = df_scan["Decision Point"].apply(area_from_dp)

# detect tote columns if present
col_source_tote = find_col(df_scan, ["Source TOTE", "Source Tote", "Source"])
col_dest_tote = find_col(df_scan, ["Destination TOTE", "Destination Tote", "Destination"])
col_dup_data = find_col(df_scan, ["Duplicate Data", "Duplicate", "Column S", "S"])

# =========================================================
# Sidebar filters
# =========================================================
st.sidebar.header("🎛 Filters")

areas_all = sorted(df_exc_pct["Area"].dropna().unique().tolist())
areas = st.sidebar.multiselect("Area", areas_all, default=areas_all)

dp_all = sorted(df_exc_pct["Decision Point"].dropna().unique().tolist())
selected_dps = st.sidebar.multiselect("Decision Points (Scanners)", dp_all, default=dp_all)

ignore_blanks = st.sidebar.checkbox("Ranking: ignore blanks (recommended)", value=True)
top_n = st.sidebar.slider("Top N", 5, 50, 15)
rank_n = st.sidebar.slider("Ranking bars (Top scanners)", 5, 50, 20)

date_min = df_scan["Updated"].min().date()
date_max = df_scan["Updated"].max().date()
date_range = st.sidebar.date_input("Date range (scan-based)", value=(date_min, date_max))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

exc_pct_f = df_exc_pct[df_exc_pct["Area"].isin(areas) & df_exc_pct["Decision Point"].isin(selected_dps)].copy()
overall_f = df_overall[df_overall["Area"].isin(areas) & df_overall["Decision Point"].isin(selected_dps)].copy()
scan_f = df_scan[
    (df_scan["Date"] >= start_date)
    & (df_scan["Date"] <= end_date)
    & (df_scan["Area"].isin(areas))
    & (df_scan["Decision Point"].isin(selected_dps))
].copy()

# =========================================================
# Core Scanner KPI CALCS
# =========================================================
noread_col = percent_col_lookup(exc_pct_f, "NOREAD")
noscan_col = percent_col_lookup(exc_pct_f, "NOSCAN")

def top_scanner_value(df, col):
    if not col or col not in df.columns or df.empty:
        return ("—", np.nan)
    tmp = df[["Decision Point", col]].copy()
    tmp[col] = normalize_percent_series(tmp[col])
    if ignore_blanks:
        tmp = tmp.dropna(subset=[col])
    else:
        tmp[col] = tmp[col].fillna(0)
    if tmp.empty:
        return ("—", np.nan)
    tmp = tmp.sort_values(col, ascending=False)
    return (tmp["Decision Point"].iloc[0], float(tmp[col].iloc[0]) if pd.notna(tmp[col].iloc[0]) else np.nan)

def avg_pct(df, col):
    if not col or col not in df.columns or df.empty:
        return np.nan
    s = pd.to_numeric(normalize_percent_series(df[col]), errors="coerce")
    return float(np.nanmean(s)) if s.notna().any() else np.nan

def scanners_affected(df, col):
    if not col or col not in df.columns or df.empty:
        return 0
    s = pd.to_numeric(normalize_percent_series(df[col]), errors="coerce").fillna(0)
    return int((s > 0).sum())

top_noread_dp, top_noread_val = top_scanner_value(exc_pct_f, noread_col)
top_noscan_dp, top_noscan_val = top_scanner_value(exc_pct_f, noscan_col)

avg_noread = avg_pct(exc_pct_f, noread_col)
avg_noscan = avg_pct(exc_pct_f, noscan_col)

aff_noread = scanners_affected(exc_pct_f, noread_col)
aff_noscan = scanners_affected(exc_pct_f, noscan_col)

site_scans = float(np.nansum(overall_f["Total Scans"].values)) if "Total Scans" in overall_f.columns else 0.0
site_excs = float(np.nansum(overall_f["Total Exceptions"].values)) if "Total Exceptions" in overall_f.columns else 0.0
site_rate = (site_excs / site_scans) * 100 if site_scans else np.nan

avg_read = float(np.nanmean(overall_f["Scanner Read %"].values)) if "Scanner Read %" in overall_f.columns and len(overall_f) else np.nan

worst_dp = "—"
worst_rate = np.nan
if "Exception Rate %" in overall_f.columns and len(overall_f):
    wr = overall_f.sort_values("Exception Rate %", ascending=False).head(1)
    if len(wr):
        worst_dp = wr["Decision Point"].iloc[0]
        worst_rate = float(wr["Exception Rate %"].iloc[0]) if pd.notna(wr["Exception Rate %"].iloc[0]) else np.nan

scanner_count_selected = int(exc_pct_f["Decision Point"].nunique()) if "Decision Point" in exc_pct_f.columns else 0
areas_count = len(areas)

# =========================================================
# TOP: Core Scanner KPIs
# =========================================================
st.markdown('<div class="section">Core Scanner KPIs</div>', unsafe_allow_html=True)

r1 = st.columns(6)
r1[0].metric("Top NOREAD %", safe_percent_str(top_noread_val), delta=f"Scanner: {top_noread_dp}")
r1[1].metric("Top NOSCAN %", safe_percent_str(top_noscan_val), delta=f"Scanner: {top_noscan_dp}")
r1[2].metric("Selected Exception Rate", safe_percent_str(site_rate))
r1[3].metric("Average Scanner Read %", safe_percent_str(avg_read))
r1[4].metric("Worst exception rate %", safe_percent_str(worst_rate), delta=f"{worst_dp}")
r1[5].metric("Selected scanners", f"{scanner_count_selected:,}", delta=f"Areas: {areas_count:,}")

r2 = st.columns(6)
r2[0].metric("Avg NOREAD %", safe_percent_str(avg_noread))
r2[1].metric("Scanners with NOREAD > 0", f"{aff_noread:,}")
r2[2].metric("Avg NOSCAN %", safe_percent_str(avg_noscan))
r2[3].metric("Scanners with NOSCAN > 0", f"{aff_noscan:,}")
r2[4].metric("Total scans (selected)", f"{site_scans:,.0f}")
r2[5].metric("Total exceptions (selected)", f"{site_excs:,.0f}")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================================================
# Tabs (added Executive Dashboard)
# =========================================================
tabs = st.tabs([
    "📊 Executive Dashboard (PowerBI-style)",
    "📌 total",
    "🎯 exception pct (MAIN)",
    "📷 overall pct",
    "🚚 shipping lanes",
    "🧾 carrier",
    "🧪 scan",
    "🧯 errors",
    "🛠 mission error",
])

# =========================================================
# TAB 0: Executive Dashboard (PowerBI-style)
# =========================================================
with tabs[0]:
    st.markdown('<div class="section">Executive Dashboard (PowerBI-style)</div>', unsafe_allow_html=True)
    st.caption("High-level view: last 24h shift split + most affected totes + ship lane % share. Carrier/Area colours preserved.")

    if scan_f.empty:
        st.warning("No scan rows match filters.")
    else:
        # Last 24h window from latest timestamp in filtered scan data
        tmax = scan_f["Updated"].max()
        tmin_24 = tmax - pd.Timedelta(hours=24)
        scan_24 = scan_f[scan_f["Updated"] >= tmin_24].copy()
        total_24 = len(scan_24)

        # --- Shift donut data ---
        shift_counts = scan_24["Shift"].value_counts(dropna=False).to_dict()
        a_cnt = int(shift_counts.get("Shift A (06-14)", 0))
        b_cnt = int(shift_counts.get("Shift B (14-22)", 0))
        c_cnt = int(shift_counts.get("Shift C (22-06)", 0))

        a_pct = (a_cnt / total_24) * 100 if total_24 else 0
        b_pct = (b_cnt / total_24) * 100 if total_24 else 0
        c_pct = (c_cnt / total_24) * 100 if total_24 else 0

        # --- Duplicate Data (Column S) check ---
        dup_count_rows = 0
        dup_unique_values = 0
        if col_dup_data and col_dup_data in scan_24.columns:
            s = scan_24[col_dup_data].replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
            if len(s):
                dup_mask = s.duplicated(keep=False)
                dup_vals = s[dup_mask]
                dup_count_rows = int(dup_vals.shape[0])
                dup_unique_values = int(dup_vals.nunique())

        # --- Tote issues (most repeated source/destination) ---
        top_source = ("—", 0)
        top_dest = ("—", 0)
        top_any = ("—", 0)

        if col_source_tote and col_source_tote in scan_24.columns:
            src = scan_24[col_source_tote].replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
            if len(src):
                vc = src.value_counts()
                top_source = (str(vc.index[0]), int(vc.iloc[0]))

        if col_dest_tote and col_dest_tote in scan_24.columns:
            dst = scan_24[col_dest_tote].replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
            if len(dst):
                vc = dst.value_counts()
                top_dest = (str(vc.index[0]), int(vc.iloc[0]))

        if (col_source_tote in scan_24.columns) or (col_dest_tote in scan_24.columns):
            parts = []
            if col_source_tote and col_source_tote in scan_24.columns:
                parts.append(scan_24[col_source_tote])
            if col_dest_tote and col_dest_tote in scan_24.columns:
                parts.append(scan_24[col_dest_tote])
            all_totes = pd.concat(parts, ignore_index=True).replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
            if len(all_totes):
                vc = all_totes.value_counts()
                top_any = (str(vc.index[0]), int(vc.iloc[0]))

        # --- Shipping lane % share (derived) for current carrier filter ---
        ship_total = float(np.nansum(df_ship["Total Boxes"].values)) if "Total Boxes" in df_ship.columns else 0.0
        ship_df = df_ship.copy()
        ship_df["% share (Derived)"] = (ship_df["Total Boxes"] / ship_total) * 100 if ship_total > 0 else np.nan

        # KPI row (clean)
        k = st.columns(6)
        k[0].metric("Last 24h exceptions", f"{total_24:,}", delta=f"{tmin_24:%d %b %H:%M} → {tmax:%d %b %H:%M}")
        k[1].metric("Shift A %", safe_percent_str(a_pct), delta=f"{a_cnt:,} rows")
        k[2].metric("Shift B %", safe_percent_str(b_pct), delta=f"{b_cnt:,} rows")
        k[3].metric("Shift C %", safe_percent_str(c_pct), delta=f"{c_cnt:,} rows")
        k[4].metric("Column S duplicate rows", f"{dup_count_rows:,}", delta=f"Unique dup values: {dup_unique_values:,}")
        k[5].metric("Most affected tote", f"{top_any[0]}", delta=f"{top_any[1]:,} rows")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Row 1: Shift donut + Ship lane % share (brand colours preserved)
        c1, c2 = st.columns(2)

        shift_df = pd.DataFrame({
            "Shift": ["Shift A (06-14)", "Shift B (14-22)", "Shift C (22-06)"],
            "Count": [a_cnt, b_cnt, c_cnt]
        })
        donut = go.Figure(data=[go.Pie(labels=shift_df["Shift"], values=shift_df["Count"], hole=0.58, textinfo="percent+label")])
        c1.plotly_chart(
            fig_style(donut, title="Last 24h: Shift % split", subtitle="Share of exceptions by shift."),
            use_container_width=True
        )

        # Ship lane % share chart (carrier brand colours preserved)
        if ship_df["% share (Derived)"].notna().any():
            top_lanes_pct = ship_df.sort_values("% share (Derived)", ascending=False).head(12)
            fig_lane_pct = px.bar(
                top_lanes_pct,
                x="Lane",
                y="% share (Derived)",
                color="Carrier",
                text="% share (Derived)",
                color_discrete_map=BRAND_COLORS  # KEEP brand colours
            )
            fig_lane_pct.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_lane_pct.update_layout(xaxis_tickangle=-25)
            c2.plotly_chart(
                fig_style(fig_lane_pct,
                          title="Top ship lanes by % share (Derived)",
                          subtitle="Lane contribution % (not boxes). Brand carrier colours preserved.",
                          x_title="Lane", y_title="% share"),
                use_container_width=True
            )
        else:
            c2.info("Shipping lanes: unable to calculate % share (Derived).")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Row 2: Most repeated Source/Destination tote + Most affected totes (severity colours)
        d1, d2 = st.columns(2)

        # KPI cards for source/destination tote
        kk = d1.columns(2)
        kk[0].metric("Top Source tote", top_source[0], delta=f"{top_source[1]:,} rows")
        kk[1].metric("Top Destination tote", top_dest[0], delta=f"{top_dest[1]:,} rows")

        # Bar: top totes (combined source + destination) with severity colours
        if top_any[0] != "—":
            all_totes_vc = all_totes.value_counts().head(15).reset_index()
            all_totes_vc.columns = ["Tote", "Rows"]
            all_totes_vc = all_totes_vc.sort_values("Rows", ascending=False)

            fig_totes = px.bar(all_totes_vc, x="Tote", y="Rows", text="Rows")
            fig_totes.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig_totes.update_layout(xaxis_tickangle=-25)
            fig_totes = apply_severity_bar_colors(fig_totes, all_totes_vc["Rows"].tolist())

            d2.plotly_chart(
                fig_style(fig_totes,
                          title="Most affected totes (Source + Destination)",
                          subtitle="Severity colours: red=highest repeated.",
                          x_title="Tote", y_title="Rows"),
                use_container_width=True
            )
        else:
            d2.info("No tote columns found (Source TOTE / Destination TOTE).")

        with st.expander("Show last 24h tote table (top 200 rows)"):
            show_cols = [c for c in ["Updated", "Decision Point", "Exception Type", col_source_tote, col_dest_tote, col_dup_data] if c and c in scan_24.columns]
            st.dataframe(scan_24[show_cols].head(200), use_container_width=True)

# =========================================================
# TAB 1: total
# =========================================================
with tabs[1]:
    st.markdown('<div class="section">total</div>', unsafe_allow_html=True)
    st.caption("High-level totals from your “total” sheet.")
    st.dataframe(df_total, use_container_width=True)

# =========================================================
# TAB 2: exception pct (MAIN)
# =========================================================
with tabs[2]:
    st.markdown('<div class="section">exception pct (main focus)</div>', unsafe_allow_html=True)
    st.caption("Identify scanners with high NOREAD% and NOSCAN%, and rank scanners by any exception type.")

    if exc_pct_f.empty:
        st.warning("No rows match the selected Area/Decision Point filters.")
    else:
        pct_cols2 = [c for c in exc_pct_f.columns if c not in ["Decision Point", "Area"]]
        if not pct_cols2:
            st.warning("No exception % columns found in exception pct sheet.")
        else:
            st.markdown("### NOREAD & NOSCAN scanner ranking (highest → lowest)")
            colA, colB = st.columns(2)

            def show_top_rank(container, label):
                exc_col = percent_col_lookup(exc_pct_f, label)
                if not exc_col:
                    container.warning(f"{label} column not found.")
                    return
                top_df, full_rank = rank_scanners_by_exception_pct(exc_pct_f, exc_col, ignore_blanks, rank_n)
                fig = px.bar(top_df, x="Decision Point", y="Rate %", text="Rate %")
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
                fig.update_layout(xaxis_tickangle=-25)
                fig = apply_severity_bar_colors(fig, top_df["Rate %"].tolist())  # severity colours
                container.plotly_chart(
                    fig_style(fig,
                              title=f"Top {rank_n} scanners by {label} %",
                              subtitle="Worst scanners first (severity colours).",
                              x_title="Scanner (Decision Point)",
                              y_title=f"{label} %"),
                    use_container_width=True
                )
                with container.expander(f"Show full ranking table ({label})"):
                    container.dataframe(full_rank, use_container_width=True)

            show_top_rank(colA, "NOREAD")
            show_top_rank(colB, "NOSCAN")

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            st.markdown("### Pick an exception type and see scanners ranked")
            display_map = {c: str(c).replace("%", "").strip().upper() for c in pct_cols2}
            reverse_map = {v: k for k, v in display_map.items()}
            default_choice = "NOREAD" if "NOREAD" in reverse_map else sorted(reverse_map.keys())[0]

            exc_choice = st.selectbox("Exception Type", options=sorted(reverse_map.keys()),
                                      index=sorted(reverse_map.keys()).index(default_choice))

            exc_col = reverse_map[exc_choice]
            top_df, full_rank = rank_scanners_by_exception_pct(exc_pct_f, exc_col, ignore_blanks, rank_n)

            if full_rank.empty:
                st.warning("No data for selected exception type.")
            else:
                fig_any = px.bar(top_df, x="Decision Point", y="Rate %", text="Rate %")
                fig_any.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
                fig_any.update_layout(xaxis_tickangle=-25)
                fig_any = apply_severity_bar_colors(fig_any, top_df["Rate %"].tolist())  # severity colours
                st.plotly_chart(
                    fig_style(fig_any,
                              title=f"Top {rank_n} scanners by {exc_choice} %",
                              subtitle="Highest to lowest (severity colours).",
                              x_title="Scanner (Decision Point)",
                              y_title="Rate %"),
                    use_container_width=True
                )

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            st.markdown("### Exception type summary (average %, blanks ignored)")
            long = exc_pct_f.melt(id_vars=["Decision Point", "Area"], value_vars=pct_cols2,
                                 var_name="Exception Type", value_name="Rate %")
            long["Exception Type"] = long["Exception Type"].astype(str).str.replace("%", "", regex=False).str.strip().str.upper()
            long["Rate %"] = normalize_percent_series(pd.to_numeric(long["Rate %"], errors="coerce"))

            top_types = (
                long.groupby("Exception Type", as_index=False)["Rate %"].mean()
                .sort_values("Rate %", ascending=False).head(top_n)
            )

            fig_types = px.bar(top_types, x="Exception Type", y="Rate %", text="Rate %")
            fig_types.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_types.update_layout(xaxis_tickangle=-25)
            fig_types = apply_severity_bar_colors(fig_types, top_types["Rate %"].tolist())  # severity colours
            st.plotly_chart(
                fig_style(fig_types,
                          title=f"Top {top_n} exception types by average %",
                          subtitle="Severity colours: red=highest average.",
                          x_title="Exception Type",
                          y_title="Average Rate %"),
                use_container_width=True
            )

            st.dataframe(exc_pct_f.drop(columns=["Area"], errors="ignore"), use_container_width=True)

# =========================================================
# TAB 3: overall pct (Area colours preserved)
# =========================================================
with tabs[3]:
    st.markdown('<div class="section">overall pct</div>', unsafe_allow_html=True)
    st.caption("Overall scanner statistics: exception rate %, scanner read %, and volume impact. (Area colours preserved.)")

    if overall_f.empty:
        st.warning("No scanner rows match filters.")
    else:
        c1, c2 = st.columns(2)
        top_rate = overall_f.sort_values("Exception Rate %", ascending=False).head(top_n)

        # KEEP Area colours (do not apply severity colours here)
        fig_rate = px.bar(top_rate, x="Decision Point", y="Exception Rate %", text="Exception Rate %", color="Area")
        fig_rate.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig_rate.update_layout(xaxis_tickangle=-30)
        c1.plotly_chart(fig_style(fig_rate, title=f"Top {top_n} scanners by exception rate % (Area colours)",
                                  subtitle="Area colours preserved.",
                                  x_title="Scanner", y_title="Exception Rate %"),
                        use_container_width=True)

        # KEEP Area colours
        fig_scatter = px.scatter(overall_f, x="Total Scans", y="Exception Rate %",
                                 size="Total Exceptions", color="Area", hover_name="Decision Point")
        c2.plotly_chart(fig_style(fig_scatter, title="Impact view: Rate vs Volume (Area colours)",
                                  subtitle="High rate + high volume = biggest impact.",
                                  x_title="Total Scans", y_title="Exception Rate %"),
                        use_container_width=True)

        # Read % (Area colours)
        if "Scanner Read %" in overall_f.columns and overall_f["Scanner Read %"].notna().any():
            top_read_bad = overall_f.sort_values("Scanner Read %", ascending=True).head(top_n)
            fig_read = px.bar(top_read_bad, x="Decision Point", y="Scanner Read %", text="Scanner Read %", color="Area")
            fig_read.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_read.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_style(fig_read, title=f"Lowest {top_n} scanners by Scanner Read % (Area colours)",
                                      subtitle="Area colours preserved.",
                                      x_title="Scanner", y_title="Scanner Read %"),
                            use_container_width=True)

        show_cols = ["Area", "Decision Point", "Total Scans", "Total Exceptions", "Exception Rate %", "Scanner Read %"]
        show_cols = [c for c in show_cols if c in overall_f.columns]
        st.dataframe(overall_f[show_cols], use_container_width=True)

# =========================================================
# TAB 4: shipping lanes (Carrier brand colours preserved)
# =========================================================
with tabs[4]:
    st.markdown('<div class="section">shipping lanes</div>', unsafe_allow_html=True)
    st.caption("Lane throughput and contribution. Carrier brand colours preserved.")

    carriers = sorted(df_ship["Carrier"].dropna().unique().tolist()) if "Carrier" in df_ship.columns else []
    pick_carriers = st.multiselect("Carrier", carriers, default=carriers, key="ship_carrier") if carriers else []
    sf = df_ship[df_ship["Carrier"].isin(pick_carriers)].copy() if carriers else df_ship.copy()

    total_boxes_filtered = float(np.nansum(sf["Total Boxes"].values)) if "Total Boxes" in sf.columns else 0.0
    sf["% of Total Boxes (Derived)"] = (sf["Total Boxes"] / total_boxes_filtered) * 100 if total_boxes_filtered > 0 else np.nan

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Lanes", f"{sf['Lane'].nunique():,}" if "Lane" in sf.columns else "—")
    k2.metric("Carriers", f"{sf['Carrier'].nunique():,}" if "Carrier" in sf.columns else "—")
    k3.metric("Total Boxes", f"{np.nansum(sf['Total Boxes'].values):,.0f}" if "Total Boxes" in sf.columns else "—")
    k4.metric("Total Boxes/Hour", f"{np.nansum(sf['Boxes per Hour'].values):,.2f}" if "Boxes per Hour" in sf.columns else "—")

    top_lane = "—"
    top_lane_pct = np.nan
    if "Lane" in sf.columns and sf["% of Total Boxes (Derived)"].notna().any():
        t = sf.sort_values("% of Total Boxes (Derived)", ascending=False).head(1)
        top_lane = t["Lane"].iloc[0]
        top_lane_pct = float(t["% of Total Boxes (Derived)"].iloc[0])
    k5.metric("Top lane % share", safe_percent_str(top_lane_pct), help=f"Lane: {top_lane}")
    k6.metric("Top lane name", top_lane)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    if "Total Boxes" in sf.columns and "Lane" in sf.columns:
        top_vol = sf.sort_values("Total Boxes", ascending=False).head(top_n)
        fig_vol = px.bar(
            top_vol,
            x="Lane",
            y="Total Boxes",
            color="Carrier",
            text="Total Boxes",
            color_discrete_map=BRAND_COLORS  # KEEP brand colours
        )
        fig_vol.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        fig_vol.update_layout(xaxis_tickangle=-25)
        c1.plotly_chart(fig_style(fig_vol, title=f"Top {top_n} lanes by total boxes (brand colours)",
                                  subtitle="Brand colours preserved for carriers.",
                                  x_title="Lane", y_title="Total Boxes"),
                        use_container_width=True)

    if "% of Total Boxes (Derived)" in sf.columns and "Lane" in sf.columns:
        top_pct = sf.sort_values("% of Total Boxes (Derived)", ascending=False).head(top_n)
        fig_pct = px.bar(
            top_pct,
            x="Lane",
            y="% of Total Boxes (Derived)",
            color="Carrier",
            text="% of Total Boxes (Derived)",
            color_discrete_map=BRAND_COLORS  # KEEP brand colours
        )
        fig_pct.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig_pct.update_layout(xaxis_tickangle=-25)
        c2.plotly_chart(fig_style(fig_pct, title=f"Top {top_n} lanes by % share (derived) (brand colours)",
                                  subtitle="Brand colours preserved for carriers.",
                                  x_title="Lane", y_title="% of Total Boxes"),
                        use_container_width=True)

    st.dataframe(sf, use_container_width=True)

# =========================================================
# TAB 5: carrier (Carrier brand colours preserved)
# =========================================================
with tabs[5]:
    st.markdown('<div class="section">carrier</div>', unsafe_allow_html=True)
    st.caption("Carrier summary with brand colours preserved.")

    if "TOTAL BOXES" in df_carrier.columns and df_carrier["TOTAL BOXES"].notna().any():
        c1, c2 = st.columns(2)

        labels = df_carrier["Carrier"].astype(str).tolist()
        values = df_carrier["TOTAL BOXES"].tolist()
        pie_colors = [BRAND_COLORS.get(canonical_carrier(l), None) for l in labels]

        pie = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                textinfo="percent+label",
                marker=dict(colors=pie_colors)
            )]
        )
        c1.plotly_chart(fig_style(pie, title="Share of total boxes by carrier (brand colours)", subtitle="Volume split."),
                        use_container_width=True)

        bar_df = df_carrier.sort_values("TOTAL BOXES", ascending=False).copy()
        bar = px.bar(
            bar_df,
            x="Carrier",
            y="TOTAL BOXES",
            text="TOTAL BOXES",
            color="Carrier",
            color_discrete_map=BRAND_COLORS
        )
        bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        bar.update_layout(showlegend=False)
        c2.plotly_chart(fig_style(bar, title="Total boxes by carrier (brand colours)", subtitle="Raw volume.",
                                  x_title="Carrier", y_title="Total Boxes"),
                        use_container_width=True)

    if "Boxes per Hour" in df_carrier.columns and df_carrier["Boxes per Hour"].notna().any():
        bph_df = df_carrier.sort_values("Boxes per Hour", ascending=False).copy()
        bph = px.bar(
            bph_df,
            x="Carrier",
            y="Boxes per Hour",
            text="Boxes per Hour",
            color="Carrier",
            color_discrete_map=BRAND_COLORS
        )
        bph.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        bph.update_layout(showlegend=False)
        st.plotly_chart(fig_style(bph, title="Boxes per hour by carrier (brand colours)", subtitle="Throughput view.",
                                  x_title="Carrier", y_title="Boxes per Hour"),
                        use_container_width=True)

    st.dataframe(df_carrier, use_container_width=True)

# =========================================================
# TAB 6: scan (24h pulse stays here too)
# =========================================================
with tabs[6]:
    st.markdown('<div class="section">scan</div>', unsafe_allow_html=True)
    st.caption("Time-based view (counts). Includes the 24h Exception Pulse section.")

    if scan_f.empty:
        st.warning("No scan rows match filters.")
    else:
        st.markdown('<div class="section">24h Exception Pulse (KPIs)</div>', unsafe_allow_html=True)
        st.caption("Last 24 hours is based on the latest timestamp in your filtered scan data.")

        tmax = scan_f["Updated"].max()
        tmin_24 = tmax - pd.Timedelta(hours=24)
        scan_24 = scan_f[scan_f["Updated"] >= tmin_24].copy()
        total_24 = len(scan_24)

        if total_24 == 0:
            st.warning("No exceptions in the last 24 hours within current filters.")
        else:
            shift_counts = scan_24["Shift"].value_counts(dropna=False).to_dict()
            a_cnt = int(shift_counts.get("Shift A (06-14)", 0))
            b_cnt = int(shift_counts.get("Shift B (14-22)", 0))
            c_cnt = int(shift_counts.get("Shift C (22-06)", 0))

            a_pct = (a_cnt / total_24) * 100
            b_pct = (b_cnt / total_24) * 100
            c_pct = (c_cnt / total_24) * 100

            type_tbl = (
                scan_24["Exception Type"].value_counts()
                .rename_axis("Exception Type")
                .reset_index(name="Count")
            )
            type_tbl["%"] = (type_tbl["Count"] / total_24) * 100

            # Column S duplicate check
            dup_count_rows = 0
            dup_unique_values = 0
            dup_examples = pd.DataFrame()
            if col_dup_data and col_dup_data in scan_24.columns:
                colS_series = scan_24[col_dup_data].copy()
                colS_series = colS_series.replace({"": np.nan, "nan": np.nan, "None": np.nan})
                colS_nonnull = colS_series.dropna()
                if len(colS_nonnull) > 0:
                    dup_mask = colS_nonnull.duplicated(keep=False)
                    dup_vals = colS_nonnull[dup_mask]
                    dup_count_rows = int(dup_vals.shape[0])
                    dup_unique_values = int(dup_vals.nunique())
                    dup_examples = (
                        dup_vals.value_counts()
                        .reset_index()
                        .rename(columns={"index": "Value", col_dup_data: "Rows"})
                    )
                    dup_examples.columns = ["Value", "Rows"]
                    dup_examples = dup_examples.head(15)

            k = st.columns(6)
            k[0].metric("Overall 24h exceptions %", "100.00%", delta=f"{total_24:,} rows")
            k[1].metric("Shift A by %", safe_percent_str(a_pct), delta=f"{a_cnt:,} rows")
            k[2].metric("Shift B by %", safe_percent_str(b_pct), delta=f"{b_cnt:,} rows")
            k[3].metric("Shift C by %", safe_percent_str(c_pct), delta=f"{c_cnt:,} rows")
            k[4].metric("Column S duplicate rows", f"{dup_count_rows:,}")
            k[5].metric("Column S duplicated values", f"{dup_unique_values:,}")

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            # Exception types by % (severity colours)
            top_types = type_tbl.head(15).copy()
            top_types = top_types.sort_values("%", ascending=False)

            fig_types = px.bar(top_types, x="Exception Type", y="%", text="%")
            fig_types.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_types.update_layout(xaxis_tickangle=-25)
            fig_types = apply_severity_bar_colors(fig_types, top_types["%"].tolist())
            st.plotly_chart(
                fig_style(fig_types,
                          title="Last 24h: exception types by % (severity colours)",
                          subtitle="Red=highest share, green=low, grey=near-zero.",
                          x_title="Exception Type", y_title="% of exceptions"),
                use_container_width=True
            )

            with st.expander("📋 List of Exception Type by % (Last 24h)"):
                st.dataframe(type_tbl, use_container_width=True)

            if col_dup_data and col_dup_data in scan_24.columns:
                with st.expander("🧾 Column S duplicate check details"):
                    if dup_count_rows == 0:
                        st.success(f"No duplicates found in Column S ('{col_dup_data}') excluding blanks.")
                    else:
                        st.warning(f"Found {dup_count_rows:,} duplicate rows across {dup_unique_values:,} values in Column S ('{col_dup_data}').")
                        st.dataframe(dup_examples, use_container_width=True)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Supporting scan charts (counts) with severity colours
        c1, c2 = st.columns(2)

        dp_counts = scan_f["Decision Point"].value_counts().head(top_n).reset_index()
        dp_counts.columns = ["Decision Point", "Exceptions"]
        dp_counts = dp_counts.sort_values("Exceptions", ascending=False)
        fig_dp = px.bar(dp_counts, x="Decision Point", y="Exceptions", text="Exceptions")
        fig_dp.update_traces(textposition="outside", cliponaxis=False)
        fig_dp.update_layout(xaxis_tickangle=-30)
        fig_dp = apply_severity_bar_colors(fig_dp, dp_counts["Exceptions"].tolist())
        c1.plotly_chart(fig_style(fig_dp, title=f"Top {top_n} scanners by exception volume (severity colours)",
                                  subtitle="Count view (filtered).",
                                  x_title="Scanner", y_title="Exceptions"),
                        use_container_width=True)

        type_counts = scan_f["Exception Type"].value_counts().head(top_n).reset_index()
        type_counts.columns = ["Exception Type", "Exceptions"]
        type_counts = type_counts.sort_values("Exceptions", ascending=False)
        fig_type = px.bar(type_counts, x="Exception Type", y="Exceptions", text="Exceptions")
        fig_type.update_traces(textposition="outside", cliponaxis=False)
        fig_type.update_layout(xaxis_tickangle=-25)
        fig_type = apply_severity_bar_colors(fig_type, type_counts["Exceptions"].tolist())
        c2.plotly_chart(fig_style(fig_type, title=f"Top {top_n} exception types (count) (severity colours)",
                                  subtitle="Count view (filtered).",
                                  x_title="Exception Type", y_title="Exceptions"),
                        use_container_width=True)

        hour_counts = scan_f.groupby("Hour").size().reset_index(name="Exceptions").sort_values("Hour")
        fig_hour = px.area(hour_counts, x="Hour", y="Exceptions")
        st.plotly_chart(fig_style(fig_hour, title="Exceptions by hour", subtitle="Time profile for spikes.",
                                  x_title="Hour of day", y_title="Exceptions"),
                        use_container_width=True)

        st.dataframe(scan_f.head(400), use_container_width=True)

# =========================================================
# TAB 7: errors
# =========================================================
with tabs[7]:
    st.markdown('<div class="section">errors</div>', unsafe_allow_html=True)
    st.caption("Supporting view: raw errors sheet.")
    st.dataframe(df_errors, use_container_width=True)

# =========================================================
# TAB 8: mission error
# =========================================================
with tabs[8]:
    st.markdown('<div class="section">mission error</div>', unsafe_allow_html=True)
    st.caption("Supporting view: raw mission error sheet.")
    st.dataframe(df_mission, use_container_width=True)

# =========================================================
# Export
# =========================================================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
with st.expander("⬇️ Quick Export (filtered)"):
    st.download_button("Download filtered exception pct (CSV)",
                       exc_pct_f.to_csv(index=False).encode("utf-8"),
                       "filtered_exception_pct.csv", "text/csv")
    st.download_button("Download filtered overall pct (CSV)",
                       overall_f.to_csv(index=False).encode("utf-8"),
                       "filtered_overall_pct.csv", "text/csv")
    st.download_button("Download filtered scan (CSV)",
                       scan_f.to_csv(index=False).encode("utf-8"),
                       "filtered_scan.csv", "text/csv")
    st.download_button("Download shipping lanes with % (CSV)",
                       df_ship.to_csv(index=False).encode("utf-8"),
                       "shipping_lanes_with_percent.csv", "text/csv")
