# app.py
# =========================================================
# WES Ops Dashboard (Upload-ready)
#
# YOUR GOAL:
# 1) Identify which scanner has the MOST NOREAD and NOSCAN % (from "exception pct")
# 2) Show scanner % statistics (overall pct) and supporting pages
# 3) Shipping lanes: show BOTH counts and % contribution (not only counts)
#
# FIXES INCLUDED:
# - KPI cards use Streamlit-native st.metric (no raw HTML showing)
# - Robust column detection (spaces/case)
# - Percent normalization (0.07 -> 7)
# - Exception pct averages ignore blanks (Excel-like)
# - Colourful charts + readable axes (Light/Dark)
#
# Workbook tab order:
# total → exception pct → overall pct → shipping lanes → carrier → scan → errors → mission error
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
st.set_page_config(page_title="WES Ops Dashboard", page_icon="🐾", layout="wide")

# =========================================================
# Theme selector + Plotly template
# =========================================================
THEME = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"], index=0)
px.defaults.template = "plotly_white" if THEME == "Light" else "plotly_dark"

# =========================================================
# CSS (background + chart cards + metric cards)
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
  h1,h2,h3,p,li,span,div {color: var(--text) !important;}

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

  /* Make st.metric look like cards */
  div[data-testid="metric-container"]{
    border: 1px solid rgba(10,20,60,0.10);
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.78));
    border-radius: 18px;
    padding: 14px 14px;
    box-shadow: 0 18px 44px rgba(10,20,60,0.10);
    height: 100%;
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
    """If values look like 0..1, convert to 0..100."""
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
                x=0.01,
                xanchor="left",
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
        colorway=["#2F7EF7", "#19B37A", "#7C3AED", "#F59E0B", "#EC4899", "#06B6D4", "#EF4444", "#A3E635", "#F97316"],
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, linecolor=axis_color, tickfont=dict(size=12, color=font_color), title_font=dict(size=16, color=font_color))
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, linecolor=axis_color, tickfont=dict(size=12, color=font_color), title_font=dict(size=16, color=font_color))
    return fig

def percent_col_lookup(df: pd.DataFrame, wanted: str) -> str | None:
    """Find exception pct column matching wanted exception type (e.g., NOREAD, NOSCAN)."""
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
    """Return (top_df, full_rank_df). Top is highest→lowest."""
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

def safe_percent_str(x):
    return f"{x:.2f}%" if pd.notna(x) else "—"

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
        <div class="subtitle">Main goal: find top NOREAD / NOSCAN scanners + scanner % statistics</div>
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

def first_number_in_total(df: pd.DataFrame, idx: int) -> float:
    if df.empty:
        return 0.0
    cols = df.columns.tolist()
    if idx < len(cols):
        v = df[cols[idx]].iloc[0]
        return float(v) if pd.notna(v) else 0.0
    return 0.0

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
df_ship["Carrier"] = df_ship["Carrier"].astype(str).str.strip()
df_ship["Total Boxes"] = to_num(df_ship["Total Boxes"])
df_ship["Boxes per Hour"] = to_num(df_ship["Boxes per Hour"])
df_ship["Percentage"] = normalize_percent_series(to_num(df_ship["Percentage"]))

# Derived % contribution from totals (this is what you asked for)
total_boxes_all = float(np.nansum(df_ship["Total Boxes"].values)) if "Total Boxes" in df_ship.columns else 0.0
if total_boxes_all > 0:
    df_ship["% of Total Boxes (Derived)"] = (df_ship["Total Boxes"] / total_boxes_all) * 100
else:
    df_ship["% of Total Boxes (Derived)"] = np.nan

# =========================================================
# carrier (keep as raw + numeric best effort)
# =========================================================
col_car2 = find_col(df_carrier, ["CARRIER Service", "Carrier", "CARRIER", "Service"])
if col_car2:
    df_carrier = df_carrier.rename(columns={col_car2: "Carrier"})
else:
    df_carrier["Carrier"] = "UNKNOWN"
df_carrier["Carrier"] = df_carrier["Carrier"].astype(str).str.strip()

# Try to standardize some numeric columns if they exist
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
# scan (time-based supporting)
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

# Apply filters
exc_pct_f = df_exc_pct[df_exc_pct["Area"].isin(areas) & df_exc_pct["Decision Point"].isin(selected_dps)].copy()
overall_f = df_overall[df_overall["Area"].isin(areas) & df_overall["Decision Point"].isin(selected_dps)].copy()
scan_f = df_scan[
    (df_scan["Date"] >= start_date)
    & (df_scan["Date"] <= end_date)
    & (df_scan["Area"].isin(areas))
    & (df_scan["Decision Point"].isin(selected_dps))
].copy()

# =========================================================
# KPI CALCULATIONS (more cards + scanner names in KPI)
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
    s = normalize_percent_series(df[col])
    # Excel-like: blanks ignored
    return float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan

def scanners_affected(df, col):
    if not col or col not in df.columns or df.empty:
        return 0
    s = pd.to_numeric(normalize_percent_series(df[col]), errors="coerce")
    # affected = scanners with a real value > 0 (ignore NaN)
    return int((s.fillna(0) > 0).sum())

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
    worst_row = overall_f.sort_values("Exception Rate %", ascending=False).head(1)
    if len(worst_row):
        worst_dp = worst_row["Decision Point"].iloc[0]
        worst_rate = float(worst_row["Exception Rate %"].iloc[0]) if pd.notna(worst_row["Exception Rate %"].iloc[0]) else np.nan

scanner_count_selected = int(exc_pct_f["Decision Point"].nunique()) if "Decision Point" in exc_pct_f.columns else 0
areas_count = len(areas)

# =========================================================
# KPI ROWS (as many as useful, but still clean)
# =========================================================
# Row 1 (main purpose)
r1 = st.columns(6)
r1[0].metric("Top NOREAD %", safe_percent_str(top_noread_val), help=f"Scanner: {top_noread_dp}")
r1[1].metric("Top NOREAD scanner", top_noread_dp)
r1[2].metric("Top NOSCAN %", safe_percent_str(top_noscan_val), help=f"Scanner: {top_noscan_dp}")
r1[3].metric("Top NOSCAN scanner", top_noscan_dp)
r1[4].metric("Selected Exception Rate", safe_percent_str(site_rate), help="Sum(exceptions) / Sum(scans) from overall pct")
r1[5].metric("Average Scanner Read %", safe_percent_str(avg_read), help="Average of Scanner Read % from overall pct")

# Row 2 (supporting KPIs)
r2 = st.columns(6)
r2[0].metric("Selected scanners", f"{scanner_count_selected:,}")
r2[1].metric("Selected areas", f"{areas_count:,}")
r2[2].metric("Total scans (selected)", f"{site_scans:,.0f}")
r2[3].metric("Total exceptions (selected)", f"{site_excs:,.0f}")
r2[4].metric("Worst exception rate %", safe_percent_str(worst_rate), help=f"Scanner: {worst_dp}")
r2[5].metric("Worst scanner", worst_dp)

# Row 3 (NOREAD/NOSCAN coverage)
r3 = st.columns(6)
r3[0].metric("Avg NOREAD %", safe_percent_str(avg_noread), help="Average ignoring blanks")
r3[1].metric("Scanners with NOREAD > 0", f"{aff_noread:,}")
r3[2].metric("Avg NOSCAN %", safe_percent_str(avg_noscan), help="Average ignoring blanks")
r3[3].metric("Scanners with NOSCAN > 0", f"{aff_noscan:,}")
r3[4].metric("Scan date range", f"{start_date} → {end_date}", help="Used in scan tab filtering")
r3[5].metric("Scan exceptions (filtered)", f"{len(scan_f):,}", help="Row count from scan tab after filters")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================================================
# Tabs (workbook order)
# =========================================================
tabs = st.tabs([
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
# TAB 1: total
# =========================================================
with tabs[0]:
    st.markdown('<div class="section">total</div>', unsafe_allow_html=True)
    st.caption("High-level totals from your “total” sheet.")
    a = first_number_in_total(df_total_num, 0)
    b = first_number_in_total(df_total_num, 1)
    c = first_number_in_total(df_total_num, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Value 1", f"{a:,.0f}")
    c2.metric("Value 2", f"{b:,.0f}")
    c3.metric("Value 3", f"{c:,.2f}" if pd.notna(c) else "—")

    st.dataframe(df_total, use_container_width=True)

# =========================================================
# TAB 2: exception pct (MAIN)
# =========================================================
with tabs[1]:
    st.markdown('<div class="section">exception pct (main focus)</div>', unsafe_allow_html=True)
    st.caption("Main use: quickly identify scanners with high NOREAD% and NOSCAN%, and rank scanners by any exception type.")

    if exc_pct_f.empty:
        st.warning("No rows match the selected Area/Decision Point filters.")
    else:
        pct_cols2 = [c for c in exc_pct_f.columns if c not in ["Decision Point", "Area"]]
        if not pct_cols2:
            st.warning("No exception % columns found in exception pct sheet.")
        else:
            # 1) NOREAD & NOSCAN charts (TOP only, no bottom)
            st.markdown("### 1) NOREAD & NOSCAN scanner ranking (highest → lowest)")

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
                container.plotly_chart(
                    fig_style(fig,
                              title=f"Top {rank_n} scanners by {label} %",
                              subtitle="Worst scanners first. Use this for action/escalation.",
                              x_title="Scanner (Decision Point)",
                              y_title=f"{label} %"),
                    use_container_width=True
                )

                with container.expander(f"Show full ranking table ({label})"):
                    container.dataframe(full_rank, use_container_width=True)

            show_top_rank(colA, "NOREAD")
            show_top_rank(colB, "NOSCAN")

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            # 2) Pick any exception type and show top ranking
            st.markdown("### 2) Pick an exception type and see scanners ranked (highest → lowest)")

            display_map = {c: str(c).replace("%", "").strip().upper() for c in pct_cols2}
            reverse_map = {v: k for k, v in display_map.items()}
            default_choice = "NOREAD" if "NOREAD" in reverse_map else sorted(reverse_map.keys())[0]

            exc_choice = st.selectbox(
                "Exception Type",
                options=sorted(reverse_map.keys()),
                index=sorted(reverse_map.keys()).index(default_choice),
            )

            exc_col = reverse_map[exc_choice]
            top_df, full_rank = rank_scanners_by_exception_pct(exc_pct_f, exc_col, ignore_blanks, rank_n)

            if full_rank.empty:
                st.warning("No data for selected exception type.")
            else:
                h1, h2, h3 = st.columns(3)
                h1.metric("Selected exception", exc_choice)
                h2.metric("Highest scanner", f"{full_rank.iloc[0]['Decision Point']}", help=f"{full_rank.iloc[0]['Rate %']:.2f}%")
                h3.metric("Highest %", f"{full_rank.iloc[0]['Rate %']:.2f}%")

                fig_any = px.bar(top_df, x="Decision Point", y="Rate %", text="Rate %")
                fig_any.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
                fig_any.update_layout(xaxis_tickangle=-25)
                st.plotly_chart(
                    fig_style(fig_any,
                              title=f"Top {rank_n} scanners by {exc_choice} %",
                              subtitle="Highest to lowest (ranking for selected exception type).",
                              x_title="Scanner (Decision Point)",
                              y_title="Rate %"),
                    use_container_width=True
                )

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            # 3) Summary across exception types
            st.markdown("### 3) Exception type summary (average %, blanks ignored)")

            long = exc_pct_f.melt(
                id_vars=["Decision Point", "Area"],
                value_vars=pct_cols2,
                var_name="Exception Type",
                value_name="Rate %"
            )
            long["Exception Type"] = long["Exception Type"].astype(str).str.replace("%", "", regex=False).str.strip().str.upper()
            long["Rate %"] = normalize_percent_series(pd.to_numeric(long["Rate %"], errors="coerce"))

            top_types = (
                long.groupby("Exception Type", as_index=False)["Rate %"]
                .mean()  # ignores NaN
                .sort_values("Rate %", ascending=False)
                .head(top_n)
            )

            fig_types = px.bar(top_types, x="Exception Type", y="Rate %", text="Rate %")
            fig_types.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_types.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(
                fig_style(fig_types,
                          title=f"Top {top_n} exception types by average %",
                          subtitle="Average is calculated only where values exist (blanks ignored).",
                          x_title="Exception Type",
                          y_title="Average Rate %"),
                use_container_width=True
            )

            st.dataframe(exc_pct_f.drop(columns=["Area"], errors="ignore"), use_container_width=True)

# =========================================================
# TAB 3: overall pct
# =========================================================
with tabs[2]:
    st.markdown('<div class="section">overall pct</div>', unsafe_allow_html=True)
    st.caption("Overall scanner statistics: exception rate %, scanner read %, and volume impact.")

    if overall_f.empty:
        st.warning("No scanner rows match filters.")
    else:
        c1, c2 = st.columns(2)

        top_rate = overall_f.sort_values("Exception Rate %", ascending=False).head(top_n)
        fig_rate = px.bar(top_rate, x="Decision Point", y="Exception Rate %", text="Exception Rate %", color="Area")
        fig_rate.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig_rate.update_layout(xaxis_tickangle=-30)
        c1.plotly_chart(
            fig_style(fig_rate,
                      title=f"Top {top_n} scanners by exception rate %",
                      subtitle="High-level health (rate-based).",
                      x_title="Scanner",
                      y_title="Exception Rate %"),
            use_container_width=True,
        )

        fig_scatter = px.scatter(
            overall_f,
            x="Total Scans",
            y="Exception Rate %",
            size="Total Exceptions",
            color="Area",
            hover_name="Decision Point",
        )
        c2.plotly_chart(
            fig_style(fig_scatter,
                      title="Rate vs volume",
                      subtitle="High rate + high volume = biggest operational impact.",
                      x_title="Total Scans",
                      y_title="Exception Rate %"),
            use_container_width=True,
        )

        # Read % chart if present
        if "Scanner Read %" in overall_f.columns and overall_f["Scanner Read %"].notna().any():
            top_read_bad = overall_f.sort_values("Scanner Read %", ascending=True).head(top_n)
            fig_read = px.bar(top_read_bad, x="Decision Point", y="Scanner Read %", text="Scanner Read %", color="Area")
            fig_read.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig_read.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(
                fig_style(fig_read,
                          title=f"Lowest {top_n} scanners by Scanner Read %",
                          subtitle="Low read % usually means scanning / label / process problems.",
                          x_title="Scanner",
                          y_title="Scanner Read %"),
                use_container_width=True,
            )

        show_cols = ["Area", "Decision Point", "Total Scans", "Total Exceptions", "Exception Rate %", "Scanner Read %"]
        show_cols = [c for c in show_cols if c in overall_f.columns]
        st.dataframe(overall_f[show_cols], use_container_width=True)

# =========================================================
# TAB 4: shipping lanes (NOW: counts + %)
# =========================================================
with tabs[3]:
    st.markdown('<div class="section">shipping lanes</div>', unsafe_allow_html=True)
    st.caption("Lane throughput and contribution. This page shows both counts (Total Boxes) and % contribution (derived).")

    carriers = sorted(df_ship["Carrier"].dropna().unique().tolist()) if "Carrier" in df_ship.columns else []
    pick_carriers = st.multiselect("Carrier", carriers, default=carriers, key="ship_carrier") if carriers else []
    sf = df_ship[df_ship["Carrier"].isin(pick_carriers)].copy() if carriers else df_ship.copy()

    # Recompute derived % on the filtered set (important!)
    total_boxes_filtered = float(np.nansum(sf["Total Boxes"].values)) if "Total Boxes" in sf.columns else 0.0
    if total_boxes_filtered > 0:
        sf["% of Total Boxes (Derived)"] = (sf["Total Boxes"] / total_boxes_filtered) * 100
    else:
        sf["% of Total Boxes (Derived)"] = np.nan

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Lanes", f"{sf['Lane'].nunique():,}" if "Lane" in sf.columns else "—")
    k2.metric("Carriers", f"{sf['Carrier'].nunique():,}" if "Carrier" in sf.columns else "—")
    k3.metric("Total Boxes", f"{np.nansum(sf['Total Boxes'].values):,.0f}" if "Total Boxes" in sf.columns else "—")
    k4.metric("Total Boxes/Hour", f"{np.nansum(sf['Boxes per Hour'].values):,.2f}" if "Boxes per Hour" in sf.columns else "—")
    top_lane = "—"
    top_lane_pct = np.nan
    if "Lane" in sf.columns and "% of Total Boxes (Derived)" in sf.columns and sf["% of Total Boxes (Derived)"].notna().any():
        t = sf.sort_values("% of Total Boxes (Derived)", ascending=False).head(1)
        top_lane = t["Lane"].iloc[0]
        top_lane_pct = float(t["% of Total Boxes (Derived)"].iloc[0])
    k5.metric("Top lane % share", safe_percent_str(top_lane_pct), help=f"Lane: {top_lane}")
    k6.metric("Top lane name", top_lane)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # A) Top lanes by total boxes (count)
    if "Total Boxes" in sf.columns and "Lane" in sf.columns:
        top_vol = sf.sort_values("Total Boxes", ascending=False).head(top_n)
        fig_vol = px.bar(
            top_vol,
            x="Lane",
            y="Total Boxes",
            color="Carrier" if "Carrier" in sf.columns else None,
            text="Total Boxes",
        )
        fig_vol.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        fig_vol.update_layout(xaxis_tickangle=-25)
        c1.plotly_chart(
            fig_style(fig_vol,
                      title=f"Top {top_n} lanes by total boxes (count)",
                      subtitle="Raw volume per lane (filtered set).",
                      x_title="Lane",
                      y_title="Total Boxes"),
            use_container_width=True
        )

    # B) Top lanes by derived % share
    if "% of Total Boxes (Derived)" in sf.columns and "Lane" in sf.columns:
        top_pct = sf.sort_values("% of Total Boxes (Derived)", ascending=False).head(top_n)
        fig_pct = px.bar(
            top_pct,
            x="Lane",
            y="% of Total Boxes (Derived)",
            color="Carrier" if "Carrier" in sf.columns else None,
            text="% of Total Boxes (Derived)",
        )
        fig_pct.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig_pct.update_layout(xaxis_tickangle=-25)
        c2.plotly_chart(
            fig_style(fig_pct,
                      title=f"Top {top_n} lanes by % share (derived)",
                      subtitle="Each lane contribution % within the filtered set.",
                      x_title="Lane",
                      y_title="% of Total Boxes"),
            use_container_width=True
        )

    # C) Throughput view
    if "Boxes per Hour" in sf.columns and "Lane" in sf.columns and sf["Boxes per Hour"].notna().any():
        top_hr = sf.sort_values("Boxes per Hour", ascending=False).head(top_n)
        fig_hr = px.bar(top_hr, x="Lane", y="Boxes per Hour", color="Carrier" if "Carrier" in sf.columns else None, text="Boxes per Hour")
        fig_hr.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig_hr.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(
            fig_style(fig_hr,
                      title=f"Top {top_n} lanes by boxes per hour",
                      subtitle="Throughput view (higher = faster).",
                      x_title="Lane",
                      y_title="Boxes per Hour"),
            use_container_width=True
        )

    st.dataframe(sf, use_container_width=True)

# =========================================================
# TAB 5: carrier
# =========================================================
with tabs[4]:
    st.markdown('<div class="section">carrier</div>', unsafe_allow_html=True)
    st.caption("Carrier summary (supporting).")

    if "TOTAL BOXES" in df_carrier.columns and df_carrier["TOTAL BOXES"].notna().any():
        c1, c2 = st.columns(2)
        pie = go.Figure(data=[go.Pie(labels=df_carrier["Carrier"], values=df_carrier["TOTAL BOXES"], hole=0.55, textinfo="percent+label")])
        c1.plotly_chart(fig_style(pie, title="Share of total boxes by carrier", subtitle="Volume split."), use_container_width=True)

        bar = px.bar(df_carrier.sort_values("TOTAL BOXES", ascending=False), x="Carrier", y="TOTAL BOXES", text="TOTAL BOXES")
        bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        c2.plotly_chart(fig_style(bar, title="Total boxes by carrier", subtitle="Raw volume per carrier.", x_title="Carrier", y_title="Total Boxes"), use_container_width=True)

    if "Boxes per Hour" in df_carrier.columns and df_carrier["Boxes per Hour"].notna().any():
        bph = px.bar(df_carrier.sort_values("Boxes per Hour", ascending=False), x="Carrier", y="Boxes per Hour", text="Boxes per Hour")
        bph.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_style(bph, title="Boxes per hour by carrier", subtitle="Throughput view.", x_title="Carrier", y_title="Boxes per Hour"), use_container_width=True)

    st.dataframe(df_carrier, use_container_width=True)

# =========================================================
# TAB 6: scan
# =========================================================
with tabs[5]:
    st.markdown('<div class="section">scan</div>', unsafe_allow_html=True)
    st.caption("Time-based view (counts). Use this to see when exceptions spike by hour/shift.")

    if scan_f.empty:
        st.warning("No scan rows match filters.")
    else:
        c1, c2 = st.columns(2)

        dp_counts = scan_f["Decision Point"].value_counts().head(top_n).reset_index()
        dp_counts.columns = ["Decision Point", "Exceptions"]
        fig_dp = px.bar(dp_counts, x="Decision Point", y="Exceptions", text="Exceptions", color="Exceptions")
        fig_dp.update_traces(textposition="outside", cliponaxis=False)
        fig_dp.update_layout(xaxis_tickangle=-30)
        c1.plotly_chart(fig_style(fig_dp, title=f"Top {top_n} scanners by exception volume", subtitle="Count view from scan tab (filtered).", x_title="Scanner", y_title="Exceptions"), use_container_width=True)

        type_counts = scan_f["Exception Type"].value_counts().head(top_n).reset_index()
        type_counts.columns = ["Exception Type", "Exceptions"]
        fig_type = px.bar(type_counts, x="Exception Type", y="Exceptions", text="Exceptions", color="Exceptions")
        fig_type.update_traces(textposition="outside", cliponaxis=False)
        fig_type.update_layout(xaxis_tickangle=-25)
        c2.plotly_chart(fig_style(fig_type, title=f"Top {top_n} exception types", subtitle="Count view from scan tab (filtered).", x_title="Exception Type", y_title="Exceptions"), use_container_width=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        hour_counts = scan_f.groupby("Hour").size().reset_index(name="Exceptions").sort_values("Hour")
        fig_hour = px.area(hour_counts, x="Hour", y="Exceptions")
        st.plotly_chart(fig_style(fig_hour, title="Exceptions by hour", subtitle="Time profile for spikes.", x_title="Hour of day", y_title="Exceptions"), use_container_width=True)

        # Heatmap (scanner vs type)
        pivot = (
            scan_f.groupby(["Decision Point", "Exception Type"]).size()
            .reset_index(name="Count")
            .pivot_table(index="Decision Point", columns="Exception Type", values="Count", aggfunc="sum")
            .fillna(0)
        )
        top_rows = scan_f["Decision Point"].value_counts().head(min(top_n, 25)).index
        pivot = pivot.loc[top_rows]
        fig_hm = px.imshow(pivot, aspect="auto")
        st.plotly_chart(fig_style(fig_hm, title="Heatmap: exceptions by scanner vs type (counts)", subtitle="From scan tab (filtered time window).", x_title="Exception Type", y_title="Scanner"), use_container_width=True)

        st.dataframe(scan_f.head(400), use_container_width=True)

# =========================================================
# TAB 7: errors (raw support)
# =========================================================
with tabs[6]:
    st.markdown('<div class="section">errors</div>', unsafe_allow_html=True)
    st.caption("Supporting view: raw errors sheet.")
    st.dataframe(df_errors, use_container_width=True)

# =========================================================
# TAB 8: mission error (raw support)
# =========================================================
with tabs[7]:
    st.markdown('<div class="section">mission error</div>', unsafe_allow_html=True)
    st.caption("Supporting view: raw mission error sheet.")
    st.dataframe(df_mission, use_container_width=True)

# =========================================================
# Export
# =========================================================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
with st.expander("⬇️ Quick Export (filtered)"):
    st.download_button("Download filtered exception pct (CSV)", exc_pct_f.to_csv(index=False).encode("utf-8"), "filtered_exception_pct.csv", "text/csv")
    st.download_button("Download filtered overall pct (CSV)", overall_f.to_csv(index=False).encode("utf-8"), "filtered_overall_pct.csv", "text/csv")
    st.download_button("Download filtered scan (CSV)", scan_f.to_csv(index=False).encode("utf-8"), "filtered_scan.csv", "text/csv")
    st.download_button("Download filtered shipping lanes (CSV)", df_ship.to_csv(index=False).encode("utf-8"), "shipping_lanes_with_percent.csv", "text/csv")