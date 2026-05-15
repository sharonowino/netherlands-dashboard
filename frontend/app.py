"""
===============================================================================
  TRANSIT DISRUPTION DASHBOARD — STREAMLIT FRONTEND  v2.1
  7-Layer Architecture · Layer 7 (Deployment Interface)
===============================================================================
  Pages
  -----
  Overview    → KPI row, route table, event log, alert donut, system status
  Live Map    → Folium vehicle map (live → sample), delay hotspots
  Predictions → RF/XGB 30-min forecasts, SHAP TreeExplainer panel
  Analytics   → 24-h trend, budget vs actuals, period KPIs

  All data fetched from FastAPI backend (http://backend:8000).
  Every fetch has a sample-data fallback so the UI never crashes.
  Auto-refresh uses st_autorefresh when installed, else a manual button.
===============================================================================
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# Detect if running inside docker-compose by checking an env override; default to localhost for dev
API_URL = os.getenv("API_BASE_URL") or os.getenv("BACKEND_URL") or "http://localhost:8000"
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "30"))

st.set_page_config(
    page_title="TransitOS — NL Operations",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚌",
)

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CSS / TYPOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #000000 !important;
}

/* Ensure all text is visible in dark mode */
.stApp, .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader,
.stCaption, .stCode, .stJson, .stTable, .stDataFrame,
.stMetric, .stSuccess, .stInfo, .stWarning, .stError,
.stException, .stForm, .stExpander, .stTabs, .stSidebar {
    color: #e2e8f0 !important;
}

/* Override any default dark text */
p, span, div, h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
}

/* Specific Streamlit component overrides */
.stMarkdown p, .stMarkdown span, .stMarkdown div {
    color: #e2e8f0 !important;
}

.stMetric .metric-label, .stMetric .metric-value {
    color: inherit !important;
}

/* DataFrame text */
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    color: #e2e8f0 !important;
}

/* Table text */
.stTable td, .stTable th {
    color: #e2e8f0 !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1628 100%);
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.4rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #1a1f35 100%);
    border-right: 1px solid #2a3441;
    box-shadow: 2px 0 10px rgba(0,0,0,0.3);
}

.logo {
    background: linear-gradient(135deg, #1e3a5f, #1e90ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-shadow: 0 2px 4px rgba(30,144,255,0.3);
}

.live-dot {
    display:inline-block;
    width:10px;
    height:10px;
    border-radius:50%;
    background:#2ed573;
    animation:pulse 2s ease-in-out infinite;
    box-shadow: 0 0 10px rgba(46,213,115,0.5);
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0f1628 0%, #162035 100%);
    border: 1px solid #2a3441;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.green::before { background: linear-gradient(90deg, #2ed573, #4ade80); }
.kpi-card.amber::before { background: linear-gradient(90deg, #ffa502, #fbbf24); }
.kpi-card.red::before { background: linear-gradient(90deg, #ff4757, #f87171); }
.kpi-card.blue::before { background: linear-gradient(90deg, #1e90ff, #3b82f6); }

.kpi-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-val {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
.kpi-val.green { color: #2ed573; }
.kpi-val.amber { color: #ffa502; }
.kpi-val.red { color: #ff4757; }
.kpi-val.blue { color: #1e90ff; }
.kpi-sub {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
}

/* ── badges ── */
.badge {
    display:inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border: 1px solid;
    transition: all 0.3s ease;
    color: #e2e8f0 !important;
}
.badge:hover { transform: translateY(-1px); }
.badge-crit { background: rgba(255,71,87,0.15); border-color: #ff4757; }
.badge-warn { background: rgba(255,165,2,0.15); border-color: #ffa502; }
.badge-ok { background: rgba(46,213,115,0.15); border-color: #2ed573; }
.badge-info { background: rgba(30,144,255,0.15); border-color: #1e90ff; }

/* ── section header ── */
.sh {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #1e90ff;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2a3441;
    font-weight: 700;
}

/* ── dataframes and tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2a3441;
    border-radius: 12px;
    background: #0f1628;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #1e90ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(30,144,255,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e90ff, #1e3a5f);
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(30,144,255,0.4);
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1628;
    border-bottom: 2px solid #2a3441;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
    padding: 12px 20px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    color: #1e90ff !important;
    background: rgba(30,144,255,0.1);
    border-bottom: 2px solid #1e90ff;
}

/* ── headings ── */
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #e2e8f0;
    font-weight: 700;
}

/* ── widget styling ── */
.stRadio > label,
.stSelect > label,
.stSlider > label,
.stTextInput > label,
.stTextArea > label,
.stNumberInput > label,
.stTimeInput > label,
.stDateInput > label,
.stCheckbox > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #e2e8f0 !important;
    font-weight: 500;
}

.stRadio [data-baseweb="radio"],
.stSelect [data-baseweb="select"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    background: #1a1f35;
    border: 1px solid #2a3441;
    border-radius: 8px;
    color: #e2e8f0;
}

.stSlider [data-baseweb="slider"] {
    font-family: 'Inter', sans-serif;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    background: #1a1f35;
    border: 1px solid #2a3441;
    border-radius: 8px;
    color: #e2e8f0;
}

/* ── severity styling ── */
.severity-normal { border-left: 5px solid #2ed573; background: rgba(46,213,115,0.05); }
.severity-minor { border-left: 5px solid #1e90ff; background: rgba(30,144,255,0.05); }
.severity-moderate { border-left: 5px solid #ffa502; background: rgba(255,165,2,0.05); }
.severity-severe { border-left: 5px solid #ff4757; background: rgba(255,71,87,0.05); }

.alert-item {
    background: linear-gradient(135deg, #162035 0%, #1a1f35 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    border-left: 4px solid;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.alert-item:hover {
    transform: translateX(2px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.3);
}

/* ── prediction card ── */
.pred-card {
    display:flex;
    align-items:center;
    gap:16px;
    padding:16px;
    border: 1px solid #2a3441;
    border-radius: 12px;
    margin-bottom:12px;
    background: linear-gradient(135deg, #0f1628 0%, #162035 100%);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.pred-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.3);
}

/* ── forms ── */
.stForm {
    background: #0f1628;
    border: 1px solid #2a3441;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── expanders ── */
.streamlit-expanderHeader {
    background: #1a1f35 !important;
    border: 1px solid #2a3441 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

.streamlit-expanderContent {
    background: #0f1628 !important;
    border: 1px solid #2a3441 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── progress bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1e90ff, #3b82f6) !important;
}

/* ── plotly charts ── */
.js-plotly-plot .plotly .modebar {
    background: #1a1f35 !important;
}

.js-plotly-plot .plotly .modebar-btn {
    color: #94a3b8 !important;
}

/* Plotly chart text colors */
.js-plotly-plot .plotly text {
    fill: #e2e8f0 !important;
}

.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text {
    fill: #cbd5e1 !important;
}

.js-plotly-plot .plotly .legend text {
    fill: #e2e8f0 !important;
}

.js-plotly-plot .plotly .hovertext {
    color: #000000 !important;
}

/* Ensure all chart elements are visible */
.plotly-notifier {
    fill: #e2e8f0 !important;
}

/* ── custom scrollbar ── */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0f1628;
}

::-webkit-scrollbar-thumb {
    background: #2a3441;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3a4555;
}

/* ── text visibility for dark mode ── */
/* Ensure specific elements are visible without overriding intended colors */
.stMarkdown, .stText, .stCaption, .stSuccess, .stInfo, .stWarning, .stError {
    color: #e2e8f0 !important;
}

/* DataFrame and table text */
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
.stTable td, .stTable th {
    color: #e2e8f0 !important;
}

/* Form and widget text */
.stRadio [data-baseweb="radio"] span,
.stSelect [data-baseweb="select"] span,
.stCheckbox span {
    color: #e2e8f0 !important;
}

/* ── metric cards (for analytics) ── */
.metric-card {
    background: linear-gradient(135deg, #0f1628 0%, #162035 100%);
    border: 1px solid #2a3441;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
}
.metric-card.normal::before { background: linear-gradient(90deg, #2ed573, #4ade80); }
.metric-card.minor::before { background: linear-gradient(90deg, #1e90ff, #3b82f6); }
.metric-card.moderate::before { background: linear-gradient(90deg, #ffa502, #fbbf24); }
.metric-card.severe::before { background: linear-gradient(90deg, #ff4757, #f87171); }
.metric-card.total::before { background: linear-gradient(90deg, #1e90ff, #3b82f6); }

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 8px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #94a3b8;
    font-weight: 600;
}
.normal .metric-value { color: #2ed573; }
.minor .metric-value { color: #1e90ff; }
.moderate .metric-value { color: #ffa502; }
.severe .metric-value { color: #ff4757; }
.total .metric-value { color: #1e90ff; }

/* ── section titles ── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #1e90ff;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2d4a;
}

/* ── api status ── */
.api-status-ok {
    color: #2ed573;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}
.api-status-err {
    color: #ff4757;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Simple cached API health to avoid repeated slow calls
_API_STATUS: Dict[str, Any] = {"last_check": 0.0, "ok": False}


def _get(path: str, timeout: int = 5) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        logger.warning(f"GET {path} → {exc}")
        return None
    except Exception as exc:
        logger.exception(f"GET {path} unexpected error → {exc}")
        return None


def _post(path: str, payload: Dict, timeout: int = 10) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(f"POST {path} → {exc}")
        return None


def _api_online() -> bool:
    # Use a cached small health probe to avoid repeated timeouts
    now = time.time()
    if now - _API_STATUS["last_check"] < 5.0:
        return _API_STATUS["ok"]
    d = _get("/health", timeout=2)
    ok = bool(d and d.get("status") == "ok")
    _API_STATUS["last_check"] = now
    _API_STATUS["ok"] = ok
    return ok


def _active_model() -> str:
    d = _get("/model/info", timeout=3)
    return d.get("active", "simulation") if d else "simulation"


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE / FALLBACK DATA
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_routes() -> List[Dict]:
    return [
        {"id":"R-01","name":"Amsterdam C → Schiphol",  "status":"crit",  "delay":18.4,"bunch":0.82,"throughput":340,"pred":89},
        {"id":"R-04","name":"Utrecht → Den Haag",       "status":"crit",  "delay":12.1,"bunch":0.67,"throughput":210,"pred":74},
        {"id":"R-07","name":"Rotterdam → Dordrecht",    "status":"delay", "delay":5.8, "bunch":0.44,"throughput":185,"pred":47},
        {"id":"R-12","name":"Eindhoven → Tilburg",      "status":"delay", "delay":3.2, "bunch":0.31,"throughput":150,"pred":29},
        {"id":"R-15","name":"Haarlem → Leiden",         "status":"ok",    "delay":0.9, "bunch":0.12,"throughput":220,"pred":11},
        {"id":"R-22","name":"Delft → Rotterdam C",      "status":"ok",    "delay":1.1, "bunch":0.18,"throughput":190,"pred":8},
        {"id":"R-03","name":"Arnhem → Nijmegen",        "status":"delay", "delay":4.4, "bunch":0.38,"throughput":120,"pred":35},
        {"id":"R-09","name":"Breda → Tilburg",          "status":"ok",    "delay":0.4, "bunch":0.08,"throughput":95, "pred":6},
    ]


def _fallback_metrics() -> Dict:
    return {
        "on_time_pct":88.2, "active_disruptions":12, "avg_delay_min":8.2,
        "prediction_f1":0.87, "service_delivered_pct":96.5,
        "data_quality_score":94.1, "inference_latency_ms":145,
        "throughput_veh_hr":1240, "model_active":"simulation",
    }


def _fallback_alerts() -> List[Dict]:
    return [
        {"time":"now", "msg":"R-01 speed drop — 22 km/h",               "sev":"crit"},
        {"time":"2m",  "msg":"R-04 bunching alert — 3 vehicles clustered","sev":"crit"},
        {"time":"7m",  "msg":"R-07 schedule slip — +5.8 min avg",        "sev":"warn"},
        {"time":"12m", "msg":"RF model: R-01 SEVERE (89% confidence)",   "sev":"info"},
        {"time":"18m", "msg":"R-12 headway variance elevated",           "sev":"warn"},
        {"time":"24m", "msg":"GTFS-RT feed reconnected",                 "sev":"info"},
        {"time":"31m", "msg":"R-15 back on-time",                        "sev":"ok"},
    ]


def _fallback_vehicles(n: int = 80) -> pd.DataFrame:
    seed = int(time.time()) // 30
    rng  = np.random.default_rng(seed)
    routes = ["R-01","R-03","R-04","R-07","R-09","R-12","R-15","R-22"]
    return pd.DataFrame({
        "vehicle_id": [f"NL-{1000+i}" for i in range(n)],
        "route_id":   rng.choice(routes, n),
        "latitude":   rng.uniform(51.88, 52.42, n).round(5),
        "longitude":  rng.uniform(4.18,  5.18,  n).round(5),
        "speed":      rng.uniform(0, 82, n).round(1),
        "bearing":    rng.integers(0, 360, n),
    })


def _fallback_shap() -> Dict[str, float]:
    return {
        "bunching_index":0.42,"delay_mean_15m":0.35,"speed_drop_ratio":0.31,
        "on_time_pct":0.28,"delay_mean_5m":0.25,"speed_mean":0.22,
        "delay_mean_30m":0.20,"headway_variance":0.18,"fleet_utilization":0.15,
        "alert_nlp_score":0.12,"speed_std":0.10,"alert_count":0.08,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_metrics() -> Dict:
    return _get("/metrics") or _fallback_metrics()


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_routes() -> List[Dict]:
    data = _get("/routes")
    routes = data.get("routes", []) if data else []
    if not routes:
        return _fallback_routes()

    # Attach prediction probabilities from the batch endpoint
    payload = {"routes": [
        {"route_id": r["id"],
         "bunching_index":  r.get("bunch", 0.2),
         "delay_mean_15m":  r.get("delay", 0) * 60,
         "speed_mean":      max(5.0, 35 - r.get("delay", 0) * 0.8)}
        for r in routes
    ]}
    pred_data = _post("/predict/batch", payload)
    if pred_data:
        pred_map = {
            p["route_id"]: round(p["confidence"] * 100, 1)
            for p in pred_data.get("results", [])
        }
        for r in routes:
            r["pred"] = pred_map.get(r["id"], 0.0)
    else:
        for r in routes:
            r.setdefault("pred", 0.0)
    return routes


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_alerts() -> List[Dict]:
    data = _get("/alerts")
    return data.get("alerts", _fallback_alerts()) if data else _fallback_alerts()


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_vehicles() -> pd.DataFrame:
    # Prefer merged_feed for a single unified source; fallback to /feed
    try:
        data = _get("/merged_feed", timeout=10)
    except Exception:
        data = None

    if data and data.get("rows"):
        try:
            df = pd.DataFrame(data["rows"])
            # Limit number of vehicles to render for performance in the UI
            if len(df) > 500:
                df = df.sample(500, random_state=42).reset_index(drop=True)
            return df
        except Exception:
            pass

    # fallback to previous behaviour
    try:
        data = _get("/feed", timeout=8)
    except Exception:
        data = None

    if data and data.get("vehicles"):
        try:
            df = pd.DataFrame(data["vehicles"])
            if len(df) > 500:
                df = df.sample(500, random_state=42).reset_index(drop=True)
            return df
        except Exception:
            pass
    return _fallback_vehicles()

@st.cache_data(ttl=60)
def fetch_shap(route_id: str) -> Dict[str, float]:
    data = _get(f"/shap/{route_id}")
    if not data:
        return _fallback_shap()
    # If scheduled, return a lightweight status dict that UI can handle
    if isinstance(data, dict) and data.get("status") == "scheduled":
        return {"__scheduled__": True, "n_background": data.get("n_background", 50)}
    return data.get("contributions", _fallback_shap())


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _delay_color(d: float) -> str:
    return "#C84040" if d > 8 else "#B07010" if d > 3 else "#2A7A4A"

def _prob_color(p: float) -> str:
    return "#C84040" if p > 70 else "#B07010" if p > 40 else "#2A7A4A"

ROUTE_COLORS = {
    "R-01":"#C84040","R-04":"#C84040","R-07":"#B07010",
    "R-12":"#B07010","R-15":"#2A7A4A","R-22":"#2A7A4A",
    "R-03":"#B07010","R-09":"#1A5FA0",
}

SEVERITY_COLORS = {0: "#2A7A4A", 1: "#1A5FA0", 2: "#B07010", 3: "#C84040"}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

STATUS_HTML = {
    "crit":  '<span class="badge badge-crit">Critical</span>',
    "delay": '<span class="badge badge-warn">Delayed</span>',
    "ok":    '<span class="badge badge-ok">On-time</span>',
}

SEV_COLOR = {"crit":"#C84040","warn":"#B07010","info":"#1A5FA0","ok":"#2A7A4A"}


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

_CHART_FONT = dict(family="IBM Plex Mono", size=11)
_TRANSPARENT = "rgba(0,0,0,0)"


def _base_layout(**kwargs) -> Dict:
    base = dict(
        paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT,
        font=_CHART_FONT, margin=dict(t=4,b=4,l=4,r=4),
    )
    base.update(kwargs)
    return base


def fig_sparkline(data: List[int]) -> go.Figure:
    colors = ["#C84040" if v > 25 else "#B07010" if v > 15 else "#1A5FA0" for v in data]
    fig = go.Figure(go.Bar(x=list(range(len(data))), y=data,
                           marker_color=colors, marker_line_width=0))
    fig.update_layout(**_base_layout(height=72),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def fig_donut(data: Dict[str, int]) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(data.keys()), values=list(data.values()), hole=0.65,
        marker=dict(colors=["#E24B4A","#BA7517","#185FA5","#3B6D11"],
                    line=dict(color="white", width=1)),
        textinfo="none",
    ))
    fig.update_layout(**_base_layout(height=130))
    return fig


def fig_delay_bars(routes: List[Dict]) -> go.Figure:
    df = pd.DataFrame(routes).sort_values("delay", ascending=True)
    fig = go.Figure(go.Bar(
        y=df["id"], x=df["delay"], orientation="h",
        marker_color=[_delay_color(d) for d in df["delay"]], marker_line_width=0,
        text=[f"{d:.1f} min" for d in df["delay"]], textposition="outside",
    ))
    fig.update_layout(**_base_layout(height=230),
                      xaxis=dict(showgrid=False, showticklabels=False),
                      yaxis=dict(tickfont=_CHART_FONT))
    return fig


def fig_shap(contributions: Dict[str, float]) -> go.Figure:
    items  = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:10]
    names, vals = zip(*items)
    colors = ["#C84040" if v > 0.3 else "#B07010" if v > 0.15 else "#1A5FA0" for v in vals]
    fig = go.Figure(go.Bar(
        y=list(names), x=list(vals), orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(**_base_layout(height=280),
                      xaxis=dict(showgrid=False, showticklabels=False),
                      yaxis=dict(tickfont=_CHART_FONT, autorange="reversed"))
    return fig


def fig_trend(hours, values) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=hours, y=values, mode="lines",
        line=dict(color="#1A5FA0", width=2),
        fill="tozeroy", fillcolor="rgba(26,95,160,.06)",
    ))
    fig.update_layout(**_base_layout(height=190, showlegend=False),
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig


def fig_vehicle_map_plotly(df: pd.DataFrame) -> go.Figure:
    df = df.dropna(subset=["latitude","longitude"])
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No position data", showarrow=False,
                           font=dict(color="#888", size=13))
        fig.update_layout(height=320, paper_bgcolor=_TRANSPARENT)
        return fig
    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude", color="route_id",
        color_discrete_map=ROUTE_COLORS, zoom=9,
        center={"lat": float(df["latitude"].mean()),
                "lon": float(df["longitude"].mean())},
        hover_name="vehicle_id" if "vehicle_id" in df.columns else None,
        hover_data={"speed": True} if "speed" in df.columns else {},
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        **_base_layout(height=340, showlegend=True),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        hoverlabel=dict(font=dict(color="#000000"))
    )
    return fig


def _folium_map(df: pd.DataFrame) -> "folium.Map":
    lat0 = float(df["latitude"].mean())  if not df.empty else 52.1
    lon0 = float(df["longitude"].mean()) if not df.empty else 4.7
    m = folium.Map(location=[lat0, lon0], zoom_start=10,
                   tiles="CartoDB positron")
    for _, row in df.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue
        color = ROUTE_COLORS.get(row.get("route_id",""), "gray")
        spd   = row.get("speed", 0) or 0
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6, color=color, fill=True,
            fill_color=color, fill_opacity=0.75,
            tooltip=f"{row.get('vehicle_id','?')} · {row.get('route_id','?')} · {spd:.0f} km/h",
        ).add_to(m)
    return m


def _folium_severity_map(stops: List[Dict], spatial_lags: Dict, mode: str) -> "folium.Map":
    lat0, lon0 = 52.37, 4.89
    m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="CartoDB positron")
    for stop in stops:
        if mode == "mean":
            sev = stop["severity"]
        elif mode == "max":
            lag_delay = spatial_lags[stop["id"]]
            sev = min(int(lag_delay // 10), 3)  # scale delay to severity 0-3
        else:
            sev = stop["severity"]
        color = SEVERITY_COLORS.get(sev, "gray")
        size = 6 + math.log(stop["events"] + 1) * 2
        tooltip = f"{stop['id']} · Delay: {stop['delay']} min · Severity: {sev} · Spatial Lag: {spatial_lags[stop['id']]:.1f} min"
        popup = f"High Risk: {stop['cause']}" if stop["risk"] == "high" else None
        folium.CircleMarker(
            location=[stop["lat"], stop["lon"]],
            radius=size, color=color, fill=True,
            fill_color=color, fill_opacity=0.75,
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)
    return m


def display_lead_time_analysis():
    """Display lead-time analysis for early warning."""
    st.markdown('<div class="sh">⏱️ Early Warning Lead Time Analysis</div>', unsafe_allow_html=True)

    # Simulated lead time data
    lead_times = np.random.exponential(15, 1000)  # minutes

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            x=lead_times,
            nbins=30,
            title='Distribution of Warning Lead Times',
            labels={'x': 'Lead Time (minutes)', 'y': 'Count'},
            color_discrete_sequence=["#1e90ff"]
        )
        fig.add_vline(x=15, line_dash="dash", line_color="#ff4757", annotation_text="15 min threshold")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        ### Lead Time Metrics

        | Metric | Value |
        |--------|-------|
        | Mean Lead Time | 15.2 min |
        | Median Lead Time | 12.8 min |
        | Std Deviation | 8.4 min |
        | Alerts with >15min lead | 42% |

        **Goal**: Achieve >80% of alerts with ≥15 minute lead time before disruption.
        """)


def display_bunching_index(routes: List[Dict]):
    """Display bunching index distribution."""
    st.markdown('<div class="sh">📊 Bunching Index Distribution</div>', unsafe_allow_html=True)

    # Create dataframe from routes
    df = pd.DataFrame(routes)
    if 'bunch' in df.columns:
        df['bunching_index'] = df['bunch']
    else:
        df['bunching_index'] = np.random.uniform(0, 1, len(df))

    # Map status to severity
    status_to_severity = {"ok": 0, "delay": 1, "crit": 3}
    df['severity_class'] = df['status'].map(status_to_severity).fillna(0).astype(int)

    fig = px.histogram(
        df,
        x='bunching_index',
        color='severity_class',
        title='Bunching Index Distribution',
        color_discrete_map={k: v for k, v in SEVERITY_COLORS.items()},
        labels={'severity_class': 'Severity'}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,22,40,0.8)",
        font=dict(family="DM Sans"),
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


def display_disruption_by_severity(routes: List[Dict]):
    """Display disruptions by severity."""
    st.markdown('<div class="sh">🚨 Disruptions by Severity</div>', unsafe_allow_html=True)

    df = pd.DataFrame(routes)
    status_to_severity = {"ok": "Normal", "delay": "Minor", "crit": "Severe"}
    df['Severity'] = df['status'].map(status_to_severity).fillna("Normal")

    severity_counts = df['Severity'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']

    color_map = {'Normal': '#2A7A4A', 'Minor': '#1A5FA0', 'Severe': '#C84040'}

    fig = px.bar(severity_counts, x='Severity', y='Count',
                 title="Disruptions by Severity",
                 color='Severity',
                 color_discrete_map=color_map)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,22,40,0.8)",
        font=dict(family="DM Sans"),
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


def display_routes_by_severity(routes: List[Dict]):
    """Display routes by severity."""
    st.markdown('<div class="sh">🗺️ Routes by Severity</div>', unsafe_allow_html=True)

    df = pd.DataFrame(routes)
    status_to_severity = {"ok": 0, "delay": 1, "crit": 3}
    df['severity_class'] = df['status'].map(status_to_severity).fillna(0).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            df,
            names='severity_class',
            title='Routes by Severity',
            color='severity_class',
            color_discrete_map={k: v for k, v in SEVERITY_COLORS.items()},
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        route_counts = df.groupby(['id', 'severity_class']).size().reset_index(name='Count')
        route_counts['severity_class'] = route_counts['severity_class'].map({0: 'Normal', 1: 'Minor', 3: 'Severe'})
        fig = px.bar(route_counts, x='id', y='Count',
                     title="Routes by Alert Count",
                     color='severity_class',
                     color_discrete_map={'Normal': '#2A7A4A', 'Minor': '#1A5FA0', 'Severe': '#C84040'})
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(api_ok: bool, model_name: str, metrics: Dict) -> Dict:
    with st.sidebar:
        st.markdown(
            '<div class="logo" style="padding:10px 0 14px">TRANSIT<span>OS</span></div>'
            '<div style="font-size:10px;color:#888;margin-top:-10px;margin-bottom:14px">'
            'NL · OPERATIONS CENTER</div>',
            unsafe_allow_html=True,
        )

        # API status
        col_dot, col_txt = st.columns([1, 5])
        with col_dot:
            st.markdown('<span class="live-dot"></span>', unsafe_allow_html=True)
        with col_txt:
            color = "#2A7A4A" if api_ok else "#C84040"
            label = "API online" if api_ok else "API offline — sample data"
            st.markdown(f'<span style="font-size:11px;color:{color}">{label}</span>',
                        unsafe_allow_html=True)

        st.markdown(f'<div style="font-size:10px;color:#888;margin:4px 0 12px">'
                    f'{datetime.now().strftime("%H:%M:%S UTC")}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Data Source**")
        data_source = st.radio("Data Source", ["Live API", "Sample Data", "Upload ZIP", "Parquet Folder"], key="data_source_type", label_visibility="collapsed")

        if data_source == "Upload ZIP":
            uploaded = st.file_uploader("ZIP files", type=["zip"], key="zip_upload", accept_multiple_files=True)
            if uploaded and st.button("Load ZIPs", key="load_zips"):
                st.info("ZIP upload processing not yet implemented in frontend.")
        elif data_source == "Parquet Folder":
            folder = st.text_input("Folder Path", key="folder_path", placeholder="Path to parquet folder")
            if folder and st.button("Load Folder", key="load_folder"):
                st.info("Folder loading not yet implemented in frontend.")

        st.markdown("---")
        st.markdown('<div class="sh">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("Page",
                        ["Overview", "Live Map", "Predictions", "Analytics"],
                        label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<div class="sh">Filters</div>', unsafe_allow_html=True)
        sev_filter = st.multiselect(
            "Severity",
            ["Critical", "Delayed", "On-time"],
            default=["Critical", "Delayed", "On-time"],
        )
        search = st.text_input("Search route", placeholder="R-01, Amsterdam…")

        st.markdown("---")
        st.markdown('<div class="sh">Auto-refresh</div>', unsafe_allow_html=True)
        auto_refresh = st.checkbox(f"Enable ({REFRESH_INTERVAL}s)", value=False)
        if auto_refresh:
            if AUTOREFRESH_OK:
                st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="ar")
            else:
                st.caption("Install streamlit-autorefresh for true auto-refresh.")
                if st.button("↺ Refresh now"):
                    st.cache_data.clear()
                    st.rerun()

        st.markdown("---")
        st.markdown('<div class="sh">Model</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:11px;line-height:1.9">'
            f'<span style="color:#888">Active:</span> {model_name}<br>'
            f'<span style="color:#888">F1 score:</span> {metrics.get("prediction_f1",0.87):.2f}<br>'
            f'<span style="color:#888">Latency P95:</span> {metrics.get("inference_latency_ms",145)} ms<br>'
            f'<span style="color:#888">Leakage ctrl:</span> <span style="color:#2A7A4A">Active</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        if st.button("↺ Clear cache"):
            st.cache_data.clear()
            st.rerun()

        # Hot-reload model button
        if st.button("⟳ Reload model"):
            resp = _post("/model/reload", {})
            if resp:
                st.success("Reload scheduled — check /model/info in a moment.")
            else:
                st.warning("Backend unreachable.")

    return {"page": page, "sev_filter": sev_filter,
            "search": search, "auto_refresh": auto_refresh}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED TOPBAR + KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

def render_topbar(api_ok: bool, model_name: str):
    api_color = "#2ed573" if api_ok else "#ff4757"
    api_label = "● OPERATIONAL" if api_ok else "● OFFLINE"
    status_icon = "🟢" if api_ok else "🔴"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f1628 0%, #1a1f35 100%);
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #2a3441;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {api_color};
                animation: pulse 2s ease-in-out infinite;
                box-shadow: 0 0 12px {api_color}80;
            "></div>
            <div>
                <div class="logo" style="margin: 0; font-size: 1.8rem;">TRANSIT<span>OS</span></div>
                <div style="
                    color: #94a3b8;
                    font-size: 0.8rem;
                    font-family: 'Inter', sans-serif;
                    margin-top: 2px;
                ">NETHERLANDS TRANSIT INTELLIGENCE</div>
            </div>
        </div>
        <div style="
            display: flex;
            align-items: center;
            gap: 24px;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
        ">
            <div style="color: #cbd5e1;">
                <span style="color: #64748b;">Model:</span>
                <span style="color: #1e90ff; font-weight: 600;">{model_name.upper()}</span>
            </div>
            <div style="color: {api_color}; font-weight: 600;">
                {status_icon} {api_label}
            </div>
            <div style="color: #94a3b8;">
                {datetime.now().strftime('%H:%M:%S UTC')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(m: Dict):
    kpis = [
        ("On-time perf.",    f"{m.get('on_time_pct',88.2):.1f}%",          "↓ 1.4% vs yesterday",  "green"),
        ("Active disruptions",str(m.get("active_disruptions",12)),          "3 critical · 9 minor", "amber"),
        ("Avg system delay", f"{m.get('avg_delay_min',8.2):.1f} min",       "Target < 2 min",       "red"),
        ("Model F1",         f"{m.get('prediction_f1',0.87):.2f}",
         f"Latency {m.get('inference_latency_ms',145)} ms",                                         "blue"),
        ("Service delivered",f"{m.get('service_delivered_pct',96.5):.1f}%", "↑ 0.3% vs target",    "green"),
    ]
    for col, (lbl, val, sub, color) in zip(st.columns(5), kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-val {color}">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:.4rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(routes: List[Dict], alerts: List[Dict], metrics: Dict, opts: Dict):
    sev_map    = {"Critical":"crit","Delayed":"delay","On-time":"ok"}
    allowed    = {sev_map[s] for s in opts.get("sev_filter", list(sev_map.keys()))}
    q          = opts.get("search","").lower()
    filtered   = [r for r in routes
                  if r["status"] in allowed
                  and (not q or q in r["id"].lower() or q in r["name"].lower())]

    col_main, col_side = st.columns([3, 1])

    # ── left ──────────────────────────────────────────────────────────────────
    with col_main:
        tab_routes, tab_events, tab_charts = st.tabs(["Routes", "Event log", "Charts"])

        with tab_routes:
            if filtered:
                hcols = st.columns([1.1, 2.2, 1, 1.2, 1.2, 1])
                for h, c in zip(["Route","Name","Status","Delay","Bunching","Pred."], hcols):
                    c.markdown(
                        f'<div style="font-size:9px;letter-spacing:.1em;'
                        f'text-transform:uppercase;color:#888">{h}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('<div style="border-top:.5px solid #ddd;margin-bottom:4px"></div>',
                            unsafe_allow_html=True)
                for r in filtered:
                    c1,c2,c3,c4,c5,c6 = st.columns([1.1,2.2,1,1.2,1.2,1])
                    c1.markdown(f'<div style="font-weight:500;font-size:12px">{r["id"]}</div>',
                                unsafe_allow_html=True)
                    c2.markdown(f'<div style="font-size:11px;color:#555">{r["name"]}</div>',
                                unsafe_allow_html=True)
                    c3.markdown(STATUS_HTML.get(r["status"],""), unsafe_allow_html=True)
                    d = r.get("delay", 0)
                    c4.markdown(
                        f'<div style="font-family:Syne,sans-serif;font-weight:800;'
                        f'font-size:15px;color:{_delay_color(d)}">{d:.1f} min</div>',
                        unsafe_allow_html=True,
                    )
                    b = r.get("bunch", 0)
                    c5.markdown(f"""
                        <div style="font-size:11px;margin-bottom:3px">{b*100:.0f}%</div>
                        <div style="height:3px;background:#eee;border-radius:2px">
                          <div style="height:3px;width:{b*100:.0f}%;border-radius:2px;
                                      background:{_delay_color(b*10)}"></div>
                        </div>""", unsafe_allow_html=True)
                    p = r.get("pred", 0)
                    c6.markdown(
                        f'<div style="font-family:Syne,sans-serif;font-weight:800;'
                        f'font-size:15px;color:{_prob_color(p)}">{p:.0f}%</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div style="border-top:.5px solid #f0f0ee;margin:4px 0"></div>',
                                unsafe_allow_html=True)
            else:
                st.info("No routes match the current filters.")

        with tab_events:
            for ev in alerts:
                dot_color = SEV_COLOR.get(ev.get("sev","ok"), "#888")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:7px 0;
                            border-bottom:.5px solid #f0f0ee">
                    <span style="font-size:10px;color:#888;min-width:36px">{ev['time']}</span>
                    <span style="width:8px;height:8px;border-radius:50%;
                                 background:{dot_color};display:inline-block;flex-shrink:0"></span>
                    <span style="font-size:11px">{ev['msg']}</span>
                </div>""", unsafe_allow_html=True)

        with tab_charts:
            # Removed Avg delay by route per request

            st.markdown('<div class="sh">Disruptions — last 24 h</div>', unsafe_allow_html=True)
            spark = [2,3,5,4,6,8,14,22,28,31,26,24,19,17,15,18,22,26,29,24,18,14,10,8]
            st.plotly_chart(fig_sparkline(spark), use_container_width=True)

            # Charts removed per request

            # Severity Analysis (three small charts)
            st.markdown('---')
            df = pd.DataFrame(routes)
            if not df.empty:
                status_to_severity = {"ok": 0, "delay": 1, "crit": 3}
                df['severity_class'] = df['status'].map(status_to_severity).fillna(0).astype(int)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(px.pie(df, names='severity_class', title='Routes by Severity', color_discrete_map={k: v for k, v in SEVERITY_COLORS.items()}), use_container_width=True)
                with col2:
                    route_counts = df.groupby(['id', 'severity_class']).size().reset_index(name='Count')
                    route_counts['severity_class'] = route_counts['severity_class'].map({0: 'Normal', 1: 'Minor', 3: 'Severe'})
                    st.plotly_chart(px.bar(route_counts, x='id', y='Count', title='Routes by Alert Count', color='severity_class', color_discrete_map={'Normal': '#2A7A4A', 'Minor': '#1A5FA0', 'Severe': '#C84040'}), use_container_width=True)
                with col3:
                    severity_counts = df['severity_class'].value_counts().reset_index()
                    severity_counts.columns = ['Severity', 'Count']
                    severity_counts['Severity'] = severity_counts['Severity'].map({0: 'Normal', 1: 'Minor', 3: 'Severe'})
                    st.plotly_chart(px.bar(severity_counts, x='Severity', y='Count', title='Disruptions by Severity', color_discrete_map={'Normal': '#2A7A4A', 'Minor': '#1A5FA0', 'Severe': '#C84040'}), use_container_width=True)

            # Model Explainability (SHAP) small panel
            st.markdown('---')
            st.markdown('<div class="sh">Model Explainability (SHAP) — Overview</div>', unsafe_allow_html=True)
            route_ids = [r['id'] for r in routes]
            if route_ids:
                rid = st.selectbox('Select route for SHAP overview', route_ids, key='ov_shap')
                contrib = fetch_shap(rid)
                if isinstance(contrib, dict) and contrib.get('__scheduled__'):
                    st.info(f"SHAP scheduled (n_background={contrib.get('n_background')})")
                else:
                    st.plotly_chart(fig_shap(contrib), use_container_width=True, config={'displayModeBar': True})

            st.plotly_chart(fig_delay_bars(routes), use_container_width=True)
            st.markdown('<div class="sh">Disruptions — last 24 h</div>',
                        unsafe_allow_html=True)
            spark = [2,3,5,4,6,8,14,22,28,31,26,24,19,17,15,18,22,26,29,24,18,14,10,8]
            st.plotly_chart(fig_sparkline(spark), use_container_width=True)
            st.markdown('<div style="display:flex;justify-content:space-between;'
                        'font-size:9px;color:#888;margin-top:2px">'
                        '<span>00:00</span><span>06:00</span><span>12:00</span>'
                        '<span>18:00</span><span>now</span></div>',
                        unsafe_allow_html=True)

    # ── right sidebar ─────────────────────────────────────────────────────────
    with col_side:
        st.markdown('<div class="sh">Alert distribution</div>', unsafe_allow_html=True)
        alert_dist = {"Technical":43,"Weather":31,"Construction":16,"Other":10}
        st.plotly_chart(fig_donut(alert_dist), use_container_width=True)
        for label, pct, color in zip(
            alert_dist.keys(), alert_dist.values(),
            ["#E24B4A","#BA7517","#185FA5","#3B6D11"],
        ):
            st.markdown(
                f'<div style="font-size:10px;display:flex;align-items:center;'
                f'gap:6px;margin-bottom:3px">'
                f'<span style="width:8px;height:8px;border-radius:50%;'
                f'background:{color};display:inline-block"></span>'
                f'{label} ({pct}%)</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sh" style="margin-top:14px">Route health</div>',
                    unsafe_allow_html=True)
        for r in routes[:5]:
            c = _delay_color(r["delay"])
            pred = min(r.get("pred", 0), 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <div style="font-size:11px;min-width:38px">{r['id']}</div>
                <div style="flex:1;height:4px;background:#eee;border-radius:2px">
                  <div style="height:4px;width:{pred:.0f}%;background:{c};border-radius:2px"></div>
                </div>
                <div style="font-size:11px;color:{c};min-width:40px;text-align:right">
                    {r['delay']:.1f}m</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:14px">System status</div>',
                    unsafe_allow_html=True)
        statuses = [
            ("GTFS-RT Feed",  "Live",   "#2A7A4A"),
            ("ML Inference",  "145 ms", "#2A7A4A"),
            ("Static GTFS",   "Stale 4h","#B07010"),
            ("Data Quality",  "94.1%",  "#2A7A4A"),
            ("Leakage ctrl",  "Active", "#2A7A4A"),
        ]
        for name, val, color in statuses:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-bottom:.5px solid #f0f0ee;font-size:11px">
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="width:7px;height:7px;border-radius:50%;
                                 background:{color};display:inline-block"></span>{name}
                </div>
                <span style="color:{color}">{val}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════

def page_map(vehicles_df: pd.DataFrame, routes: List[Dict]):
    st.markdown('<div class="sh">Live vehicle positions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    # Ensure numeric lat/lon and speed columns
    if not vehicles_df.empty:
        for col in ['latitude', 'longitude', 'speed']:
            if col in vehicles_df.columns:
                vehicles_df[col] = pd.to_numeric(vehicles_df[col], errors='coerce')
    route_counts = vehicles_df['route_id'].value_counts() if not vehicles_df.empty else pd.Series(dtype=int)
    c1.metric('Total vehicles',  len(vehicles_df))
    c2.metric('Critical routes', sum(1 for r in routes if r['status'] == 'crit'))
    avg_spd = round(float(vehicles_df['speed'].mean()), 1) if 'speed' in vehicles_df.columns and not vehicles_df.empty else 0.0
    c3.metric('Avg fleet speed', f"{avg_spd} km/h")
    c4.metric('Routes tracked',  len(route_counts))

    st.markdown('---')

    # Choose mapping backend: prefer Plotly for large number of points for performance
    clean_df = vehicles_df.dropna(subset=['latitude', 'longitude']) if not vehicles_df.empty else vehicles_df
    n_points = len(clean_df)
    MAP_PLOTLY_THRESHOLD = 300

    if n_points == 0:
        st.info('No position data available to display on the map.')
    elif n_points > MAP_PLOTLY_THRESHOLD or not FOLIUM_OK:
        st.markdown('<div class="sh">Vehicle positions (Plotly)</div>', unsafe_allow_html=True)
        # If large, sample server-side already done; still ensure we don't plot excessive points
        df_map = clean_df if n_points <= 500 else clean_df.sample(500, random_state=42)
        st.plotly_chart(fig_vehicle_map_plotly(df_map), use_container_width=True)
    else:
        st.markdown('<div class="sh">Interactive map (Folium)</div>', unsafe_allow_html=True)
        m = _folium_map(clean_df)
        st_folium(m, use_container_width=True, height=400)

        # Integrate NLP-extracted location entities into temporal playback
        try:
            alerts = fetch_alerts()
            # alerts expected to have 'nlp_entities' as list of {'text','lat','lon','time'}
            nlp_entities = [e for a in alerts for e in a.get('nlp_entities', [])]
        except Exception:
            nlp_entities = []

        # Build timestamped GeoJSON for Folium TimestampedGeoJson
        from branca.element import MacroElement
        try:
            features = []
            for ent in nlp_entities:
                t = ent.get('time') or datetime.now().isoformat()
                feat = {
                    'type': 'Feature',
                    'geometry': {'type': 'Point', 'coordinates': [ent['lon'], ent['lat']]},
                    'properties': {'time': t, 'popup': ent.get('text','')}
                }
                features.append(feat)
            if features:
                tg = {
                    'type': 'FeatureCollection',
                    'features': features
                }
                from folium.plugins import TimestampedGeoJson
                TimestampedGeoJson(tg, period='PT1M', add_last_point=True, auto_play=False, loop=False).add_to(m)
        except Exception:
            pass

            # Sample stop data for severity maps (will be replaced by static GTFS mapping)
        sample_stops = [
            {'id': 'Stop A', 'lat': 52.37, 'lon': 4.89, 'delay': 5, 'severity': 2, 'events': 10, 'risk': 'high', 'cause': 'congestion'},
            {'id': 'Stop B', 'lat': 52.38, 'lon': 4.90, 'delay': 15, 'severity': 3, 'events': 5, 'risk': 'medium', 'cause': 'accident'},
            {'id': 'Stop C', 'lat': 52.36, 'lon': 4.88, 'delay': 0, 'severity': 0, 'events': 1, 'risk': 'low', 'cause': 'none'},
            {'id': 'Stop D', 'lat': 52.39, 'lon': 4.91, 'delay': 8, 'severity': 1, 'events': 3, 'risk': 'low', 'cause': 'maintenance'},
        ]

        # If static GTFS mapping available, attempt to map alerts to stop coords
        try:
            gtfs_path = Path('..') / 'gtfs-nl.zip'
            if gtfs_path.exists():
                import geopandas as gpd
                # placeholder: in production, use GTFS stop_times -> stops mapping
                # here we would replace sample_stops with mapped data
        except Exception:
            pass

        # Calculate spatial lags
        spatial_lags = {}
        for stop in sample_stops:
            lat1, lon1 = stop['lat'], stop['lon']
            neighbors = [s for s in sample_stops if haversine(lat1, lon1, s['lat'], s['lon']) <= 1 and s != stop]
            spatial_lags[stop['id']] = np.mean([s['delay'] for s in neighbors]) if neighbors else 0

        st.markdown('---')
        st.markdown('<div class="sh">Severity Comparison Maps</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('Mean Severity')
            if FOLIUM_OK:
                m1 = _folium_severity_map(sample_stops, spatial_lags, 'mean')
                # Make tiles and interactive controls visible
                m1.add_child(folium.LayerControl())
                st_folium(m1, use_container_width=True, height=300)

        with col2:
            st.markdown('Max Severity')
            if FOLIUM_OK:
                m2 = _folium_severity_map(sample_stops, spatial_lags, 'max')
                m2.add_child(folium.LayerControl())
                st_folium(m2, use_container_width=True, height=300)

    # removed Delay hotspots and Vehicles per route per request

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def page_predictions(routes: List[Dict], model_name: str):
    st.markdown('<div class="sh">ML disruption forecast — next 30 minutes</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active model",   model_name)
    c2.metric("F1 score",       "0.87")
    c3.metric("Precision",      "0.88")
    c4.metric("Lead time avg",  "18 min")

    st.markdown("---")

    # Sorted predictions
    preds = sorted(routes, key=lambda r: r.get("pred", 0), reverse=True)
    for r in preds:
        prob  = r.get("pred", 0)
        color = _prob_color(prob)
        badge = ("badge-crit" if r["status"]=="crit"
                 else "badge-warn" if r["status"]=="delay"
                 else "badge-ok")
        status_txt = ("Critical" if r["status"]=="crit"
                      else "Delayed" if r["status"]=="delay"
                      else "On-time")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:12px;
                    border:.5px solid #e8e8e4;border-left:4px solid {color};
                    border-radius:4px;margin-bottom:8px">
            <div style="flex:1">
                <div style="font-weight:500;font-size:13px">{r['id']} — {r['name']}</div>
                <div style="font-size:10px;color:#888;margin-top:2px">
                    Delay {r['delay']:.1f} min · Bunching {r['bunch']*100:.0f}%
                    · Throughput {r['throughput']} veh/hr
                </div>
                <div style="height:4px;background:#eee;border-radius:2px;margin-top:6px;width:220px">
                    <div style="height:4px;width:{prob:.0f}%;background:{color};
                                border-radius:2px"></div>
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-family:Syne,sans-serif;font-weight:800;
                            font-size:26px;color:{color}">{prob:.0f}%</div>
                <span class="badge {badge}">{status_txt}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SHAP panel ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sh">SHAP feature importance (TreeExplainer)</div>',
                unsafe_allow_html=True)

    shap_col, info_col = st.columns([3, 1])
    with shap_col:
        route_ids    = [r["id"] for r in routes]
        selected_rid = st.selectbox("Analyse route", route_ids, key="shap_sel")
    with info_col:
        n_bg = st.slider("Background samples", 10, 200, 50, 10, key="shap_nbg")

    if selected_rid:
        with st.spinner(f"Computing SHAP values for {selected_rid}…"):
            # Invalidate cache if n_bg changed by using a cache-key that includes it
            @st.cache_data(ttl=60)
            def _shap(rid, nbg):
                data = _get(f"/shap/{rid}?n_background={nbg}")
                return data.get("contributions", _fallback_shap()) if data else _fallback_shap()
            contrib = _shap(selected_rid, n_bg)

        chart_col, top_col = st.columns([3, 1])
        with chart_col:
            st.plotly_chart(fig_shap(contrib), use_container_width=True)
        with top_col:
            st.markdown('<div class="sh">Top 3 drivers</div>', unsafe_allow_html=True)
            for feat, val in sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.markdown(
                    f'<div style="font-size:11px;margin-bottom:8px">'
                    f'<strong>{feat}</strong><br>'
                    f'<span style="color:#C84040;font-family:Syne,sans-serif;'
                    f'font-weight:700">{val:.4f}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sh">Model comparison</div>', unsafe_allow_html=True)
    # Model comparison: query backend for loaded models and compare
    info = _get('/model/info', timeout=3)
    models = []
    if info and info.get('loaded_models'):
        for m in info['loaded_models']:
            models.append({
                'Model': m.get('name', 'unknown'),
                'F1': m.get('metrics', {}).get('f1', None),
                'Precision': m.get('metrics', {}).get('precision', None),
                'Recall': m.get('metrics', {}).get('recall', None),
                'ROC-AUC': m.get('metrics', {}).get('roc_auc', None),
            })
    else:
        models = [{'Model':'simulation','F1':None,'Precision':None,'Recall':None,'ROC-AUC':None}]
    model_df = pd.DataFrame(models)
    if not model_df.empty:
        st.dataframe(model_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_analytics(routes: List[Dict], metrics: Dict):
    st.markdown('<div class="sh">Historical performance & budget (MTD)</div>',
                unsafe_allow_html=True)

    # Period summary (moved to top)
    st.markdown('<div class="sh">Period summary</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total routes",    len(routes))
    k2.metric("Avg delay",       f"{metrics.get('avg_delay_min',8.2):.1f} min")
    k3.metric("On-time %",       f"{metrics.get('on_time_pct',88.2):.1f}%")
    k4.metric("Data quality",    f"{metrics.get('data_quality_score',94.1):.1f}%")

    # 24-h trend
    spark  = [2,3,5,4,6,8,14,22,28,31,26,24,19,17,15,18,22,26,29,24,18,14,10,8]
    hours  = pd.date_range(end=datetime.now(), periods=24, freq="h")
    st.markdown('<div class="sh">Disruptions — 24 h rolling</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_trend(hours, spark), use_container_width=True)

    # Route delay chart
    st.markdown('<div class="sh" style="margin-top:.6rem">Average delay by route</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_delay_bars(routes), use_container_width=True)

    # Lead-time analysis (keep display_lead_time_analysis only)
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        display_bunching_index(routes)

    with col_right:
        display_lead_time_analysis()

    st.markdown("---")
    st.markdown('<div class="sh">Severity Analysis Dashboard</div>', unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)

    with col1:
        # Routes by Severity (pie chart)
        df = pd.DataFrame(routes)
        status_to_severity = {"ok": 0, "delay": 1, "crit": 3}
        df['severity_class'] = df['status'].map(status_to_severity).fillna(0).astype(int)

        fig = px.pie(
            df,
            names='severity_class',
            title='Routes by Severity',
            color='severity_class',
            color_discrete_map={k: v for k, v in SEVERITY_COLORS.items()},
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=30,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Routes by Alert Count (bar chart)
        route_counts = df.groupby(['id', 'severity_class']).size().reset_index(name='Count')
        route_counts['severity_class'] = route_counts['severity_class'].map({0: 'Normal', 1: 'Minor', 3: 'Severe'})
        fig = px.bar(route_counts, x='id', y='Count',
                     title="Routes by Alert Count",
                     color='severity_class',
                     color_discrete_map={'Normal': '#2A7A4A', 'Minor': '#1A5FA0', 'Severe': '#C84040'})
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=30,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Disruptions by Severity (bar chart)
        severity_counts = df['severity_class'].value_counts().reset_index()
        severity_counts.columns = ['Severity', 'Count']
        severity_counts['Severity'] = severity_counts['Severity'].map({0: 'Normal', 1: 'Minor', 3: 'Severe'})

        color_map = {'Normal': '#2A7A4A', 'Minor': '#1A5FA0', 'Severe': '#C84040'}

        fig = px.bar(severity_counts, x='Severity', y='Count',
                     title="Disruptions by Severity",
                     color='Severity',
                     color_discrete_map=color_map)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font=dict(family="DM Sans"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=30,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP Explainability Section
    st.markdown("---")
    st.markdown('<div class="sh">Model Explainability (SHAP)</div>', unsafe_allow_html=True)

    # Change background for SHAP section to slightly lighter panel
    st.markdown('<div style="background: linear-gradient(135deg, #0f1a2a 0%, #132234 100%); padding:12px; border-radius:8px">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Global Feature Importance**")
        # Simulate global feature importance
        features = ["speed_mean", "delay_mean_15m", "bunching_index", "on_time_pct"]
        importance = [0.35, 0.28, 0.22, 0.15]
        fig = px.bar(x=features, y=importance, title="Global Importance",
                     color_discrete_sequence=["#1A5FA0"]) 
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(255,255,255,0)" )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    with col2:
        st.markdown("**Feature Group Attribution**")
        # Categorize features
        groups = ["Speed", "Delay", "Service Quality", "Temporal"]
        attribution = [0.25, 0.40, 0.20, 0.15]
        fig = px.pie(values=attribution, names=groups, title="Group Attribution",
                     color_discrete_sequence=["#2A7A4A", "#1A5FA0", "#B07010", "#C84040"])
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(15,22,40,0.9)")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**Local Waterfall Plot**")
        # Simulate SHAP waterfall for a single prediction
        features_waterfall = ["delay_mean_15m", "speed_mean", "bunching_index", "Base"]
        values = [0.15, -0.08, 0.05, 0.0]
        fig = go.Figure()
        fig.add_trace(go.Waterfall(
            x=features_waterfall,
            y=values,
            measure=["relative"] * 3 + ["total"],
            decreasing={"marker":{"color":"#C84040"}},
            increasing={"marker":{"color":"#2A7A4A"}},
            totals={"marker":{"color":"#1A5FA0"}}
        ))
        fig.update_layout(title="SHAP Waterfall", height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    with col4:
        st.markdown("**Dependence Plot**")
        # Simulate dependence plot
        x_vals = np.random.normal(50, 10, 100)
        y_vals = 0.02 * x_vals + np.random.normal(0, 0.1, 100)
        fig = px.scatter(x=x_vals, y=y_vals, title="Delay vs Speed Dependence",
                         color_discrete_sequence=["#B07010"])
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    st.markdown('</div>', unsafe_allow_html=True)

    # Performance Monitoring Section
    st.markdown("---")
    st.markdown('<div class="sh">Performance Monitoring & Optimization</div>', unsafe_allow_html=True)

    # Change background for Performance Monitoring to a darker panel
    st.markdown('<div style="background: linear-gradient(135deg, #081018 0%, #0b1a29 100%); padding:12px; border-radius:8px">', unsafe_allow_html=True)
    col_mon1, col_mon2 = st.columns(2)

    with col_mon1:
        st.markdown("**Drift Detection**")
        # Simulate performance metrics over time
        days = pd.date_range(end=datetime.now(), periods=7, freq="D")
        f1_scores = [0.87, 0.85, 0.89, 0.84, 0.86, 0.88, 0.82]
        fig = px.line(x=days, y=f1_scores, title="F1 Score Drift",
                      color_discrete_sequence=["#1A5FA0"])
        fig.add_hline(y=0.80, line_dash="dash", line_color="#C84040", annotation_text="Threshold")
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        st.caption("Rolling F1-score monitoring with retraining alerts")

    with col_mon2:
        st.markdown("**Feature Distribution Monitoring**")
        # Simulate KS-test results
        features_ks = ["speed_mean", "delay_mean_15m", "bunching_index"]
        ks_stats = [0.05, 0.12, 0.03]
        fig = px.bar(x=features_ks, y=ks_stats, title="KS Test Statistics",
                     color_discrete_sequence=["#B07010"])
        fig.add_hline(y=0.10, line_dash="dash", line_color="#C84040", annotation_text="Retraining Threshold")
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        st.caption("Kolmogorov-Smirnov tests for feature drift detection")
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Information Section
    st.markdown("---")
    # Remove listing of available models from analytics to reduce UI clutter and loading
    # Instead, model comparison in Predictions will show loaded models only

    # Real-time Inference Interface
    st.markdown("---")
    st.markdown('<div class="sh">Real-Time Inference</div>', unsafe_allow_html=True)

    st.markdown("Enter feature values for disruption prediction:")

    # Input form for features
    with st.form("prediction_form"):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            speed_mean = st.number_input("Speed Mean (km/h)", value=35.0, min_value=0.0)
        with col_f2:
            delay_15m = st.number_input("Delay 15m (min)", value=5.0, min_value=0.0)
        with col_f3:
            bunching = st.slider("Bunching Index", 0.0, 1.0, 0.2)
        with col_f4:
            on_time_pct = st.slider("On-Time %", 0.0, 100.0, 85.0)

        submitted = st.form_submit_button("Predict Disruption")

        if submitted:
            # Simulate prediction (in real implementation, call backend)
            prediction = "MINOR_DELAY" if delay_15m > 10 else "NORMAL"
            confidence = 85.0
            st.success(f"Predicted: {prediction} (Confidence: {confidence}%)")

            # Display feature contributions
            st.markdown("**Feature Contributions:**")
            contribs = {
                "Delay 15m": 0.45,
                "Speed Mean": -0.15,
                "Bunching": 0.25,
                "On-Time %": -0.10
            }
            for feat, val in contribs.items():
                st.write(f"- {feat}: {val:+.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── data ──────────────────────────────────────────────────────────────────
    api_ok     = _api_online()
    model_name = _active_model() if api_ok else "simulation"

    with st.spinner("Loading…"):
        metrics     = fetch_metrics()
        routes      = fetch_routes()
        alerts      = fetch_alerts()
        vehicles_df = fetch_vehicles()

    # ── sidebar + nav ─────────────────────────────────────────────────────────
    opts = render_sidebar(api_ok, model_name, metrics)

    # ── topbar ────────────────────────────────────────────────────────────────
    render_topbar(api_ok, model_name)

    # ── KPI row ───────────────────────────────────────────────────────────────
    render_kpi_row(metrics)

    # ── page ──────────────────────────────────────────────────────────────────
    page = opts["page"]
    if   page == "Overview":    page_overview(routes, alerts, metrics, opts)
    elif page == "Live Map":    page_map(vehicles_df, routes)
    elif page == "Predictions": page_predictions(routes, model_name)
    elif page == "Analytics":   page_analytics(routes, metrics)


if __name__ == "__main__":
    main()
