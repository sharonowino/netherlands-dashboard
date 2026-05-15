# TransitOS — Netherlands Transit Disruption Dashboard

Real-time ML-powered disruption monitoring.  
RandomForest primary model · XGBoost fallback · Rule-based simulation final fallback  
GTFS-RT live feed · Sample data fallback · SHAP TreeExplainer · Folium map · Auto-refresh

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    docker-compose.yml                        │
│                                                             │
│  ┌─────────────────────┐      ┌──────────────────────────┐  │
│  │   FastAPI backend   │      │   Streamlit frontend     │  │
│  │   :8000             │◄─────│   :8501                  │  │
│  │                     │      │                          │  │
│  │  GET  /health        │      │  Overview  page         │  │
│  │  GET  /model/info    │      │  Live Map  page         │  │
│  │  POST /model/reload  │      │  Predictions page       │  │
│  │  GET  /feed          │      │  Analytics page         │  │
│  │  GET  /metrics       │      │                          │  │
│  │  GET  /routes        │      │  API_BASE_URL=           │  │
│  │  GET  /alerts        │      │  http://backend:8000     │  │
│  │  POST /predict       │      └──────────────────────────┘  │
│  │  POST /predict/batch │                                    │
│  │  GET  /shap/{id}     │                                    │
│  └─────────────────────┘                                    │
│           ▲                                                  │
│     GTFS-RT ovapi.nl          models/*.pkl                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
cd transit_dashboard

# Optional — drop your trained models in before building
cp /path/to/model_RandomForest.pkl  models/
cp /path/to/model_XGBoost.pkl       models/   # optional secondary
cp /path/to/scaler_latest.pkl       models/   # optional

docker compose up --build

# Dashboard  → http://localhost:8501
# API docs   → http://localhost:8000/docs
```

---

## File structure

```
transit_dashboard/
├── docker-compose.yml
├── README.md
├── models/                        ← mount point for .pkl files
│   └── .gitkeep
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                    ← FastAPI (all endpoints)
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                     ← Streamlit (all 4 pages)
```

---

## Backend endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model health |
| GET | `/model/info` | Active model, features, load time |
| POST | `/model/reload` | Hot-reload .pkl without restart |
| GET | `/feed` | Vehicle positions — GTFS-RT → sample |
| GET | `/metrics` | Aggregated KPIs |
| GET | `/routes` | Route statuses + live vehicle counts |
| GET | `/alerts` | Disruption events (live + curated) |
| POST | `/predict` | Single-route prediction |
| POST | `/predict/batch` | Batch predictions (max 50 routes) |
| GET | `/shap/{route_id}` | SHAP TreeExplainer contributions |

Full interactive docs: **http://localhost:8000/docs**

---

## ML model

### Priority order
1. **RandomForest** — `models/model_RandomForest.pkl` (primary, user-specified)
2. **XGBoost**      — `models/model_XGBoost.pkl`      (secondary fallback)
3. **Rule-based simulation**                           (final fallback, no file needed)

### Features (12, all backward-looking — no temporal leakage)

| Feature | Description |
|---------|-------------|
| `speed_mean` | km/h average over rolling window |
| `speed_std` | km/h standard deviation |
| `delay_mean_5m` | seconds, 5-min rolling average |
| `delay_mean_15m` | seconds, 15-min rolling average |
| `delay_mean_30m` | seconds, 30-min rolling average |
| `bunching_index` | 0–1, fraction of vehicles bunched |
| `on_time_pct` | 0–1, fraction of trips on-time |
| `headway_variance` | seconds², schedule regularity |
| `alert_nlp_score` | 0–1, NLP severity of service alerts |
| `alert_count` | number of active alerts on route |
| `fleet_utilization` | 0–1, fraction of planned vehicles active |
| `speed_drop_ratio` | 0–1, fraction of stops below speed threshold |

### Output classes
| Class | Label | Color |
|-------|-------|-------|
| 0 | NORMAL   | Green  |
| 1 | MINOR    | Blue   |
| 2 | MODERATE | Amber  |
| 3 | SEVERE   | Red    |

---

## Frontend pages

| Page | Features |
|------|----------|
| **Overview** | KPI row, filterable route table, event log, alert donut, system status panel |
| **Live Map** | Folium interactive map (vehicle positions), Plotly fallback, delay hotspots |
| **Predictions** | 30-min RF forecasts per route, SHAP TreeExplainer chart, model comparison table |
| **Analytics** | 24-h disruption trend, budget vs actuals (MTD), period KPIs, lead-time histogram |

---

## Environment variables

### Backend
| Variable | Default | Description |
|----------|---------|-------------|
| `RF_MODEL_PATH` | `models/model_RandomForest.pkl` | RandomForest pickle |
| `XGB_MODEL_PATH` | `models/model_XGBoost.pkl` | XGBoost pickle |
| `SCALER_PATH` | `models/scaler_latest.pkl` | sklearn scaler pickle |
| `GTFS_RT_URL` | `http://gtfs.ovapi.nl/nl/vehiclePositions.pb` | Positions feed |
| `GTFS_RT_ALERTS_URL` | `http://gtfs.ovapi.nl/nl/alerts.pb` | Alerts feed |

### Frontend
| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://backend:8000` | FastAPI base URL |
| `REFRESH_INTERVAL` | `30` | Cache TTL + auto-refresh interval (seconds) |

---

## Development (without Docker)

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 streamlit run app.py
```

---

## Hot-reloading a new model

```bash
# Copy new pkl into the mounted volume
cp new_model_RandomForest.pkl transit_dashboard/models/model_RandomForest.pkl

# Trigger reload — no container restart needed
curl -X POST http://localhost:8000/model/reload

# Confirm
curl http://localhost:8000/model/info
```

Or click **"⟳ Reload model"** in the dashboard sidebar.

---

## 7-Layer architecture mapping

| Layer | Component |
|-------|-----------|
| L1 Data Ingestion | `/feed` + `/alerts` → GTFS-RT ovapi.nl |
| L2 Preprocessing | `_enrich_routes`, sample fallback generation |
| L3 Feature Engineering | `RouteFeatures` Pydantic schema, 12 backward-looking features |
| L4 Leakage Control | No alert/future features in prediction vector; all windows backward-only |
| L5 Model Training | RF `.pkl` loaded at startup; XGBoost secondary; simulation final fallback |
| L6 Evaluation | SHAP TreeExplainer, `/predict/batch` returns F1/latency/cost summary |
| L7 Deployment | Streamlit + FastAPI + Docker Compose + hot-reload |
