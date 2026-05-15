import json
import plotly.graph_objects as go
from importlib import import_module
from pathlib import Path

# Import frontend module (contains fig_shap)
frontend = import_module('transit_dashboard.frontend.app')

# Generate a sample figure using fig_shap
sample = {'bunching_index':0.42,'delay_mean_15m':0.35,'speed_drop_ratio':0.31,'on_time_pct':0.28,'delay_mean_5m':0.25,'speed_mean':0.22,'delay_mean_30m':0.20,'headway_variance':0.18,'fleet_utilization':0.15,'alert_nlp_score':0.12,'speed_std':0.10,'alert_count':0.08}
fig = frontend.fig_shap(sample)
# Basic assertions
assert isinstance(fig, go.Figure)
assert len(fig.data) == 1
# Save snapshot JSON for visual diffing in CI
snap_dir = Path(__file__).parent
snap_dir.mkdir(parents=True, exist_ok=True)
with open(snap_dir / 'fig_shap_snapshot.json','w') as f:
    f.write(fig.to_json())
