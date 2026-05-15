import pytest
from transit_dashboard.backend import main as md
import numpy as np


def test_rule_simulate_thresholds():
    from transit_dashboard.backend.main import RouteFeatures, _rule_simulate
    r_high_bunch = RouteFeatures(route_id='R1', bunching_index=0.8, speed_mean=35,
                                 delay_mean_15m=50, delay_mean_5m=30, delay_mean_30m=70,
                                 speed_std=5, on_time_pct=0.8, headway_variance=10,
                                 alert_nlp_score=0.1, alert_count=0, fleet_utilization=0.9, speed_drop_ratio=0.05)
    sc, conf = _rule_simulate(r_high_bunch)
    assert sc == 3

    r_moderate = RouteFeatures(route_id='R2', bunching_index=0.5, speed_mean=30,
                                delay_mean_15m=120, delay_mean_5m=30, delay_mean_30m=70,
                                speed_std=5, on_time_pct=0.8, headway_variance=10,
                                alert_nlp_score=0.1, alert_count=0, fleet_utilization=0.9, speed_drop_ratio=0.05)
    sc2, _ = _rule_simulate(r_moderate)
    assert sc2 == 2

    r_minor = RouteFeatures(route_id='R3', bunching_index=0.25, speed_mean=40,
                            delay_mean_15m=40, delay_mean_5m=30, delay_mean_30m=70,
                            speed_std=5, on_time_pct=0.8, headway_variance=10,
                            alert_nlp_score=0.1, alert_count=0, fleet_utilization=0.9, speed_drop_ratio=0.05)
    sc3, _ = _rule_simulate(r_minor)
    assert sc3 == 1

    r_normal = RouteFeatures(route_id='R4', bunching_index=0.1, speed_mean=50,
                              delay_mean_15m=10, delay_mean_5m=5, delay_mean_30m=20,
                              speed_std=3, on_time_pct=0.95, headway_variance=5,
                              alert_nlp_score=0.0, alert_count=0, fleet_utilization=0.95, speed_drop_ratio=0.01)
    sc4, _ = _rule_simulate(r_normal)
    assert sc4 == 0


def test_predict_fallback_simulation():
    # Force registry to have no models
    md.registry.rf_model = None
    md.registry.xgb_model = None
    md.registry.active_model = None
    md.registry.active_name = "simulation"

    rf = md.RouteFeatures(route_id='R-01')
    res = md._predict_one(rf)
    assert res.source == 'simulation'
    assert 0 <= res.severity_class <= 3


def test_build_metrics_and_sample_vehicles():
    vehicles = [
        {'vehicle_id':'v1','speed':22},
        {'vehicle_id':'v2','speed':18},
        {'vehicle_id':'v3','speed':35},
    ]
    metrics = md._build_metrics(vehicles)
    # on_time_pct = percent speeds > 20 => 2/3 => 66.7
    assert pytest.approx(metrics['on_time_pct'], rel=1e-3) == 66.7
    assert 'model_active' in metrics

    # empty vehicles should return defaults
    metrics2 = md._build_metrics([])
    assert 'on_time_pct' in metrics2

    # sample vehicles length
    sv = md._sample_vehicles(10)
    assert len(sv) == 10

    # more precise check for avg_delay formula
    # avg_spd = mean([22,18,35]) = 25.0 -> avg_delay = max(0, round((35-25.0)*0.4,1)) = 4.0
    assert pytest.approx(metrics['avg_delay_min'], rel=1e-6) == 4.0


def test_enrich_routes_counts():
    vehicles = [
        {'vehicle_id':'a','route_id':'R-01'},
        {'vehicle_id':'b','route_id':'R-01'},
        {'vehicle_id':'c','route_id':'R-04'},
    ]
    enriched = md._enrich_routes(vehicles)
    r01 = next((r for r in enriched if r['id']=='R-01'), None)
    r04 = next((r for r in enriched if r['id']=='R-04'), None)
    assert r01 is not None and r01['vehicle_count'] == 2
    assert r04 is not None and r04['vehicle_count'] == 1


def test_haversine_zero_and_nonzero():
    dist0 = md.haversine(52.0, 4.0, 52.0, 4.0)
    assert pytest.approx(dist0, rel=1e-6) == 0
    # small known distance: ~111 km per degree lat
    d = md.haversine(51.0, 4.0, 52.0, 4.0)
    assert d > 110 and d < 112


def test_get_merged_feed_sample_structure():
    out = md.get_merged_feed(use_live=False)
    assert 'source' in out
    assert out['source'] in ('sample','sample_pending_features','live','live_pending_features')
    assert 'rows' in out
    assert isinstance(out['rows'], list)


def test_get_shap_poll_returns_contributions_keys():
    # poll=True should compute (or return mock) contributions
    res = md.get_shap('R-01', n_background=10, poll=True)
    assert 'contributions' in res
    contrib = res['contributions']
    for fn in md.FEATURE_NAMES:
        assert fn in contrib
