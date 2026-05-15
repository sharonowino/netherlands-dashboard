import joblib, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print('ROOT', ROOT)
models_dir = ROOT / 'transit_dashboard' / 'models'
print('MODELS_DIR', models_dir)
for p in sorted(models_dir.glob('*')):
    if not p.is_file():
        continue
    print('\n---', p.name)
    try:
        obj = joblib.load(p)
        print('type:', type(obj).__name__)
        if isinstance(obj, dict):
            print('keys:', sorted(list(obj.keys())))
            m = obj.get('model') or obj.get('trained_model')
            if m is not None:
                print('wrapped model type:', type(m).__name__)
                for attr in ('n_features_in_', 'feature_names_in_', 'classes_', 'n_outputs_'):
                    print(attr, getattr(m, attr, None))
        else:
            for attr in ('n_features_in_', 'feature_names_in_', 'classes_', 'n_outputs_'):
                try:
                    print(attr, getattr(obj, attr, None))
                except Exception as e:
                    print(attr, 'err', e)
    except Exception as e:
        print('load error:', e)

# Also instantiate ModelRegistry
try:
    from transit_dashboard.backend.main import ModelRegistry
    r = ModelRegistry()
    print('\n=== Registry info ===')
    print(json.dumps(r.info(), indent=2))
    print('loaded active_name=', r.active_name, 'loaded=', r.loaded)
    print('rf_meta present=', bool(getattr(r, 'rf_meta', None)), 'xgb_meta present=', bool(getattr(r, 'xgb_meta', None)))
    print('rf_model type=', type(r.rf_model).__name__ if r.rf_model is not None else None)
    print('xgb_model type=', type(r.xgb_model).__name__ if r.xgb_model is not None else None)
    print('expected_n_features=', getattr(r, 'expected_n_features', None))
except Exception as e:
    print('Registry instantiation error:', e)
