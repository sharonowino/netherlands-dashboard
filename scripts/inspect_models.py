import os
import sys
import json
try:
    import joblib
except Exception:
    joblib = None


def inspect_models(models_dir):
    out = []
    if not os.path.isdir(models_dir):
        print(json.dumps({"error":"models dir not found","path":models_dir}))
        return
    for fn in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, fn)
        if not os.path.isfile(path):
            continue
        if not (fn.endswith('.pkl') or fn.endswith('.joblib')):
            continue
        info = {"file": fn}
        if joblib is None:
            info['error'] = 'joblib not installed'
            out.append(info)
            continue
        try:
            m = joblib.load(path)
            info['type'] = type(m).__name__
            info['n_features_in_'] = getattr(m, 'n_features_in_', None)
            info['feature_names_in_'] = getattr(m, 'feature_names_in_', None)
            info['has_predict'] = callable(getattr(m, 'predict', None))
        except Exception as e:
            info['load_error'] = str(e)
        out.append(info)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--models-dir', default='transit_dashboard/models')
    args = p.parse_args()
    inspect_models(args.models_dir)
