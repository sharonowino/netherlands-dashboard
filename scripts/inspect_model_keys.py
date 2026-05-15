import joblib, os, json
from pathlib import Path
MD = Path('transit_dashboard/models')
out = []
for p in sorted(MD.glob('*')):
    if not p.is_file():
        continue
    info = {'file': str(p.name)}
    try:
        obj = joblib.load(p)
        info['type'] = type(obj).__name__
        if isinstance(obj, dict):
            info['keys'] = sorted(list(obj.keys()))
            # show sizes for arrays if present
            info['summary'] = {k: (type(v).__name__, (len(v) if hasattr(v,'__len__') else None)) for k,v in list(obj.items())[:10]}
    except Exception as e:
        info['error'] = str(e)
    out.append(info)
print(json.dumps(out, indent=2))
