import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transit_dashboard.backend import main
import json

try:
    out = main.get_merged_feed(use_live=False)
    import pprint
    pprint.pprint(out)

except Exception:
    import traceback
    traceback.print_exc()
