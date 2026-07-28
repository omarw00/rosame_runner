import os
import sys

_MA_SAM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs", "ma-sam")

if _MA_SAM_ROOT not in sys.path:
    sys.path.append(_MA_SAM_ROOT)
