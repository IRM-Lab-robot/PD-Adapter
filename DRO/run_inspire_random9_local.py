#!/usr/bin/env python3
"""Inspire-DRO/GenDex local Random-9 on ten objects × 64 source grasps.

Use the same eight frozen wide-position random starts as spatial Random-9,
but search each of the nine branches using local P32 Gaussian proposals.
Pass --all-objects for the full line; otherwise default to contactdb+banana.
Search, one-time recheck, and per-line summary have independent output paths.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DRO import run_inspire_random9_spatial as run

WIDE_OUT = run.OUT
run.__doc__ = __doc__
run.OUT = ROOT / "results/supplement_inspire_random9_local_b64"
run.engine.OUT = run.OUT
run.engine.METHOD = "Random-9-Local/288"
run.engine.WORKER_ENTRYPOINT = Path(__file__).resolve()
run.WARM_OUT = WIDE_OUT
run.WARM_METHOD = "Random-9-Spatial/288"
run.SPATIAL_MODE = "local"
run.SEARCH_TAG = "random9_local"


if __name__ == "__main__":
    run.main()
