#!/usr/bin/env python3
"""Run equal-288, independently retrained K=1 Inspire-DRO/GenDex ablations.

Same stages as run_inspire_pd288_ablations.py, with --variant k1-equal9 fixed.
The first stage is `train --line all`; training is shared by both generators.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DRO import run_inspire_pd288_ablations as ablation


if __name__ == "__main__":
    if "--variant" in sys.argv:
        raise ValueError("K1 entry point fixes --variant k1-equal9")
    sys.argv[2:2] = ["--variant", "k1-equal9"]
    ablation.main()
