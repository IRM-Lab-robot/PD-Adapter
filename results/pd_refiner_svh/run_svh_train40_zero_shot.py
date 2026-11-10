#!/usr/bin/env python3
"""Generate SVH train40 DRO zero-shot sources one object at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


NEW_ROOT = Path("/home/user/yangbo_program/new")
SCRIPT = NEW_ROOT / "results/pd_refiner_svh/generate_svh_dro_zero_shot.py"
OUT_DIR = NEW_ROOT / "results/pd_refiner_svh/train40_zero_shot_b60_objects"
OUT = NEW_ROOT / "results/pd_refiner_svh/svh_train40_zero_shot_b60.pt"
SUMMARY = NEW_ROOT / "results/pd_refiner_svh/svh_train40_zero_shot_b60_summary.json"

OBJECTS = [
    "contactdb+airplane",
    "contactdb+alarm_clock",
    "contactdb+apple",
    "contactdb+binoculars",
    "contactdb+camera",
    "contactdb+cell_phone",
    "contactdb+cube_large",
    "contactdb+cube_medium",
    "contactdb+cube_small",
    "contactdb+cylinder_large",
    "contactdb+cylinder_medium",
    "contactdb+cylinder_small",
    "contactdb+elephant",
    "contactdb+eyeglasses",
    "contactdb+flashlight",
    "contactdb+hammer",
    "contactdb+headphones",
    "contactdb+light_bulb",
    "contactdb+mouse",
    "contactdb+pan",
    "contactdb+piggy_bank",
    "contactdb+ps_controller",
    "contactdb+pyramid_large",
    "contactdb+pyramid_medium",
    "contactdb+pyramid_small",
    "contactdb+rubber_duck",
    "contactdb+sphere_large",
    "contactdb+sphere_medium",
    "contactdb+sphere_small",
    "contactdb+stanford_bunny",
    "contactdb+stapler",
    "contactdb+toothpaste",
    "contactdb+torus_large",
    "contactdb+torus_small",
    "contactdb+utah_teapot",
    "contactdb+wine_glass",
    "contactdb+wristwatch",
    "ycb+cracker_box",
    "ycb+power_drill",
    "ycb+sugar_box",
]


def safe_name(object_name: str) -> str:
    return object_name.replace("+", "_")


def merge(batch_size: int, rows: list[dict]) -> None:
    all_results = []
    merged_rows = []
    joint_order = None
    for object_name in OBJECTS:
        object_path = OUT_DIR / f"{safe_name(object_name)}.pt"
        if not object_path.exists():
            continue
        result_list = torch.load(object_path, map_location="cpu")
        if len(result_list) != 1:
            raise ValueError(f"expected one row in {object_path}, got {len(result_list)}")
        result = result_list[0]
        all_results.append(result)
        q = result["q_dro_order"]
        joint_order = result["joint_order"]
        merged_rows.append({
            "object_name": object_name,
            "q_shape": list(q.shape),
            "q_min": float(q.min()),
            "q_max": float(q.max()),
            "q_finite": bool(torch.isfinite(q).all()),
        })
    if len(all_results) == len(OBJECTS):
        torch.save(all_results, OUT)
    payload = {
        "out": str(OUT),
        "robot_name": "svh",
        "batch_size": batch_size,
        "num_objects": len(all_results),
        "objects": OBJECTS,
        "complete": len(all_results) == len(OBJECTS),
        "joint_order": joint_order,
        "rows": merged_rows,
        "run_rows": rows,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    start_all = time.time()
    for object_name in OBJECTS:
        object_path = OUT_DIR / f"{safe_name(object_name)}.pt"
        object_summary = OUT_DIR / f"{safe_name(object_name)}_summary.json"
        if object_path.exists():
            rows.append({"object_name": object_name, "out": str(object_path), "returncode": 0, "elapsed_sec": 0.0, "skipped_existing": True})
            print(f"skipping {object_name}; existing {object_path}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--objects",
            object_name,
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--out",
            str(object_path),
            "--summary",
            str(object_summary),
        ]
        print(f"running {object_name}", flush=True)
        start = time.time()
        ret = subprocess.run(cmd, cwd=str(NEW_ROOT), text=True)
        elapsed = time.time() - start
        rows.append({"object_name": object_name, "out": str(object_path), "returncode": ret.returncode, "elapsed_sec": elapsed, "skipped_existing": False})
        merge(args.batch_size, rows)
        if ret.returncode != 0:
            break
    merge(args.batch_size, rows)
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    payload["total_wall_sec"] = time.time() - start_all
    payload["all_returncode_zero"] = all(row["returncode"] == 0 for row in rows) and payload["complete"]
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
