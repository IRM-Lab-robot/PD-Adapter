#!/usr/bin/env python3
"""Append 44 new D(R,O) Grasp sources to each existing test10 set of 20."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DRO = ROOT.parent / "DRO-Grasp"
OBJECTS = tuple(json.loads((ROOT / "eval/inspire_train_test_split.json").read_text())["test"][:10])
HANDS = ("inspire", "ability", "svh")
DIMS = {"inspire": 18, "ability": 16, "svh": 26}
OLD = {
    "inspire": ROOT / "eval/original_test10_zero_shot_b20.pt",
    "ability": ROOT / "results/pd_refiner_ability/ability_test10_zero_shot_b20.pt",
    "svh": ROOT / "results/pd_refiner_svh/svh_test10_zero_shot_b20.pt",
}
SCRIPTS = {
    "inspire": ROOT / "eval/dro_zero_shot_inspire_objects.py",
    "ability": ROOT / "results/pd_refiner_ability/generate_ability_dro_zero_shot.py",
    "svh": ROOT / "results/pd_refiner_svh/generate_svh_dro_zero_shot.py",
}


@functools.lru_cache(maxsize=None)
def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(piece)
    return hasher.hexdigest()


def safe(obj: str) -> str:
    return obj.replace("+", "_")


def source_rows(path: Path, hand: str, expected_n: int, objects=OBJECTS):
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or len(rows) != len(objects) or {r["object_name"] for r in rows} != set(objects):
        raise ValueError(f"Unexpected object set or duplicate objects in {path}")
    # Historical Ability/SVH DRO files use a different object ordering. Preserve
    # each object's 20 grasp indices while writing every merged file in test10 order.
    by_name = {r["object_name"]: r for r in rows}
    rows = [by_name[obj] for obj in objects]
    for row in rows:
        q = torch.as_tensor(row["q_dro_order"])
        if q.shape != (expected_n, DIMS[hand]) or not bool(torch.isfinite(q).all()):
            raise ValueError(f"Bad grasp tensor for {hand}/{row['object_name']} in {path}: {tuple(q.shape)}")
        if torch.unique(q, dim=0).shape[0] != expected_n:
            raise ValueError(f"Exact duplicate source grasps in {path} for {row['object_name']}")
        if "robot_name" in row and row["robot_name"] != hand:
            raise ValueError(f"Wrong hand in {path}: {row['robot_name']}")
    return rows


def inspect(hand: str):
    old = source_rows(OLD[hand], hand, 20)
    if not SCRIPTS[hand].is_file() or not (DRO / "ckpt/model/model_3robots.pth").is_file():
        raise FileNotFoundError("DRO generator or model_3robots.pth is missing")
    for obj in OBJECTS:
        dataset, name = obj.split("+", 1)
        pointcloud = DRO / "data/PointCloud/object" / dataset / f"{name}.pt"
        if not pointcloud.is_file():
            raise FileNotFoundError(pointcloud)
    print(f"DRO/{hand}: original {len(old)} objects × 20 valid, model and point clouds available", flush=True)
    return old


def part_location(hand: str, obj: str, part: int, count: int, seed: int) -> Path:
    return OUT / "_parts" / hand / f"{safe(obj)}_part{part}_n{count}_seed{seed}.pt"


def parts_for(hand: str, obj: str, chunk: int, base_seed: int):
    obj_idx = OBJECTS.index(obj)
    remaining = 44
    part = 0
    while remaining:
        n = min(remaining, chunk)
        seed = base_seed + 10000 * HANDS.index(hand) + 100 * obj_idx + part
        yield part_location(hand, obj, part, n, seed), n, seed
        remaining -= n
        part += 1


def generate(hand: str, chunk: int, base_seed: int, objects=OBJECTS):
    inspect(hand)
    model = DRO / "ckpt/model/model_3robots.pth"
    for obj in objects:
        for path, count, seed in parts_for(hand, obj, chunk, base_seed):
            meta_path = path.with_suffix(".json")
            expected = {"generator": "DRO", "hand": hand, "object": obj,
                        "new_grasps": count, "seed": seed, "old_source_sha256": sha(OLD[hand]),
                        "model_sha256": sha(model), "generator_script_sha256": sha(SCRIPTS[hand])}
            if path.exists():
                rows = source_rows(path, hand, count, (obj,))
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                if any(meta.get(key) != value for key, value in expected.items()) or meta.get("part_sha256") != sha(path):
                    raise ValueError(f"Cannot safely reuse unverified part {path}")
                print(f"reuse DRO/{hand} {obj}: {count} grasps, seed={seed}", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(SCRIPTS[hand]), "--objects", obj,
                   "--batch-size", str(count), "--seed", str(seed), "--out", str(path),
                   "--model", str(model)]
            if hand in ("ability", "svh"):
                cmd.extend(("--summary", str(path.with_name(path.stem + "_generator_summary.json"))))
            with path.with_suffix(".log").open("w") as log:
                status = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            if status.returncode or not path.is_file():
                raise RuntimeError(f"DRO generation failed: {path.with_suffix('.log')}")
            source_rows(path, hand, count, (obj,))
            meta_path.write_text(json.dumps({**expected, "part_sha256": sha(path), "command": cmd}, indent=2))
            print(f"generated DRO/{hand} {obj}: {count} extra grasps, seed={seed}", flush=True)


def merge(hand: str, chunk: int, base_seed: int):
    import torch

    old = inspect(hand)
    model = DRO / "ckpt/model/model_3robots.pth"
    merged, provenance = [], []
    for row in old:
        obj = row["object_name"]
        extra = []
        part_meta = []
        for path, count, seed in parts_for(hand, obj, chunk, base_seed):
            meta_path = path.with_suffix(".json")
            if not meta_path.is_file():
                raise FileNotFoundError(f"Generate missing part first: {path}")
            meta = json.loads(meta_path.read_text())
            if (meta["old_source_sha256"] != sha(OLD[hand]) or meta["model_sha256"] != sha(model)
                    or meta["generator_script_sha256"] != sha(SCRIPTS[hand])
                    or meta["generator"] != "DRO" or meta["hand"] != hand or meta["object"] != obj
                    or meta["seed"] != seed or meta["new_grasps"] != count or meta["part_sha256"] != sha(path)):
                raise ValueError(f"Part provenance changed: {path}")
            new_row = source_rows(path, hand, count, (obj,))[0]
            if new_row.get("joint_order") != row.get("joint_order"):
                raise ValueError(f"Different joint order for {hand}/{obj}: {path}")
            extra.append(torch.as_tensor(new_row["q_dro_order"]).clone())
            part_meta.append({"file": str(path), "count": count, "seed": seed, "sha256": sha(path)})
        q_old = torch.as_tensor(row["q_dro_order"]).clone()
        q = torch.cat([q_old, *extra], dim=0)
        if q.shape != (64, DIMS[hand]) or not torch.equal(q[:20], q_old) or torch.unique(q, dim=0).shape[0] != 64:
            raise ValueError(f"Incomplete or duplicate merged grasps: {hand}/{obj}")
        merged.append({**row, "q_dro_order": q})
        provenance.append({"object_name": obj, "old_indices": [0, 19], "new_indices": [20, 63],
                           "parts": part_meta})
    dest = OUT / f"{hand}.pt"
    manifest = dest.with_suffix(".json")
    config = {"generator": "DRO", "hand": hand, "objects": list(OBJECTS),
              "sources_per_object": 64, "old_per_object": 20, "added_per_object": 44,
              "old_source": str(OLD[hand]), "old_source_sha256": sha(OLD[hand]),
              "model_sha256": sha(model), "generator_script_sha256": sha(SCRIPTS[hand]),
              "base_seed": base_seed, "chunk_size": chunk,
              "per_object": provenance}
    if dest.exists():
        existing = source_rows(dest, hand, 64)
        existing_meta = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if (not all(torch.equal(torch.as_tensor(a["q_dro_order"]), torch.as_tensor(b["q_dro_order"]))
                    for a, b in zip(existing, merged)) or
                any(existing_meta.get(k) != v for k, v in config.items()) or
                existing_meta.get("output_sha256") != sha(dest)):
            raise ValueError(f"Refusing to overwrite mismatched completed source file: {dest}")
        print(f"verified existing {dest}: 10 × 64", flush=True)
        return
    torch.save(merged, dest)
    manifest.write_text(json.dumps({**config, "output_sha256": sha(dest)}, indent=2, ensure_ascii=False))
    print(f"saved {dest}: 10 objects × 64 sources, first 20/obj identical to original", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("check", "generate", "merge", "all"))
    ap.add_argument("--hand", choices=(*HANDS, "all"), default="all")
    ap.add_argument("--chunk-size", type=int, default=11,
                    help="Number of new grasps saved per .pt; default 11 gives 4 resumable parts/object; 22 or 44 trades resilience for speed")
    ap.add_argument("--seed-base", type=int, default=20266400)
    ap.add_argument("--objects", nargs="+", choices=OBJECTS,
                    help="Optional subset for the generate stage; useful for a one-object A5000 pilot")
    a = ap.parse_args()
    if not 1 <= a.chunk_size <= 44:
        ap.error("--chunk-size must be between 1 and 44")
    if a.objects and (a.stage != "generate" or len(a.objects) != len(set(a.objects))):
        ap.error("--objects is only for generate and must not contain duplicates")
    for hand in HANDS if a.hand == "all" else (a.hand,):
        if a.stage == "check":
            inspect(hand)
        else:
            if a.stage in ("generate", "all"):
                generate(hand, a.chunk_size, a.seed_base, a.objects or OBJECTS)
            if a.stage in ("merge", "all"):
                merge(hand, a.chunk_size, a.seed_base)


if __name__ == "__main__":
    main()
