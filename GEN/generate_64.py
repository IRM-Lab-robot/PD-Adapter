#!/usr/bin/env python3
"""Append 44 new GenDexGrasp sources to each existing test10 set of 20."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DRO import generate_64 as audit  # same object-order, tensor and hash checks

OUT = Path(__file__).resolve().parent
GENDEX = ROOT.parent / "GenDexGrasp"
OBJECTS, HANDS, DIMS = audit.OBJECTS, audit.HANDS, audit.DIMS
OLD = {
    "inspire": ROOT / "eval/gendex_inspire_mainset_grasps.pt",
    "ability": ROOT / "results/pd_refiner_ability/gendex_test10/ability_gendex_test10_grasps.pt",
    "svh": ROOT / "results/pd_refiner_svh/gendex_test10/svh_gendex_test10_grasps.pt",
}


def script(hand: str) -> Path:
    return GENDEX / f"run_{hand}_mainset.py"


def inspect(hand: str):
    old = audit.source_rows(OLD[hand], hand, 20)
    weights = GENDEX / "ckpts/SqrtUnseenShadowhand/weights/pointnet_cvae_model.pth"
    if not script(hand).is_file() or not weights.is_file():
        raise FileNotFoundError(f"GenDex generator/model missing for {hand}")
    for obj in OBJECTS:
        typ, name = obj.split("+", 1)
        mesh = GENDEX / "data/object" / typ / name / f"{name}.stl"
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
    print(f"GEN/{hand}: original {len(old)} objects × 20 valid; model and meshes available", flush=True)
    return old


def part_location(hand: str, obj: str, part: int, count: int, seed: int, chunk: int) -> Path:
    if chunk == 44:
        # Preserve any already generated files from the original 44-at-once script.
        return OUT / "_parts" / hand / f"{audit.safe(obj)}_n44_seed{seed}.pt"
    return OUT / "_parts" / hand / f"{audit.safe(obj)}_part{part}_n{count}_seed{seed}.pt"


def object_seed(hand: str, obj: str, base_seed: int) -> int:
    return base_seed + 10000 * HANDS.index(hand) + 100 * OBJECTS.index(obj)


def parts_for(hand: str, obj: str, chunk: int, base_seed: int):
    remaining = 44
    part = 0
    while remaining:
        count = min(chunk, remaining)
        seed = object_seed(hand, obj, base_seed) + part
        yield part_location(hand, obj, part, count, seed, chunk), count, seed, part
        remaining -= count
        part += 1


def generate(hand: str, chunk: int, base_seed: int, objects=OBJECTS):
    inspect(hand)
    weights = GENDEX / "ckpts/SqrtUnseenShadowhand/weights/pointnet_cvae_model.pth"
    for obj in objects:
        for path, count, seed, part in parts_for(hand, obj, chunk, base_seed):
            meta_path = path.with_suffix(".json")
            expected = {"generator": "GenDexGrasp", "hand": hand, "object": obj,
                        "new_grasps": count, "seed": seed, "old_source_sha256": audit.sha(OLD[hand]),
                        "cvae_model_sha256": audit.sha(weights), "generator_script_sha256": audit.sha(script(hand)),
                        "num_particles": 32, "max_iter": 100,
                        "learning_rate": .005, "init_rand_scale": .5, "energy_func": "align_dist"}
            if path.exists():
                audit.source_rows(path, hand, count, (obj,))
                meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
                if any(meta.get(key) != val for key, val in expected.items()) or meta.get("part_sha256") != audit.sha(path):
                    raise ValueError(f"Cannot safely reuse unverified part {path}")
                print(f"reuse GEN/{hand} {obj} part{part}: {count} grasps, seed={seed}", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script(hand)), "--objects", obj,
                   "--num-per-object", str(count), "--num-particles", "32", "--max-iter", "100",
                   "--learning-rate", ".005", "--init-rand-scale", ".5",
                   "--energy-func", "align_dist", "--seed", str(seed), "--out", str(path)]
            with path.with_suffix(".log").open("w") as log:
                process = subprocess.run(cmd, cwd=GENDEX, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            if process.returncode or not path.is_file():
                raise RuntimeError(f"GenDex generation failed: {path.with_suffix('.log')}")
            audit.source_rows(path, hand, count, (obj,))
            meta_path.write_text(json.dumps({**expected, "part_sha256": audit.sha(path),
                                             "command": cmd}, indent=2))
            print(f"generated GEN/{hand} {obj} part{part}: {count} extra grasps, seed={seed}", flush=True)


def merge(hand: str, chunk: int, base_seed: int):
    import torch

    old = inspect(hand)
    weights = GENDEX / "ckpts/SqrtUnseenShadowhand/weights/pointnet_cvae_model.pth"
    merged, provenance = [], []
    for row in old:
        obj = row["object_name"]
        extra_parts, part_meta = [], []
        for path, count, seed, part in parts_for(hand, obj, chunk, base_seed):
            meta_path = path.with_suffix(".json")
            if not meta_path.is_file():
                raise FileNotFoundError(f"Generate missing part first: {path}")
            meta = json.loads(meta_path.read_text())
            if (meta["old_source_sha256"] != audit.sha(OLD[hand]) or
                    meta["cvae_model_sha256"] != audit.sha(weights) or
                    meta["generator_script_sha256"] != audit.sha(script(hand)) or
                    meta["generator"] != "GenDexGrasp" or meta["hand"] != hand or meta["object"] != obj or
                    meta["seed"] != seed or meta["new_grasps"] != count or
                    meta["part_sha256"] != audit.sha(path)):
                raise ValueError(f"Part provenance changed: {path}")
            new_row = audit.source_rows(path, hand, count, (obj,))[0]
            extra_parts.append(torch.as_tensor(new_row["q_dro_order"]).clone())
            part_meta.append({"part": part, "file": str(path), "count": count,
                              "seed": seed, "sha256": audit.sha(path)})
        original = torch.as_tensor(row["q_dro_order"]).clone()
        q = torch.cat([original, *extra_parts], dim=0)
        if q.shape != (64, DIMS[hand]) or not torch.equal(original, q[:20]) or torch.unique(q, dim=0).shape[0] != 64:
            raise ValueError(f"Incomplete or duplicate merged grasps: {hand}/{obj}")
        merged.append({**row, "q_dro_order": q})
        provenance.append({"object_name": obj, "old_indices": [0, 19],
                           "new_indices": [20, 63], "parts": part_meta})
    dest = OUT / f"{hand}.pt"
    manifest = dest.with_suffix(".json")
    config = {"generator": "GenDexGrasp", "hand": hand, "objects": list(OBJECTS),
              "sources_per_object": 64, "old_per_object": 20, "added_per_object": 44,
              "old_source": str(OLD[hand]), "old_source_sha256": audit.sha(OLD[hand]),
              "cvae_model_sha256": audit.sha(weights),
              "generator_script_sha256": audit.sha(script(hand)), "base_seed": base_seed,
              "chunk_size": chunk,
              "generation_params": {"num_particles": 32, "max_iter": 100,
                                    "learning_rate": .005, "init_rand_scale": .5,
                                    "energy_func": "align_dist"},
              "per_object": provenance}
    if dest.exists():
        existing = audit.source_rows(dest, hand, 64)
        old_manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if (not all(torch.equal(torch.as_tensor(a["q_dro_order"]), torch.as_tensor(b["q_dro_order"]))
                    for a, b in zip(existing, merged)) or
                any(old_manifest.get(k) != v for k, v in config.items()) or
                old_manifest.get("output_sha256") != audit.sha(dest)):
            raise ValueError(f"Refusing to overwrite mismatched completed source file: {dest}")
        print(f"verified existing {dest}: 10 × 64", flush=True)
        return
    torch.save(merged, dest)
    manifest.write_text(json.dumps({**config, "output_sha256": audit.sha(dest)},
                                   indent=2, ensure_ascii=False))
    print(f"saved {dest}: 10 objects × 64 sources, first 20/obj identical to original", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("check", "generate", "merge", "all"))
    ap.add_argument("--hand", choices=(*HANDS, "all"), default="all")
    ap.add_argument("--seed-base", type=int, default=20267500)
    ap.add_argument("--chunk-size", type=int, default=11,
                    help="Number of new grasps saved per .pt; default 11 gives 4 resumable parts/object")
    ap.add_argument("--objects", nargs="+", choices=OBJECTS,
                    help="Optional subset for the generate stage; useful for a one-object pilot")
    args = ap.parse_args()
    if not 1 <= args.chunk_size <= 44:
        ap.error("--chunk-size must be between 1 and 44")
    if args.objects and (args.stage != "generate" or len(args.objects) != len(set(args.objects))):
        ap.error("--objects is only for generate and must not contain duplicates")
    for hand in HANDS if args.hand == "all" else (args.hand,):
        if args.stage == "check":
            inspect(hand)
        else:
            if args.stage in ("generate", "all"):
                generate(hand, args.chunk_size, args.seed_base, args.objects or OBJECTS)
            if args.stage in ("merge", "all"):
                merge(hand, args.chunk_size, args.seed_base)


if __name__ == "__main__":
    main()
