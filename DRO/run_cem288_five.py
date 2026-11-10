#!/usr/bin/env python3
"""Five hand–generator CEM-288 lines: 64 sources/object, one CPU recheck.

Search 72×4 per source and freeze the GPU-ranked top-1 before fresh physical
labels. Both top-1 and any-candidate recovery use the same full-pool recheck.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DRO import generate_64 as source_audit
from DRO import run_svh_cem288 as metrics

OUT = ROOT / "results/supplement_cem288_five_b64"
OBJECTS = metrics.OBJECTS
BATTERY = metrics.BATTERY
POPULATION = 72
GENERATIONS = 4
PART_SIZE = 8
SEARCH_BATCH = 4
KEEP = 3
PART_CAP = PART_SIZE * KEEP
VERIFY_BATCH = 32
SEED = 10001
WRIST_XYZ_STD = .01
WRIST_ROT_STD = .10
U_STD = .25
SPATIAL_SAMPLING = "local"
LINES = {
    "inspire-dro": ("inspire", "DRO", ROOT / "eval/original_test10_zero_shot_b20.pt"),
    "inspire-gen": ("inspire", "GEN", ROOT / "eval/gendex_inspire_mainset_grasps.pt"),
    "svh-gen": ("svh", "GEN", ROOT / "results/pd_refiner_svh/gendex_test10/svh_gendex_test10_grasps.pt"),
    "ability-dro": ("ability", "DRO", ROOT / "results/pd_refiner_ability/ability_test10_zero_shot_b20.pt"),
    "ability-gen": ("ability", "GEN", ROOT / "results/pd_refiner_ability/gendex_test10/ability_gendex_test10_grasps.pt"),
}
DIMS = source_audit.DIMS
PARAM_DIMS = {"inspire": 12, "ability": 12, "svh": 15}


def source_file(line: str) -> Path:
    hand, gen, _ = LINES[line]
    return ROOT / gen / f"{hand}.pt"


def manifest_file(line: str) -> Path:
    return source_file(line).with_suffix(".json")


def search_file(line: str, obj: str, part: int) -> Path:
    ids = source_ids(part)
    return OUT / line / "search" / metrics.safe(obj) / f"part{part:02d}_s{ids[0]:02d}_e{ids[-1]:02d}.pt"


def recheck_file(line: str, obj: str) -> Path:
    return OUT / line / "revalidation_once" / f"{metrics.safe(obj)}.json"


def source_ids(part: int) -> list[int]:
    if part not in range(64 // PART_SIZE):
        raise ValueError(part)
    return list(range(part * PART_SIZE, (part + 1) * PART_SIZE))


def check_source(line: str) -> str:
    import torch

    hand, generator, original = LINES[line]
    current, manifest_path = source_file(line), manifest_file(line)
    if not current.is_file() or not manifest_path.is_file() or not original.is_file():
        raise FileNotFoundError(f"Missing merged source, manifest, or original20 for {line}")
    info = json.loads(manifest_path.read_text())
    expected_gen = "DRO" if generator == "DRO" else "GenDexGrasp"
    if (info.get("generator") != expected_gen or info.get("hand") != hand
            or info.get("objects") != list(OBJECTS) or info.get("sources_per_object") != 64
            or info.get("old_per_object") != 20 or info.get("added_per_object") != 44
            or info.get("old_source_sha256") != metrics.sha(original)
            or info.get("output_sha256") != metrics.sha(current)):
        raise ValueError(f"Source lineage/sha256 mismatch: {line}")
    merged = source_audit.source_rows(current, hand, 64)
    old = source_audit.source_rows(original, hand, 20)
    for combined, before in zip(merged, old):
        if not torch.equal(torch.as_tensor(combined["q_dro_order"])[:20], torch.as_tensor(before["q_dro_order"])):
            raise ValueError(f"Original20 altered for {line}/{combined['object_name']}")
    print(f"verified {line}: test10 × 64, original20 unchanged, {DIMS[hand]}D", flush=True)
    return info["output_sha256"]


def driver(line: str) -> Path:
    hand = LINES[line][0]
    return ROOT / "eval" / f"generate_{hand}_train_bsearch.py"


def check_pool(line: str, obj: str, part: int, source_sha: str):
    import torch

    hand = LINES[line][0]
    path = search_file(line, obj, part)
    if not path.is_file():
        raise FileNotFoundError(f"Run search first: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"zero_file": str(source_file(line)), "selection_battery": BATTERY,
                "max_grasps": 64, "population": POPULATION, "generations": GENERATIONS,
                "elites": 8, "restarts": 1, "batch_source_grasps": SEARCH_BATCH,
                "keep_per_grasp": KEEP, "keep_per_object": PART_CAP,
                "cpu_final_candidates": PART_CAP, "search_pipeline": "gpu",
                "reuse_gpu_sim": True, "log_rejected": True,
                "seed": SEED + 1009 * part, "source_indices": source_ids(part),
                "warm_start_file": None, "init_std_scale": 1.0,
                 "wrist_xyz_std": WRIST_XYZ_STD, "wrist_rot_std": WRIST_ROT_STD, "u_std": U_STD}
    for key, value in expected.items():
        if data["args"].get(key) != value:
            raise ValueError(f"{path}: wrong search {key}: {data['args'].get(key)!r}")
    if data["args"].get("spatial_sampling", "local") != SPATIAL_SAMPLING:
        raise ValueError(f"{path}: wrong spatial sampling mode")
    if hand == "inspire" and data["args"].get("allow_non_train") is not True:
        raise ValueError(f"Missing Inspire test-object permission: {path}")
    if (data.get("objects") != [obj] or data.get("total_source_grasps") != PART_SIZE
            or len(data.get("rows", [])) != 1 or
            ("robot_name" in data and data["robot_name"] != hand)):
        raise ValueError(f"Wrong hand/object/source count: {path}")
    row = data["rows"][0]
    if ([int(grasp["grasp_index"]) for grasp in row["grasp_rows"]] != source_ids(part)
            or torch.as_tensor(row["q_seed"]).shape != (PART_SIZE, DIMS[hand])):
        raise ValueError(f"Source indices/configuration are incomplete: {path}")
    count = 0
    for key, expected_label in (("passing_candidates", True), ("cpu_final_rejected_candidates", False)):
        if key not in row:
            raise ValueError(f"Missing CPU-final failures/successes: {path}")
        for candidate in row[key]:
            if (int(candidate["source_grasp_index"]) not in source_ids(part)
                    or bool(candidate["cpu_success"]) != expected_label
                    or "gpu_rank_score" not in candidate
                    or not math.isfinite(float(candidate["gpu_rank_score"]))
                    or torch.as_tensor(candidate["q"]).numel() != DIMS[hand]
                    or torch.as_tensor(candidate["params"]).numel() != PARAM_DIMS[hand]):
                raise ValueError(f"Invalid candidate state/GPU score: {path}")
            count += 1
    if count != row["cpu_final_evaluated"] or count > PART_CAP:
        raise ValueError(f"Search CPU-final pool incomplete/truncated: {path}")
    if source_sha != metrics.sha(source_file(line)):
        raise ValueError(f"Input source changed while searching {line}")
    return row


def search(line: str, objects, chosen_part=None):
    source_sha = check_source(line)
    script = driver(line)
    if not script.is_file():
        raise FileNotFoundError(script)
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    hand = LINES[line][0]
    for obj in objects:
        for part in ((chosen_part,) if chosen_part is not None else range(64 // PART_SIZE)):
            dest = search_file(line, obj, part)
            if dest.exists():
                check_pool(line, obj, part, source_sha)
                print(f"reuse {line} {obj} part{part}", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--zero-file", str(source_file(line)),
                   "--objects", obj, "--max-grasps", "64", "--selection-battery", BATTERY,
                   "--population", str(POPULATION), "--generations", str(GENERATIONS),
                   "--elites", "8", "--restarts", "1", "--batch-source-grasps", str(SEARCH_BATCH),
                   "--keep-per-grasp", str(KEEP), "--keep-per-object", str(PART_CAP),
                   "--cpu-final-candidates", str(PART_CAP), "--search-pipeline", "gpu",
                   "--reuse-gpu-sim", "--log-rejected", "--seed", str(SEED + 1009 * part),
                   "--source-indices", *(str(i) for i in source_ids(part)), "--out", str(dest)]
            if hand == "inspire":
                cmd.extend(("--allow-non-train", "--arm-id", "cem288"))
            log = dest.with_suffix(".log")
            with log.open("w") as stream:
                process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode or not dest.is_file():
                raise RuntimeError(f"Search failed for {line}/{obj}/part{part}; inspect {log}")
            check_pool(line, obj, part, source_sha)
            print(f"searched {line} {obj} part{part} ({PART_SIZE} sources)", flush=True)


def recheck(line: str, objects):
    # In a fresh process the Isaac/validator module MUST load before torch.
    sys.path.insert(0, str(ROOT / "eval"))
    import importlib
    validator = importlib.import_module(f"generate_{LINES[line][0]}_train_bsearch")
    import torch

    source_sha = check_source(line)
    hand = LINES[line][0]
    for obj in objects:
        candidates = []
        for part in range(64 // PART_SIZE):
            row = check_pool(line, obj, part, source_sha)
            for kind in ("passing_candidates", "cpu_final_rejected_candidates"):
                for i, cand in enumerate(row[kind]):
                    params = torch.as_tensor(cand["params"], dtype=torch.float32).contiguous()
                    candidates.append({"id": f"part{part:02d}/{kind}/{i}",
                                       "source_index": int(cand["source_grasp_index"]),
                                       "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(DIMS[hand]),
                                       "gpu_rank_score": float(cand["gpu_rank_score"]),
                                       "search_params_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
                                       "search_cpu_success": bool(cand["cpu_success"])})
        selected = metrics.freeze_top1(candidates)
        selected_ids = [candidates[i]["id"] if i is not None else None for i in selected]
        pool_shas = [metrics.sha(search_file(line, obj, part)) for part in range(64 // PART_SIZE)]
        dest = recheck_file(line, obj)
        if dest.exists():
            record = json.loads(dest.read_text())
            if (record.get("source_sha256") != source_sha or record.get("search_part_sha256") != pool_shas
                    or record.get("candidate_ids") != [c["id"] for c in candidates]
                    or record.get("top1_candidate_ids") != selected_ids
                    or record.get("battery") != BATTERY or record.get("verify_batch_size") != VERIFY_BATCH
                    or record.get("candidate_count") != len(candidates)
                    or len(record.get("candidate_success", [])) != len(candidates)):
                raise ValueError(f"Refusing to reuse stale recheck: {dest}")
            top1, multi = metrics.outcomes(candidates, record["candidate_success"], selected)
            if record["top1_success"] != top1 or record["multi_success"] != multi:
                raise ValueError(f"Candidate success keys differ from saved results: {dest}")
            print(f"reuse one-time {line} recheck {obj}", flush=True)
            continue
        distances, labels = [None] * len(candidates), [False] * len(candidates)
        batches = []
        for start in range(0, len(candidates), VERIFY_BATCH):
            block = candidates[start:start + VERIFY_BATCH]
            q = torch.stack([c["q"] for c in block])
            if len(block) < VERIFY_BATCH:
                q = torch.cat([q, q[-1:].expand(VERIFY_BATCH - len(block), -1)], dim=0)
            d, _ = validator.score_fresh(obj, q, BATTERY, use_gpu_pipeline=False)
            batches.append({"real_candidates": len(block), "simulated_envs": int(q.shape[0])})
            for offset in range(len(block)):
                value = float(d[offset])
                distances[start + offset] = value if math.isfinite(value) else None
                labels[start + offset] = math.isfinite(value) and value <= metrics.THRESHOLD
        top1, multi = metrics.outcomes(candidates, labels, selected)
        result = {"line": line, "hand": hand, "generator": LINES[line][1],
                  "object_name": obj, "source_sha256": source_sha,
                  "search_part_sha256": pool_shas, "gpu_evaluations_per_source": 288,
                  "battery": BATTERY, "threshold_m": metrics.THRESHOLD,
                  "verify_batch_size": VERIFY_BATCH, "tail_padded": True,
                  "shuffle": False, "revalidation_count": 1,
                  "candidate_count": len(candidates), "cpu_batches": batches,
                  "selection_rule": "minimum search-time gpu_rank_score within CPU-final-evaluated pool; search-parameter hash breaks ties",
                  "candidate_ids": [c["id"] for c in candidates],
                  "source_indices": [c["source_index"] for c in candidates],
                  "candidate_gpu_rank_scores": [c["gpu_rank_score"] for c in candidates],
                  "candidate_search_params_sha256": [c["search_params_sha256"] for c in candidates],
                  "search_cpu_success": [c["search_cpu_success"] for c in candidates],
                  "distance_m": distances, "candidate_success": labels,
                  "top1_candidate_indices": selected, "top1_candidate_ids": selected_ids,
                  "top1_success": top1, "multi_success": multi,
                  "top1_old20_success": sum(top1[:20]), "top1_new44_success": sum(top1[20:]),
                  "multi_old20_success": sum(multi[:20]), "multi_new44_success": sum(multi[20:])}
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        tmp.replace(dest)
        print(f"{line} {obj}: top-1 {sum(top1)}/64, multi {sum(multi)}/64 (one recheck)", flush=True)


def summarize(line: str):
    source_sha = check_source(line)
    counts = {name: [0, 0] for name in ("top1", "multi")}
    per_object, rates = {}, {name: [] for name in counts}
    for obj in OBJECTS:
        path = recheck_file(line, obj)
        if not path.is_file():
            raise FileNotFoundError(f"Run all ten one-time rechecks first: {path}")
        row = json.loads(path.read_text())
        if (row.get("line") != line or row.get("object_name") != obj or row.get("source_sha256") != source_sha
                or row.get("search_part_sha256") != [metrics.sha(search_file(line, obj, p)) for p in range(8)]
                or row.get("battery") != BATTERY or row.get("verify_batch_size") != VERIFY_BATCH
                or row.get("tail_padded") is not True or row.get("revalidation_count") != 1):
            raise ValueError(f"Mismatched frozen validation protocol: {path}")
        n = row["candidate_count"]
        if any(len(row.get(key, [])) != n for key in ("candidate_ids", "candidate_success",
               "source_indices", "candidate_gpu_rank_scores", "candidate_search_params_sha256")):
            raise ValueError(f"Incomplete candidate labels: {path}")
        candidates = [{"id": cid, "source_index": sid, "gpu_rank_score": float(score),
                       "search_params_sha256": phash}
                      for cid, sid, score, phash in zip(row["candidate_ids"], row["source_indices"],
                                                        row["candidate_gpu_rank_scores"],
                                                        row["candidate_search_params_sha256"])]
        selected = metrics.freeze_top1(candidates)
        if (row.get("top1_candidate_indices") != selected or row.get("top1_candidate_ids") != [
                candidates[i]["id"] if i is not None else None for i in selected]):
            raise ValueError(f"Top-1 differs from saved GPU scores: {path}")
        top1, multi = metrics.outcomes(candidates, row["candidate_success"], selected)
        if row.get("top1_success") != top1 or row.get("multi_success") != multi:
            raise ValueError(f"Saved success values differ from candidate labels: {path}")
        per_object[obj] = {"candidate_count": n,
                           "targets_with_gpu_ranked_candidate": sum(i is not None for i in selected)}
        for name, labels in (("top1", top1), ("multi", multi)):
            a, b = sum(labels[:20]), sum(labels[20:])
            if row.get(f"{name}_old20_success") != a or row.get(f"{name}_new44_success") != b:
                raise ValueError(f"Old/new grasp index accounting mismatch: {path}")
            counts[name][0] += a
            counts[name][1] += b
            rates[name].append((a + b) / 64 * 100)
            per_object[obj][name] = {"old20_success": a, "new44_success": b,
                                     "overall64_success": a + b, "rate_pct": (a + b) / 64 * 100,
                                     "source_success": labels}
    rng = random.Random(20261221)
    draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    results = {}
    for name in ("top1", "multi"):
        a, b = counts[name]
        boot = [sum(rates[name][i] for i in draw) / 10 for draw in draws]
        results[name] = {"old20": {"success": a, "total": 200, "rate_pct": a / 2},
                         "new44": {"success": b, "total": 440, "rate_pct": b / 4.4},
                         "overall": {"success": a + b, "total": 640, "rate_pct": (a + b) / 6.4,
                                     "object_cluster_95ci_pct": [metrics.quantile(boot, .025),
                                                                  metrics.quantile(boot, .975)]}}
    gap = [m - t for t, m in zip(rates["top1"], rates["multi"])]
    gap_boot = [sum(gap[i] for i in draw) / 10 for draw in draws]
    report = {"line": line, "method": "CEM-288", "source_sha256": source_sha,
              "battery": BATTERY, "search_samples_per_source": 288,
              "revalidation_count": 1, "verify_batch_size": VERIFY_BATCH, "tail_padded": True,
              "top1": results["top1"], "multi": results["multi"],
              "multi_minus_top1_pp": results["multi"]["overall"]["rate_pct"] - results["top1"]["overall"]["rate_pct"],
              "multi_minus_top1_object_cluster_95ci_pp": [metrics.quantile(gap_boot, .025),
                                                            metrics.quantile(gap_boot, .975)],
              "per_object": per_object}
    destination = OUT / line / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    for name in ("top1", "multi"):
        entry = results[name]
        print(f"{line} CEM-288 {name}: old {entry['old20']['success']}/200 = {entry['old20']['rate_pct']:.2f}% | "
              f"new {entry['new44']['success']}/440 = {entry['new44']['rate_pct']:.2f}% | "
              f"overall {entry['overall']['success']}/640 = {entry['overall']['rate_pct']:.2f}%")
    print(f"saved {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "search", "recheck", "summarize"))
    parser.add_argument("--line", choices=(*LINES, "all"), default="all")
    parser.add_argument("--object", choices=OBJECTS, help="One-object pilot; search/recheck only")
    parser.add_argument("--part", type=int, choices=range(64 // PART_SIZE),
                        help="One eight-target search part; requires --line and --object")
    args = parser.parse_args()
    if args.object and args.stage not in ("search", "recheck"):
        parser.error("--object is for search/recheck only")
    if args.part is not None and (args.stage != "search" or args.object is None or args.line == "all"):
        parser.error("--part requires search --line <one line> --object <one object>")
    chosen = LINES if args.line == "all" else (args.line,)
    for line in chosen:
        if args.stage == "check":
            check_source(line)
        elif args.stage == "search":
            search(line, (args.object,) if args.object else OBJECTS, args.part)
        elif args.stage == "recheck":
            recheck(line, (args.object,) if args.object else OBJECTS)
        else:
            summarize(line)


if __name__ == "__main__":
    main()
