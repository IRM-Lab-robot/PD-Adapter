#!/usr/bin/env python3
"""SVH–DRO CEM-288 on test10×64: search, one recheck, two recovery rates.

Each target receives 72 samples × 4 generations. Before fresh CPU-state
revalidation, select one candidate per source by search-time GPU score.
The same full-pool recheck labels yield both top-1 and any-candidate recovery.
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
INPUT = ROOT / "DRO/svh.pt"
INPUT_MANIFEST = ROOT / "DRO/svh.json"
ORIGINAL = ROOT / "results/pd_refiner_svh/svh_test10_zero_shot_b20.pt"
OUT = ROOT / "results/supplement_svh_dro_cem288_b64"
SEARCH = OUT / "search"
CHECKS = OUT / "revalidation_once"
OBJECTS = tuple(json.loads((ROOT / "eval/inspire_train_test_split.json").read_text())["test"][:10])
BATTERY = "standard_six_force05_steps6"
THRESHOLD = .02
SEED = 10001
GPU_POPULATION = 72
GENERATIONS = 4
SEARCH_BATCH = 4  # 4×72=288 GPU envs, conservative for the A5000 24 GB
SOURCES_PER_PART = 8  # checkpoint every 8 targets, instead of losing a whole object
VERIFY_BATCH = 32
KEEP_PER_GRASP = 3
# 8 sources × 3 retained/source: the per-part cap cannot exclude a source.
PART_CAP = SOURCES_PER_PART * KEEP_PER_GRASP


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe(name: str) -> str:
    return name.replace("+", "_")


def part_path(obj: str, part: int) -> Path:
    begin = part * SOURCES_PER_PART
    end = min(begin + SOURCES_PER_PART, 64)
    return SEARCH / safe(obj) / f"part{part:02d}_s{begin:02d}_e{end-1:02d}.pt"


def part_indices(part: int) -> list[int]:
    if part not in range(64 // SOURCES_PER_PART):
        raise ValueError(part)
    return list(range(part * SOURCES_PER_PART, (part + 1) * SOURCES_PER_PART))


def pool_sha(obj: str) -> list[str]:
    return [sha(part_path(obj, part)) for part in range(64 // SOURCES_PER_PART)]


def result_path(obj: str) -> Path:
    return CHECKS / f"{safe(obj)}.json"


def check_source() -> str:
    import torch  # Offline; simulator imports Isaac Gym before torch in recheck().

    if not INPUT.is_file() or not INPUT_MANIFEST.is_file() or not ORIGINAL.is_file():
        raise FileNotFoundError("DRO/svh.pt, its JSON manifest, and the original b20 source are required")
    manifest = json.loads(INPUT_MANIFEST.read_text())
    if (manifest.get("hand") != "svh" or manifest.get("generator") != "DRO"
            or manifest.get("objects") != list(OBJECTS) or manifest.get("sources_per_object") != 64
            or manifest.get("old_per_object") != 20 or manifest.get("added_per_object") != 44
            or manifest.get("output_sha256") != sha(INPUT)
            or manifest.get("old_source_sha256") != sha(ORIGINAL)):
        raise ValueError("The merged SVH–DRO source manifest does not match this experiment")
    sys.path.insert(0, str(ROOT))
    from DRO import generate_64 as source_audit
    full = source_audit.source_rows(INPUT, "svh", 64)
    old = source_audit.source_rows(ORIGINAL, "svh", 20)
    for current, before in zip(full, old):
        if current["object_name"] != before["object_name"] or not torch.equal(
                torch.as_tensor(current["q_dro_order"])[:20], torch.as_tensor(before["q_dro_order"])):
            raise ValueError(f"Historical 20 SVH–DRO grasps changed for {current['object_name']}")
    print("verified SVH–DRO 10×64: original20 unchanged, new44 present in every object", flush=True)
    return manifest["output_sha256"]


def check_search(obj: str, part: int, source_sha: str) -> dict:
    import torch

    path = part_path(obj, part)
    if not path.is_file():
        raise FileNotFoundError(f"Run search first: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"zero_file": str(INPUT), "selection_battery": BATTERY,
                "max_grasps": 64, "population": GPU_POPULATION,
                "generations": GENERATIONS, "elites": 8, "restarts": 1,
                "batch_source_grasps": SEARCH_BATCH, "keep_per_grasp": KEEP_PER_GRASP,
                "keep_per_object": PART_CAP, "cpu_final_candidates": PART_CAP,
                "search_pipeline": "gpu", "reuse_gpu_sim": True,
                "log_rejected": True, "seed": SEED + 1009 * part, "warm_start_file": None,
                "source_indices": part_indices(part), "init_std_scale": 1.0,
                "wrist_xyz_std": .01, "wrist_rot_std": .10, "u_std": .25}
    args = data["args"]
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"{path}: {key}: expected {value!r}, got {args.get(key)!r}")
    if (data.get("objects") != [obj] or data.get("robot_name") != "svh"
            or data.get("total_source_grasps") != SOURCES_PER_PART or len(data.get("rows", [])) != 1):
        raise ValueError(f"Incomplete or mismatched SVH search pool: {path}")
    row = data["rows"][0]
    if [int(r["grasp_index"]) for r in row["grasp_rows"]] != part_indices(part):
        raise ValueError(f"Search did not cover source indices {part_indices(part)}: {path}")
    for kind, items in (("passing", row["passing_candidates"]),
                        ("rejected", row.get("cpu_final_rejected_candidates", []))):
        for cand in items:
            if (int(cand["source_grasp_index"]) not in range(64)
                    or bool(cand["cpu_success"]) != (kind == "passing")):
                raise ValueError(f"Candidate identity/CPU label mismatch: {path}")
            # CPU `score` was overwritten during search finalization and MUST
            # never be used to choose the top-1 candidate.
            if "gpu_rank_score" not in cand or not math.isfinite(float(cand["gpu_rank_score"])):
                raise ValueError(f"Missing/non-finite search-time GPU rank score: {path}")
    evaluated = len(row["passing_candidates"]) + len(row.get("cpu_final_rejected_candidates", []))
    if evaluated != int(row["cpu_final_evaluated"]) or evaluated > PART_CAP:
        raise ValueError(f"Search CPU-final candidates are incomplete or truncated: {path}")
    if source_sha != sha(INPUT):
        raise ValueError("SVH source file changed during search")
    return data


def search(objects, only_part=None):
    source_sha = check_source()
    driver = ROOT / "eval/generate_svh_train_bsearch.py"
    if not driver.is_file():
        raise FileNotFoundError(driver)
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    for obj in objects:
        for part in ((only_part,) if only_part is not None else range(64 // SOURCES_PER_PART)):
            dst = part_path(obj, part)
            if dst.exists():
                check_search(obj, part, source_sha)
                print(f"reuse CEM-288: {obj} part{part} ({SOURCES_PER_PART} targets)", flush=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(driver), "--zero-file", str(INPUT), "--objects", obj,
                   "--max-grasps", "64", "--selection-battery", BATTERY,
                   "--population", str(GPU_POPULATION), "--generations", str(GENERATIONS),
                   "--elites", "8", "--restarts", "1", "--batch-source-grasps", str(SEARCH_BATCH),
                   "--keep-per-grasp", str(KEEP_PER_GRASP), "--keep-per-object", str(PART_CAP),
                   "--cpu-final-candidates", str(PART_CAP), "--search-pipeline", "gpu",
                   "--reuse-gpu-sim", "--log-rejected", "--seed", str(SEED + 1009 * part),
                   "--source-indices", *(str(i) for i in part_indices(part)), "--out", str(dst)]
            log = dst.with_suffix(".log")
            with log.open("w") as stream:
                process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode or not dst.exists():
                raise RuntimeError(f"CEM-288 search failed for {obj} part{part}; inspect {log}")
            check_search(obj, part, source_sha)
            print(f"searched SVH–DRO CEM-288: {obj} part{part} ({SOURCES_PER_PART}/64 targets)", flush=True)


def freeze_top1(candidates):
    """Pick an eligible pre-recheck candidate per source, never using CPU labels.

    Equal GPU scores are broken by a hash of the sampled search parameters;
    serialized candidate ordering may otherwise depend on CPU-final sorting.
    """
    chosen = [None] * 64
    for index, cand in enumerate(candidates):
        sid = cand["source_index"]
        score = cand["gpu_rank_score"]
        if sid not in range(64) or not math.isfinite(score):
            raise ValueError(f"Invalid source/GPU rank in {cand['id']}")
        previous = chosen[sid]
        if previous is None or (score, cand["search_params_sha256"]) < (
                candidates[previous]["gpu_rank_score"], candidates[previous]["search_params_sha256"]):
            chosen[sid] = index
    return chosen


def outcomes(candidates, labels, top1_indices):
    if len(candidates) != len(labels) or len(top1_indices) != 64:
        raise ValueError("Incomplete candidate labels or top-1 selections")
    multi = [False] * 64
    for cand, passed in zip(candidates, labels):
        if passed:
            multi[cand["source_index"]] = True
    top1 = [index is not None and bool(labels[index]) for index in top1_indices]
    if any(a and not b for a, b in zip(top1, multi)):
        raise ValueError("Top-1 success must be a subset of multi-candidate success")
    return top1, multi


def recheck(objects):
    # A fresh process must load the simulator/IsaacGym before importing torch.
    sys.path.insert(0, str(ROOT / "eval"))
    import generate_svh_train_bsearch as validator
    import torch

    source_sha = check_source()
    for obj in objects:
        candidates = []
        for part in range(64 // SOURCES_PER_PART):
            row = check_search(obj, part, source_sha)["rows"][0]
            for kind in ("passing_candidates", "cpu_final_rejected_candidates"):
                for i, cand in enumerate(row.get(kind, [])):
                    params = torch.as_tensor(cand["params"], dtype=torch.float32).view(15).contiguous()
                    candidates.append({"id": f"part{part:02d}/{kind}/{i}",
                                       "source_index": int(cand["source_grasp_index"]),
                                       "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(26),
                                       "gpu_rank_score": float(cand["gpu_rank_score"]),
                                       "search_params_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
                                       "search_cpu_success": bool(cand["cpu_success"])})
        # Freeze all 64 choices BEFORE reading or computing any fresh label.
        top1_indices = freeze_top1(candidates)
        top1_ids = [candidates[i]["id"] if i is not None else None for i in top1_indices]
        search_shas = pool_sha(obj)
        dest = result_path(obj)
        if dest.exists():
            result = json.loads(dest.read_text())
            if (result.get("search_part_sha256") != search_shas or result.get("source_sha256") != source_sha
                    or result.get("battery") != BATTERY or result.get("verify_batch_size") != VERIFY_BATCH
                    or result.get("tail_padded") is not True or result.get("candidate_count") != len(candidates)
                    or result.get("candidate_ids") != [c["id"] for c in candidates]
                    or len(result.get("success", [])) != 64):
                raise ValueError(f"Stale one-time verification; not overwriting {dest}")
            if (len(result.get("candidate_success", [])) != len(candidates)
                    or result.get("source_indices") != [c["source_index"] for c in candidates]
                    or ("candidate_search_params_sha256" in result and
                        result["candidate_search_params_sha256"] != [c["search_params_sha256"] for c in candidates])):
                raise ValueError(f"Incomplete existing candidate-level labels: {dest}")
            top1, multi = outcomes(candidates, result["candidate_success"], top1_indices)
            if result["success"] != multi:
                raise ValueError(f"Stored multi-candidate labels disagree with candidate-level labels: {dest}")
            additions = {"selection_rule": "minimum search-time gpu_rank_score within CPU-final-evaluated pool; search-parameter hash breaks ties",
                         "candidate_gpu_rank_scores": [c["gpu_rank_score"] for c in candidates],
                         "candidate_search_params_sha256": [c["search_params_sha256"] for c in candidates],
                         "top1_candidate_indices": top1_indices,
                         "top1_candidate_ids": top1_ids,
                         "top1_success": top1, "multi_success": multi,
                         "top1_old20_success": sum(top1[:20]),
                         "top1_new44_success": sum(top1[20:]),
                         "multi_old20_success": sum(multi[:20]),
                         "multi_new44_success": sum(multi[20:])}
            for key, value in additions.items():
                if key in result and result[key] != value:
                    raise ValueError(f"Conflicting {key} in existing recheck result: {dest}")
            if any(key not in result for key in additions):
                result.update(additions)  # only annotate our own previous single recheck; never re-simulate
                tmp = dest.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
                tmp.replace(dest)
            print(f"reuse one-time recheck: {obj}", flush=True)
            continue
        distances, labels = [None] * len(candidates), [False] * len(candidates)
        chunks = []
        for start in range(0, len(candidates), VERIFY_BATCH):
            part = candidates[start:start + VERIFY_BATCH]
            q = torch.stack([c["q"] for c in part])
            if len(part) < VERIFY_BATCH:
                q = torch.cat([q, q[-1:].expand(VERIFY_BATCH - len(part), -1)], dim=0)
            d, _ = validator.score_fresh(obj, q, BATTERY, use_gpu_pipeline=False)
            chunks.append({"real_candidates": len(part), "simulated_envs": int(q.shape[0])})
            for offset, _ in enumerate(part):
                value = float(d[offset])
                distances[start + offset] = value if math.isfinite(value) else None
                labels[start + offset] = bool(math.isfinite(value) and value <= THRESHOLD)
        top1, multi = outcomes(candidates, labels, top1_indices)
        successful = [sid for sid, ok in enumerate(multi) if ok]
        result = {"method": "CEM-288", "hand": "svh", "generator": "DRO",
                  "object_name": obj, "source_sha256": source_sha, "search_part_sha256": search_shas,
                  "gpu_evaluations_per_source": GPU_POPULATION * GENERATIONS,
                  "battery": BATTERY, "threshold_m": THRESHOLD,
                  "verify_batch_size": VERIFY_BATCH, "tail_padded": True, "shuffle": False,
                  "revalidation_count": 1, "source_count": 64,
                  "candidate_count": len(candidates), "cpu_batches": chunks,
                  "candidate_ids": [c["id"] for c in candidates],
                  "source_indices": [c["source_index"] for c in candidates],
                  "candidate_gpu_rank_scores": [c["gpu_rank_score"] for c in candidates],
                  "candidate_search_params_sha256": [c["search_params_sha256"] for c in candidates],
                  "search_cpu_success": [c["search_cpu_success"] for c in candidates],
                  "distance_m": distances, "candidate_success": labels,
                  "selection_rule": "minimum search-time gpu_rank_score within CPU-final-evaluated pool; search-parameter hash breaks ties",
                  "top1_candidate_indices": top1_indices,
                  "top1_candidate_ids": top1_ids,
                  "top1_success": top1, "multi_success": multi,
                  "success": multi, "successful_source_indices": successful,
                  "top1_old20_success": sum(top1[:20]),
                  "top1_new44_success": sum(top1[20:]),
                  "multi_old20_success": sum(multi[:20]),
                  "multi_new44_success": sum(multi[20:]),
                  "old20_success": sum(multi[:20]), "new44_success": sum(multi[20:])}
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        tmp.replace(dest)
        print(f"rechecked {obj}: top-1={sum(top1)}/64, "
              f"multi={sum(multi)}/64 (same physical recheck)", flush=True)


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    low = int(x)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (x - low)


def summarize():
    source_sha = check_source()
    totals = {name: {"old": 0, "new": 0, "object_rates": []} for name in ("top1", "multi")}
    per_object = {}
    for obj in OBJECTS:
        path = result_path(obj)
        if not path.is_file():
            raise FileNotFoundError(f"Run one recheck for all ten objects first: {path}")
        result = json.loads(path.read_text())
        if (result.get("source_sha256") != source_sha or result.get("search_part_sha256") != pool_sha(obj)
                or result.get("object_name") != obj or result.get("battery") != BATTERY
                or result.get("verify_batch_size") != VERIFY_BATCH or result.get("tail_padded") is not True
                or result.get("revalidation_count") != 1 or len(result.get("success", [])) != 64):
            raise ValueError(f"Incorrect or stale result: {path}")
        count = result["candidate_count"]
        if (len(result.get("candidate_success", [])) != count
                or len(result.get("candidate_ids", [])) != count
                or len(result.get("source_indices", [])) != count
                or len(result.get("candidate_gpu_rank_scores", [])) != count
                or len(result.get("candidate_search_params_sha256", [])) != count):
            raise ValueError(f"Incomplete candidate-level data: {path}")
        candidates = [{"id": cid, "source_index": sid, "gpu_rank_score": float(score),
                       "search_params_sha256": phash}
                      for cid, sid, score, phash in zip(
                          result["candidate_ids"], result["source_indices"],
                          result["candidate_gpu_rank_scores"], result["candidate_search_params_sha256"])]
        top1_indices = freeze_top1(candidates)
        if (result.get("top1_candidate_indices") != top1_indices
                or result.get("top1_candidate_ids") != [
                    candidates[i]["id"] if i is not None else None for i in top1_indices]):
            raise ValueError(f"Top-1 selection was not frozen from GPU scores: {path}")
        top1, multi = outcomes(candidates, result["candidate_success"], top1_indices)
        if (result.get("top1_success") != top1 or result.get("multi_success") != multi
                or result["success"] != multi or
                result.get("top1_old20_success") != sum(top1[:20]) or
                result.get("top1_new44_success") != sum(top1[20:]) or
                result.get("multi_old20_success") != sum(multi[:20]) or
                result.get("multi_new44_success") != sum(multi[20:])):
            raise ValueError(f"Stored top-1/multi recovery does not match raw candidate labels: {path}")
        per_object[obj] = {"candidate_count": count,
                           "sources_with_top1_candidate": sum(i is not None for i in top1_indices)}
        for name, indicators in (("top1", top1), ("multi", multi)):
            n_old, n_new = sum(indicators[:20]), sum(indicators[20:])
            totals[name]["old"] += n_old
            totals[name]["new"] += n_new
            totals[name]["object_rates"].append((n_old + n_new) / 64 * 100)
            per_object[obj][name] = {"old20_success": n_old, "new44_success": n_new,
                                     "total64_success": n_old + n_new,
                                     "rate_pct": (n_old + n_new) / 64 * 100,
                                     "source_success": indicators}

    results = {}
    rng = random.Random(20261221)
    resamples = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    for name in ("top1", "multi"):
        old, new = totals[name]["old"], totals[name]["new"]
        rates = totals[name]["object_rates"]
        boots = [sum(rates[i] for i in draw) / 10 for draw in resamples]
        results[name] = {"old20": {"success": old, "total": 200, "rate_pct": old / 2},
                         "new44": {"success": new, "total": 440, "rate_pct": new / 4.4},
                         "overall": {"success": old + new, "total": 640,
                                     "rate_pct": (old + new) / 6.4,
                                     "object_cluster_95ci_pct": [quantile(boots, .025), quantile(boots, .975)]}}
    differences = [m - t for m, t in zip(totals["multi"]["object_rates"],
                                          totals["top1"]["object_rates"])]
    boots = [sum(differences[i] for i in draw) / 10 for draw in resamples]
    report = {"method": "SVH–DRO CEM-288",
              "metric": "GPU-ranked top-1 and candidate-set recovery from the same one-time full-pool CPU-state recheck",
              "battery": BATTERY, "gpu_samples_per_source": 288,
              "source_sha256": source_sha, "revalidation_count": 1,
              "selection_rule": "minimum search-time gpu_rank_score within CPU-final-evaluated pool; search-parameter hash breaks ties",
              "top1": results["top1"], "multi": results["multi"],
              "multi_minus_top1_pp": results["multi"]["overall"]["rate_pct"] - results["top1"]["overall"]["rate_pct"],
              "multi_minus_top1_object_cluster_95ci_pp": [quantile(boots, .025), quantile(boots, .975)],
              "per_object": per_object}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, display in (("top1", "GPU top-1"), ("multi", "any candidate")):
        entry = results[name]
        print(f"SVH–DRO CEM-288 {display} | original: {entry['old20']['success']}/200 = "
              f"{entry['old20']['rate_pct']:.2f}% | new: {entry['new44']['success']}/440 = "
              f"{entry['new44']['rate_pct']:.2f}% | merged: {entry['overall']['success']}/640 = "
              f"{entry['overall']['rate_pct']:.2f}%")
    print(f"multi - top-1: {report['multi_minus_top1_pp']:+.2f} pp; "
          f"object-cluster 95% CI {report['multi_minus_top1_object_cluster_95ci_pp']}")
    print(f"Details: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "search", "recheck", "summarize"))
    parser.add_argument("--object", choices=OBJECTS,
                        help="Optional one-object pilot for search/recheck; summary always needs all 10")
    parser.add_argument("--part", type=int, choices=range(64 // SOURCES_PER_PART),
                        help="Optional one eight-source part; search only, for a short pilot")
    args = parser.parse_args()
    if args.object and args.stage not in ("search", "recheck"):
        parser.error("--object is only valid for search/recheck")
    if args.part is not None and (args.stage != "search" or args.object is None):
        parser.error("--part requires search --object")
    objects = (args.object,) if args.object else OBJECTS
    if args.stage == "check":
        check_source()
    elif args.stage == "search":
        search(objects, only_part=args.part)
    elif args.stage == "recheck":
        recheck(objects)
    else:
        summarize()


if __name__ == "__main__":
    main()
