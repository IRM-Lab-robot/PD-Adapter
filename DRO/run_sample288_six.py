#!/usr/bin/env python3
"""Six-line Sample-288: 10×64 sources, one fresh recheck, two recovery rates.

The projected source seeds a single P=288, G=1 branch. A fixed GPU-ranked
top-1 and any-retained-candidate recovery share the same CPU-state labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/supplement_sample288_six_b64"
SHARED_PATH = ROOT / "DRO/run_cem288_five.py"
spec = importlib.util.spec_from_file_location("sample288_source_and_pool_audit", SHARED_PATH)
if spec is None or spec.loader is None:
    raise ImportError(SHARED_PATH)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

# Reuse only the audited source/part/candidate-schema helpers, never CEM's
# method-named result files or its long-lived physical recheck process.
POPULATION = 288
GENERATIONS = 1
SEARCH_BATCH = 1
METHOD = "Sample-288"
METHOD_SLUG = "sample288"
WRIST_XYZ_STD = .01
WRIST_ROT_STD = .10
U_STD = .25
SPATIAL_SAMPLING = "local"
WORKER_ENTRYPOINT = Path(__file__).resolve()
WORKER_ARGS = ()
PROJECTION_MODE = "none"  # Legacy Ability/SVH; Inspire already uses its historical LSQ validator.
shared.OUT = OUT
shared.LINES = dict(shared.LINES)
shared.LINES["svh-dro"] = ("svh", "DRO", shared.ROOT / "results/pd_refiner_svh/svh_test10_zero_shot_b20.pt")
shared.POPULATION = POPULATION
shared.GENERATIONS = GENERATIONS
shared.SEARCH_BATCH = SEARCH_BATCH  # 1×288 envs matches CEM-288's 4×72 GPU env count
LINES = shared.LINES
OBJECTS = shared.OBJECTS
BATTERY = shared.BATTERY
PART_SIZE = shared.PART_SIZE
VERIFY_BATCH = shared.VERIFY_BATCH
WORKER_BATCHES = 4


def sha(path: Path) -> str:
    return shared.metrics.sha(path)


def validation_projection(line: str) -> str:
    return "lsq" if LINES[line][0] == "inspire" else PROJECTION_MODE


def original_validation_projection(line: str) -> str:
    return "lsq" if LINES[line][0] == "inspire" else "none"


def result_path(line: str, obj: str) -> Path:
    return OUT / line / "revalidation_once" / f"{shared.metrics.safe(obj)}.json"


def cache_path(line: str, obj: str) -> Path:
    return OUT / line / "recheck_cache" / f"{shared.metrics.safe(obj)}.pt"


def batch_path(line: str, obj: str, batch: int) -> Path:
    return OUT / line / "revalidation_batches" / shared.metrics.safe(obj) / f"batch{batch:04d}.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def check_pool(line: str, obj: str, part: int, source_sha: str):
    row = shared.check_pool(line, obj, part, source_sha)
    if LINES[line][0] == "inspire":
        # The Inspire search stores an arm tag. Detect a CEM artifact accidentally
        # copied into this otherwise independent sample output directory.
        import torch
        path = shared.search_file(line, obj, part)
        args = torch.load(path, map_location="cpu", weights_only=False)["args"]
        if args.get("arm_id") != METHOD_SLUG:
            raise ValueError(f"Wrong Inspire search arm in {path}")
    return row


def search(line: str, objects, only_part=None) -> None:
    source_sha = shared.check_source(line)
    script = shared.driver(line)
    if not script.is_file():
        raise FileNotFoundError(script)
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    for obj in objects:
        for part in ((only_part,) if only_part is not None else range(64 // PART_SIZE)):
            dest = shared.search_file(line, obj, part)
            if dest.exists():
                check_pool(line, obj, part, source_sha)
                print(f"reuse {line} {METHOD} {obj} part{part}", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--zero-file", str(shared.source_file(line)),
                   "--objects", obj, "--max-grasps", "64", "--selection-battery", BATTERY,
                   "--population", str(POPULATION), "--generations", str(GENERATIONS), "--elites", "8",
                   "--restarts", "1", "--batch-source-grasps", str(SEARCH_BATCH),
                   "--keep-per-grasp", "3", "--keep-per-object", "24",
                   "--cpu-final-candidates", "24", "--search-pipeline", "gpu",
                   "--reuse-gpu-sim", "--log-rejected", "--seed", str(10001 + 1009 * part),
                   "--wrist-xyz-std", str(WRIST_XYZ_STD),
                   "--wrist-rot-std", str(WRIST_ROT_STD), "--u-std", str(U_STD),
                   "--spatial-sampling", SPATIAL_SAMPLING,
                   "--source-indices", *(str(i) for i in shared.source_ids(part)), "--out", str(dest)]
            if LINES[line][0] == "inspire":
                cmd.extend(("--allow-non-train", "--arm-id", METHOD_SLUG))
            log = dest.with_suffix(".log")
            with log.open("w") as stream:
                process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode or not dest.is_file():
                raise RuntimeError(f"{METHOD} search failed; inspect {log}")
            check_pool(line, obj, part, source_sha)
            print(f"searched {line} {METHOD} {obj} part{part}: eight sources", flush=True)


def collect(line: str, obj: str, source_sha: str):
    import torch

    candidates, pool_hashes = [], []
    for part in range(64 // PART_SIZE):
        row = check_pool(line, obj, part, source_sha)
        pool_hashes.append(sha(shared.search_file(line, obj, part)))
        for kind in ("passing_candidates", "cpu_final_rejected_candidates"):
            for index, cand in enumerate(row[kind]):
                params = torch.as_tensor(cand["params"], dtype=torch.float32).contiguous()
                candidates.append({"id": f"part{part:02d}/{kind}/{index}",
                                   "source_index": int(cand["source_grasp_index"]),
                                   "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(shared.DIMS[LINES[line][0]]),
                                   "gpu_rank_score": float(cand["gpu_rank_score"]),
                                   "search_params_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
                                   "search_cpu_success": bool(cand["cpu_success"])})
    return candidates, pool_hashes


def ensure_cache(line: str, obj: str, candidates, pool_hashes, top1_ids, source_sha: str, torch):
    path = cache_path(line, obj)
    ids = [c["id"] for c in candidates]
    indices = [c["source_index"] for c in candidates]
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("line") != line or saved.get("source_sha256") != source_sha
                or saved.get("pool_sha256") != pool_hashes
                or saved.get("validator_projection", original_validation_projection(line)) != validation_projection(line)
                or saved.get("candidate_ids") != ids or saved.get("source_indices") != indices
                or saved.get("top1_candidate_ids") != top1_ids
                or torch.as_tensor(saved.get("q")).shape != (len(candidates), shared.DIMS[LINES[line][0]])):
            raise ValueError(f"Refusing to overwrite stale candidate cache: {path}")
    else:
        saved = {"line": line, "object_name": obj, "source_sha256": source_sha,
                  "pool_sha256": pool_hashes, "battery": BATTERY,
                  "validator_projection": validation_projection(line),
                 "verify_batch_size": VERIFY_BATCH, "candidate_ids": ids,
                 "source_indices": indices, "top1_candidate_ids": top1_ids,
                 "q": torch.stack([c["q"] for c in candidates]) if candidates
                 else torch.empty((0, shared.DIMS[LINES[line][0]]))}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(saved, temporary)
        temporary.replace(path)
    return sha(path)


def checked_batch(line: str, obj: str, batch: int, candidates, cache_sha: str):
    path = batch_path(line, obj, batch)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    current = candidates[batch * VERIFY_BATCH:(batch + 1) * VERIFY_BATCH]
    if (record.get("line") != line or record.get("object_name") != obj
            or record.get("batch_index") != batch or record.get("cache_sha256") != cache_sha
            or record.get("candidate_ids") != [c["id"] for c in current]
            or record.get("source_indices") != [c["source_index"] for c in current]
            or record.get("validator_projection", original_validation_projection(line)) != validation_projection(line)
            or record.get("battery") != BATTERY or record.get("verify_batch_size") != VERIFY_BATCH
            or record.get("threshold_m") != shared.metrics.THRESHOLD
            or record.get("simulated_envs") != VERIFY_BATCH
            or len(record.get("distance_m", [])) != len(current)
            or len(record.get("candidate_success", [])) != len(current)
            or any(type(v) is not bool for v in record["candidate_success"])):
        raise ValueError(f"Incomplete/incompatible CPU-state batch: {path}")
    return record


def worker(line: str, obj: str, start_batch: int, count: int, expected_sha: str) -> None:
    # Fresh process: import validator/Isaac Gym BEFORE torch.
    sys.path.insert(0, str(ROOT / "eval"))
    import importlib
    validator = importlib.import_module(f"generate_{LINES[line][0]}_train_bsearch")
    import torch

    path = cache_path(line, obj)
    if not path.is_file() or sha(path) != expected_sha:
        raise ValueError(f"Candidate cache changed before worker started: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if (cache["line"] != line or cache["object_name"] != obj or cache["battery"] != BATTERY
            or cache.get("validator_projection", original_validation_projection(line)) != validation_projection(line)
            or cache["verify_batch_size"] != VERIFY_BATCH):
        raise ValueError(f"Invalid short-lived worker cache: {path}")
    total = len(cache["candidate_ids"])
    batches = (total + VERIFY_BATCH - 1) // VERIFY_BATCH
    if start_batch not in range(batches) or not 1 <= count <= WORKER_BATCHES:
        raise ValueError(f"Worker batch range {start_batch}+{count} out of {batches}")
    for batch in range(start_batch, min(batches, start_batch + count)):
        begin, end = batch * VERIFY_BATCH, min(total, (batch + 1) * VERIFY_BATCH)
        dest = batch_path(line, obj, batch)
        if dest.exists():
            existing = json.loads(dest.read_text())
            if (existing.get("cache_sha256") != expected_sha
                    or existing.get("candidate_ids") != cache["candidate_ids"][begin:end]
                    or existing.get("source_indices") != cache["source_indices"][begin:end]):
                raise ValueError(f"Stale worker result: {dest}")
            continue
        q = cache["q"][begin:end].clone()
        if len(q) < VERIFY_BATCH:
            q = torch.cat([q, q[-1:].expand(VERIFY_BATCH - len(q), -1)], dim=0)
        if LINES[line][0] == "inspire":
            distance, _ = validator.score_fresh(obj, q, BATTERY, use_gpu_pipeline=False)
        else:
            distance, _ = validator.score_fresh(
                obj, q, BATTERY, use_gpu_pipeline=False, projection_mode=validation_projection(line))
        measured = [float(distance[i]) for i in range(end - begin)]
        atomic_json(dest, {"line": line, "object_name": obj, "batch_index": batch,
                           "cache_sha256": expected_sha, "battery": BATTERY,
                           "validator_projection": validation_projection(line),
                           "threshold_m": shared.metrics.THRESHOLD,
                           "verify_batch_size": VERIFY_BATCH, "simulated_envs": int(q.shape[0]),
                           "candidate_ids": cache["candidate_ids"][begin:end],
                           "source_indices": cache["source_indices"][begin:end],
                           "distance_m": [v if math.isfinite(v) else None for v in measured],
                           "candidate_success": [math.isfinite(v) and v <= shared.metrics.THRESHOLD
                                                 for v in measured]})
        print(f"checkpointed {line}/{obj} CPU batch {batch + 1}/{batches}", flush=True)


def recheck(line: str, objects, limit=None) -> None:
    import torch  # The manager never imports Isaac Gym or creates a simulator.

    source_sha = shared.check_source(line)
    started = 0
    for obj in objects:
        candidates, pool_shas = collect(line, obj, source_sha)
        selected = shared.metrics.freeze_top1(candidates)  # before any new labels
        chosen_ids = [candidates[i]["id"] if i is not None else None for i in selected]
        dest = result_path(line, obj)
        if dest.exists():
            saved = json.loads(dest.read_text())
            if (saved.get("method") != METHOD or saved.get("line") != line
                    or saved.get("source_sha256") != source_sha
                    or saved.get("validator_projection", original_validation_projection(line)) != validation_projection(line)
                    or saved.get("search_part_sha256") != pool_shas
                    or saved.get("candidate_ids") != [c["id"] for c in candidates]
                    or saved.get("top1_candidate_ids") != chosen_ids
                    or saved.get("battery") != BATTERY or saved.get("verify_batch_size") != VERIFY_BATCH
                    or len(saved.get("candidate_success", [])) != len(candidates)):
                raise ValueError(f"Stale full-pool recheck; not overwriting {dest}")
            if (saved["top1_success"], saved["multi_success"]) != shared.metrics.outcomes(
                    candidates, saved["candidate_success"], selected):
                raise ValueError(f"Candidate labels and saved target recovery differ: {dest}")
            print(f"reuse {line}/{obj} one-time recheck", flush=True)
            continue
        cache_sha = ensure_cache(line, obj, candidates, pool_shas, chosen_ids, source_sha, torch)
        batches = (len(candidates) + VERIFY_BATCH - 1) // VERIFY_BATCH
        for first in range(0, batches, WORKER_BATCHES):
            end = min(batches, first + WORKER_BATCHES)
            if all(checked_batch(line, obj, i, candidates, cache_sha) is not None
                   for i in range(first, end)):
                continue
            cmd = [sys.executable, str(WORKER_ENTRYPOINT), "worker", "--line", line,
                    "--object", obj, "--start-batch", str(first), "--batch-count", str(end - first),
                    "--cache-sha", cache_sha, *WORKER_ARGS]
            log = batch_path(line, obj, first).with_name(f"worker{first:04d}_{end-1:04d}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(f"running {line}/{obj} CPU batches {first}..{end-1}\n")
                stream.flush()
                status = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f"Short-lived worker failed ({status.returncode}); resume recheck. Inspect {log}")
            if any(checked_batch(line, obj, i, candidates, cache_sha) is None for i in range(first, end)):
                raise RuntimeError(f"Worker failed to persist all batches; inspect {log}")
            print(f"{line}/{obj}: saved physical batches {first+1}..{end}/{batches}", flush=True)
            started += 1
            if limit is not None and started >= limit:
                print("Worker limit reached; run recheck again to resume.", flush=True)
                return
        distances, labels, batch_meta = [], [], []
        for batch in range(batches):
            record = checked_batch(line, obj, batch, candidates, cache_sha)
            if record is None:
                raise RuntimeError(f"Missing batch {batch}: {line}/{obj}")
            distances.extend(record["distance_m"])
            labels.extend(record["candidate_success"])
            batch_meta.append({"real_candidates": len(record["candidate_ids"]),
                               "simulated_envs": VERIFY_BATCH})
        if len(labels) != len(candidates):
            raise ValueError(f"Incomplete physical labels for {line}/{obj}")
        top1, multi = shared.metrics.outcomes(candidates, labels, selected)
        atomic_json(dest, {"line": line, "method": METHOD, "object_name": obj,
                           "hand": LINES[line][0], "generator": LINES[line][1],
                           "validator_projection": validation_projection(line),
                           "source_sha256": source_sha, "search_part_sha256": pool_shas,
                           "gpu_evaluations_per_source": POPULATION * GENERATIONS, "battery": BATTERY,
                           "threshold_m": shared.metrics.THRESHOLD,
                           "verify_batch_size": VERIFY_BATCH, "tail_padded": True,
                           "shuffle": False, "revalidation_count": 1,
                           "cache_sha256": cache_sha, "worker_batches_per_process": WORKER_BATCHES,
                           "candidate_count": len(candidates), "cpu_batches": batch_meta,
                           "selection_rule": "minimum search-time gpu_rank_score within CPU-final-evaluated pool; parameter hash breaks ties",
                           "candidate_ids": [c["id"] for c in candidates],
                           "source_indices": [c["source_index"] for c in candidates],
                           "candidate_gpu_rank_scores": [c["gpu_rank_score"] for c in candidates],
                           "candidate_search_params_sha256": [c["search_params_sha256"] for c in candidates],
                           "search_cpu_success": [c["search_cpu_success"] for c in candidates],
                           "distance_m": distances, "candidate_success": labels,
                           "top1_candidate_indices": selected, "top1_candidate_ids": chosen_ids,
                           "top1_success": top1, "multi_success": multi,
                           "top1_old20_success": sum(top1[:20]), "top1_new44_success": sum(top1[20:]),
                           "multi_old20_success": sum(multi[:20]), "multi_new44_success": sum(multi[20:])})
        print(f"{line}/{obj} {METHOD}: top-1 {sum(top1)}/64, multi {sum(multi)}/64", flush=True)


def summarize(line: str) -> None:
    source_sha = shared.check_source(line)
    counts = {name: [0, 0] for name in ("top1", "multi")}
    object_rates = {name: [] for name in counts}
    per_object = {}
    for obj in shared.OBJECTS:
        path = result_path(line, obj)
        if not path.is_file():
            raise FileNotFoundError(f"Run all ten one-time rechecks first: {path}")
        saved = json.loads(path.read_text())
        if (saved.get("method") != METHOD or saved.get("line") != line
                or saved.get("object_name") != obj or saved.get("source_sha256") != source_sha
                or saved.get("validator_projection", original_validation_projection(line)) != validation_projection(line)
                or saved.get("search_part_sha256") != [sha(shared.search_file(line, obj, p)) for p in range(8)]
                or saved.get("battery") != BATTERY or saved.get("verify_batch_size") != VERIFY_BATCH
                or saved.get("revalidation_count") != 1 or saved.get("tail_padded") is not True):
            raise ValueError(f"Stale/incomplete final record: {path}")
        n = saved["candidate_count"]
        if any(len(saved.get(field, [])) != n for field in (
                "candidate_ids", "candidate_success", "source_indices",
                "candidate_gpu_rank_scores", "candidate_search_params_sha256")):
            raise ValueError(f"Incomplete per-candidate recheck labels: {path}")
        candidates = [{"id": cid, "source_index": idx, "gpu_rank_score": float(score),
                       "search_params_sha256": phash}
                      for cid, idx, score, phash in zip(saved["candidate_ids"], saved["source_indices"],
                                                        saved["candidate_gpu_rank_scores"],
                                                        saved["candidate_search_params_sha256"])]
        selected = shared.metrics.freeze_top1(candidates)
        selected_ids = [candidates[i]["id"] if i is not None else None for i in selected]
        if saved["top1_candidate_indices"] != selected or saved["top1_candidate_ids"] != selected_ids:
            raise ValueError(f"Stored top-1 differs from search-time GPU scores: {path}")
        top1, multi = shared.metrics.outcomes(candidates, saved["candidate_success"], selected)
        if saved["top1_success"] != top1 or saved["multi_success"] != multi:
            raise ValueError(f"Candidate and target success disagree: {path}")
        per_object[obj] = {"candidate_count": n,
                           "targets_with_gpu_ranked_candidate": sum(i is not None for i in selected)}
        for name, labels in (("top1", top1), ("multi", multi)):
            before, added = sum(labels[:20]), sum(labels[20:])
            if (saved[f"{name}_old20_success"] != before or saved[f"{name}_new44_success"] != added):
                raise ValueError(f"Old/new source index accounting differs: {path}")
            counts[name][0] += before
            counts[name][1] += added
            object_rates[name].append((before + added) / 64 * 100)
            per_object[obj][name] = {"old20_success": before, "new44_success": added,
                                     "overall64_success": before + added,
                                     "rate_pct": (before + added) / 64 * 100,
                                     "source_success": labels}
    rng = random.Random(20261221)
    draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    results = {}
    for name in ("top1", "multi"):
        before, added = counts[name]
        boot = [sum(object_rates[name][i] for i in draw) / 10 for draw in draws]
        results[name] = {"old20": {"success": before, "total": 200, "rate_pct": before / 2},
                         "new44": {"success": added, "total": 440, "rate_pct": added / 4.4},
                         "overall": {"success": before + added, "total": 640,
                                     "rate_pct": (before + added) / 6.4,
                                     "object_cluster_95ci_pct": [shared.metrics.quantile(boot, .025),
                                                                  shared.metrics.quantile(boot, .975)]}}
    delta = [multi - top for top, multi in zip(object_rates["top1"], object_rates["multi"])]
    delta_boot = [sum(delta[i] for i in draw) / 10 for draw in draws]
    report = {"line": line, "method": METHOD, "battery": BATTERY,
              "validator_projection": validation_projection(line),
               "source_sha256": source_sha, "search_samples_per_source": POPULATION * GENERATIONS,
               "search_configuration": f"one default/projection branch, population={POPULATION}, generations={GENERATIONS}, batch_source_grasps={SEARCH_BATCH}",
              "revalidation_count": 1, "verify_batch_size": VERIFY_BATCH, "tail_padded": True,
              "top1": results["top1"], "multi": results["multi"],
              "multi_minus_top1_pp": results["multi"]["overall"]["rate_pct"] - results["top1"]["overall"]["rate_pct"],
              "multi_minus_top1_object_cluster_95ci_pp": [shared.metrics.quantile(delta_boot, .025),
                                                            shared.metrics.quantile(delta_boot, .975)],
              "per_object": per_object}
    dest = OUT / line / "summary.json"
    atomic_json(dest, report)
    for name, item in results.items():
        print(f"{line} {METHOD} {name}: old {item['old20']['success']}/200 = {item['old20']['rate_pct']:.2f}% | "
              f"new {item['new44']['success']}/440 = {item['new44']['rate_pct']:.2f}% | "
              f"overall {item['overall']['success']}/640 = {item['overall']['rate_pct']:.2f}%")
    print(f"saved {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "search", "recheck", "summarize", "worker"))
    parser.add_argument("--line", choices=(*LINES, "all"), default="all")
    parser.add_argument("--object", choices=shared.OBJECTS,
                        help="One-object search/recheck pilot")
    parser.add_argument("--part", type=int, choices=range(64 // PART_SIZE),
                        help="One eight-source search part; requires one --line and --object")
    parser.add_argument("--max-worker-groups", type=int,
                        help="Recheck pilot: launch at most this many short worker processes")
    parser.add_argument("--start-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cache-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "worker":
        if (args.line == "all" or args.object is None or args.start_batch is None
                or args.batch_count is None or not args.cache_sha
                or args.part is not None or args.max_worker_groups is not None):
            parser.error("worker requires one --line/--object, batch range and cache SHA")
        worker(args.line, args.object, args.start_batch, args.batch_count, args.cache_sha)
        return
    if args.start_batch is not None or args.batch_count is not None or args.cache_sha is not None:
        parser.error("Batch worker arguments are only for worker")
    if args.object and args.stage not in ("search", "recheck"):
        parser.error("--object is only for search/recheck")
    if args.part is not None and (args.stage != "search" or args.object is None or args.line == "all"):
        parser.error("--part requires search, a single --line and --object")
    if args.max_worker_groups is not None and (args.stage != "recheck" or args.max_worker_groups < 1):
        parser.error("--max-worker-groups must be positive and only applies to recheck")
    lines = tuple(LINES) if args.line == "all" else (args.line,)
    for line in lines:
        if args.stage == "check":
            shared.check_source(line)
        elif args.stage == "search":
            search(line, (args.object,) if args.object else shared.OBJECTS, args.part)
        elif args.stage == "recheck":
            recheck(line, (args.object,) if args.object else shared.OBJECTS, args.max_worker_groups)
        else:
            summarize(line)


if __name__ == "__main__":
    main()
