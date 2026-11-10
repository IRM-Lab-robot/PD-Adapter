#!/usr/bin/env python3
"""Rerun CEM-288 independently for all six hand-generator branches.

Each source receives one CEM restart, 72 proposals × 4 generations = 288
GPU evaluations; four sources share a 288-environment search batch. Each
branch has its own search/checkpoints, single CPU-state recheck, per-object
report, and ten-object summary. Previous CEM-288 artifacts remain separate.

Run from new/ with the three_papers environment, e.g.:
  python DRO/run_cem288_six.py check --line all
  python DRO/run_cem288_six.py search --line ability-dro --object contactdb+banana --part 0
  python DRO/run_cem288_six.py all --line ability-dro
  python DRO/run_cem288_six.py summarize --line ability-dro
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "DRO/run_sample288_six.py"
spec = importlib.util.spec_from_file_location("cem288_six_recheck_engine", ENGINE)
if spec is None or spec.loader is None:
    raise ImportError(ENGINE)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

engine.OUT = ROOT / "results/supplement_cem288_six_independent_b64"
engine.POPULATION = 72
engine.GENERATIONS = 4
engine.SEARCH_BATCH = 4
engine.METHOD = "CEM-288-Six-Independent"
engine.METHOD_SLUG = "cem288_six_independent"
engine.WORKER_ENTRYPOINT = Path(__file__).resolve()
engine.shared.OUT = engine.OUT
engine.shared.POPULATION = engine.POPULATION
engine.shared.GENERATIONS = engine.GENERATIONS
engine.shared.SEARCH_BATCH = engine.SEARCH_BATCH


def report_object(line: str, obj: str) -> dict:
    """Verify and summarize one branch/object without requiring the other nine."""
    path = engine.result_path(line, obj)
    record = json.loads(path.read_text())
    source_sha = engine.shared.check_source(line)
    candidates, pool_shas = engine.collect(line, obj, source_sha)
    selected = engine.shared.metrics.freeze_top1(candidates)
    ids = [candidates[i]["id"] if i is not None else None for i in selected]
    if (record.get("line") != line or record.get("method") != engine.METHOD
            or record.get("object_name") != obj or record.get("source_sha256") != source_sha
            or record.get("search_part_sha256") != pool_shas or record.get("battery") != engine.BATTERY
            or record.get("validator_projection") != engine.validation_projection(line)
            or record.get("gpu_evaluations_per_source") != 288
            or record.get("verify_batch_size") != engine.VERIFY_BATCH
            or record.get("revalidation_count") != 1 or record.get("tail_padded") is not True
            or record.get("candidate_count") != len(candidates)
            or record.get("candidate_ids") != [c["id"] for c in candidates]
            or record.get("source_indices") != [c["source_index"] for c in candidates]
            or record.get("candidate_search_params_sha256")
            != [c["search_params_sha256"] for c in candidates]
            or record.get("candidate_gpu_rank_scores") != [c["gpu_rank_score"] for c in candidates]
            or record.get("top1_candidate_indices") != selected
            or record.get("top1_candidate_ids") != ids
            or len(record.get("candidate_success", [])) != len(candidates)):
        raise ValueError(f"Incomplete or incompatible single-object recheck: {path}")
    top1, multi = engine.shared.metrics.outcomes(candidates, record["candidate_success"], selected)
    if record.get("top1_success") != top1 or record.get("multi_success") != multi:
        raise ValueError(f"Stored recovery does not match candidate labels: {path}")
    metrics = {}
    for name, labels in (("top1", top1), ("multi", multi)):
        metrics[name] = {"old20": sum(labels[:20]), "new44": sum(labels[20:]),
                         "overall": sum(labels), "total": 64, "rate_pct": sum(labels) / 64 * 100,
                         "source_success": labels}
        if (record.get(f"{name}_old20_success") != metrics[name]["old20"]
                or record.get(f"{name}_new44_success") != metrics[name]["new44"]):
            raise ValueError(f"Stored source split differs from labels: {path}")
    report = {"line": line, "method": engine.METHOD, "object_name": obj,
              "source_sha256": source_sha, "battery": engine.BATTERY,
              "validator_projection": engine.validation_projection(line),
              "gpu_evaluations_per_source": engine.POPULATION * engine.GENERATIONS,
              "recheck_sha256": engine.sha(path), "search_part_sha256": pool_shas,
              "candidate_count": len(candidates),
              "sources_with_gpu_ranked_candidate": sum(i is not None for i in selected),
              "top1": metrics["top1"], "multi": metrics["multi"]}
    dest = engine.OUT / line / "reports" / f"{engine.shared.metrics.safe(obj)}.json"
    engine.atomic_json(dest, report)
    print(f"{line}/{obj} {engine.METHOD}: top-1 {metrics['top1']['overall']}/64, "
          f"multi {metrics['multi']['overall']}/64 | {dest}", flush=True)
    return report


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # The reused recheck engine launches a fresh worker with Isaac Gym
        # imported before torch. Do not import the validator in this manager.
        engine.main()
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "search", "recheck", "summarize", "all"))
    parser.add_argument("--line", required=True, choices=(*engine.LINES, "all"),
                        help="One independent hand-generator branch, or all six sequentially")
    parser.add_argument("--object", choices=engine.OBJECTS,
                        help="Run/evaluate a single object; omit for all ten")
    parser.add_argument("--part", type=int, choices=range(64 // engine.PART_SIZE),
                        help="One eight-source search block; requires search, one line and one object")
    parser.add_argument("--max-worker-groups", type=int,
                        help="Pause recheck after this many fresh, at-most-four-batch processes")
    args = parser.parse_args()
    if args.object is not None and args.stage == "check":
        parser.error("--object does not apply to check")
    if args.part is not None and (args.stage != "search" or args.object is None or args.line == "all"):
        parser.error("--part requires search --line <one branch> --object <one object>")
    if args.max_worker_groups is not None and (
            args.max_worker_groups < 1 or args.stage not in ("recheck", "all")):
        parser.error("--max-worker-groups requires recheck/all and a positive number")
    lines = tuple(engine.LINES) if args.line == "all" else (args.line,)
    objects = (args.object,) if args.object else engine.OBJECTS
    for line in lines:
        if args.stage == "check":
            engine.shared.check_source(line)
        elif args.stage == "search":
            engine.search(line, objects, args.part)
        elif args.stage == "recheck":
            engine.recheck(line, objects, args.max_worker_groups)
        elif args.stage == "summarize":
            if args.object is not None:
                report_object(line, args.object)
            else:
                engine.summarize(line)
        else:
            engine.search(line, objects)
            engine.recheck(line, objects, args.max_worker_groups)
            if args.max_worker_groups is None:
                if args.object is not None:
                    report_object(line, args.object)
                else:
                    engine.summarize(line)


if __name__ == "__main__":
    main()
