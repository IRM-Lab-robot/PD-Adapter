#!/usr/bin/env python3
"""Six-source-line Raw-active/Φ and Native-LSQ/Φ direct-grasp CPU validation.

Raw takes the source's active finger components (clipped to active limits) and
drives passive joints via Φ. Native LSQ fits active commands to ALL original
finger joints and uses the same Φ. Both preserve the source wrist and use the
same new LSQ projection of controller open/close targets on ALL three hands.
No search, no model predictions, and exactly one physical attempt per target.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("raw_native_source_audit",
                                             ROOT / "DRO/run_sample288_six.py")
if spec is None or spec.loader is None:
    raise ImportError("DRO/run_sample288_six.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

OUT = ROOT / "results/supplement_raw_native_six_b64"
LINES = ("inspire-dro", "inspire-gen", "ability-dro", "ability-gen", "svh-dro", "svh-gen")
OBJECTS = engine.OBJECTS
MODES = ("raw", "native")
BATTERY = engine.BATTERY
THRESHOLD = engine.shared.metrics.THRESHOLD
BATCH = 32
METHOD = "Source-RawActivePhi-vs-NativeLSQPhi"
COUPLING_SPECS = {"inspire": ROOT / "assets/inspire/coupling.json",
                  "ability": ROOT / "assets/ability_hand/coupling.json",
                  "svh": ROOT / "assets/svh_hand/coupling.json"}


def safe(obj: str) -> str:
    return engine.shared.metrics.safe(obj)


def sha(path: Path) -> str:
    return engine.sha(path)


def atomic_json(path: Path, payload: dict) -> None:
    engine.atomic_json(path, payload)


def cache_path(line: str, obj: str) -> Path:
    return OUT / line / "source_cache" / f"{safe(obj)}.pt"


def batch_path(line: str, obj: str, mode: str, batch: int) -> Path:
    return OUT / line / "physical_batches" / safe(obj) / f"{mode}_batch{batch:02d}.json"


def result_path(line: str, obj: str) -> Path:
    return OUT / line / "once" / f"{safe(obj)}.json"


def source_sha(line: str) -> str:
    if line not in LINES:
        raise ValueError(line)
    return engine.shared.check_source(line)


def prepare(line: str, obj: str, source_hash=None) -> str:
    """Freeze and audit both physical inputs before any new success labels."""
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT.parent / "DRO-Grasp"))
    from DRO import raw_native_projection as projection
    from utils.hand_model import create_hand_model

    source_hash = source_hash or source_sha(line)
    hand = engine.LINES[line][0]
    source_file = engine.shared.source_file(line)
    original = next(row for row in torch.load(source_file, map_location="cpu", weights_only=False)
                    if row["object_name"] == obj)
    q = torch.as_tensor(original["q_dro_order"], dtype=torch.float32)
    if tuple(q.shape) != (64, engine.shared.DIMS[hand]) or not torch.isfinite(q).all():
        raise ValueError(f"Source contains the wrong number of finite grasps: {line}/{obj}")
    model = create_hand_model(hand, device=torch.device("cpu"))
    names = model.get_joint_orders()
    if len(names) != q.shape[1]:
        raise ValueError(f"Hand/source joint order dimensions differ: {line}/{obj}")
    lower, upper = model.pk_chain.get_joint_limits()
    coupling = projection.load_spec(hand)
    raw, native = projection.project(q, names[6:], np.asarray(lower[6:]),
                                     np.asarray(upper[6:]), coupling)
    if (raw.shape != q.shape or native.shape != q.shape
            or not torch.equal(raw[:, :6], q[:, :6])
            or not torch.equal(native[:, :6], q[:, :6])
            or not torch.isfinite(raw).all() or not torch.isfinite(native).all()):
        raise ValueError(f"Projection changed wrists or yielded invalid targets: {line}/{obj}")
    coupling_sha = sha(COUPLING_SPECS[hand])
    payload = {"line": line, "object_name": obj, "method": METHOD,
               "source_sha256": source_hash, "coupling_sha256": coupling_sha,
               "source_file": str(source_file), "battery": BATTERY,
               "validator_projection": "lsq", "verify_batch_size": BATCH,
               "source_indices": list(range(64)), "q_source": q,
               "q_raw": raw, "q_native": native}
    dest = cache_path(line, obj)
    if dest.is_file():
        prior = torch.load(dest, map_location="cpu", weights_only=False)
        if (any(prior.get(key) != payload[key] for key in (
                "line", "object_name", "method", "source_sha256", "coupling_sha256",
                "source_file", "battery", "validator_projection", "verify_batch_size", "source_indices"))
                or any(not torch.equal(torch.as_tensor(prior.get(key)), payload[key])
                       for key in ("q_source", "q_raw", "q_native"))):
            raise ValueError(f"Stale raw/native source cache; refusing to overwrite: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(dest)
    print(f"checked {line}/{obj}: 64 aligned Raw and Native commands", flush=True)
    return sha(dest)


def checked_batch(line: str, obj: str, mode: str, batch: int, cache_sha: str):
    path = batch_path(line, obj, mode, batch)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    ids = list(range(batch * BATCH, (batch + 1) * BATCH))
    if (record.get("line") != line or record.get("object_name") != obj
            or record.get("method") != METHOD or record.get("mode") != mode
            or record.get("batch_index") != batch or record.get("source_indices") != ids
            or record.get("cache_sha256") != cache_sha or record.get("battery") != BATTERY
            or record.get("threshold_m") != THRESHOLD
            or record.get("validator_projection") != "lsq"
            or record.get("simulated_envs") != BATCH
            or len(record.get("distance_m", [])) != BATCH
            or len(record.get("success", [])) != BATCH
            or any(type(ok) is not bool for ok in record["success"])
            or any((d is not None and (not isinstance(d, (int, float)) or not math.isfinite(d)))
                   or ok != (d is not None and d <= THRESHOLD)
                   for d, ok in zip(record["distance_m"], record["success"]))):
        raise ValueError(f"Wrong or incomplete physical batch: {path}")
    return record


def worker(line: str, obj: str, cache_sha: str) -> None:
    # Import the Isaac Gym validator before importing torch in this fresh process.
    sys.path.insert(0, str(ROOT / "eval"))
    driver = importlib.import_module(f"generate_{engine.LINES[line][0]}_train_bsearch")
    import torch

    path = cache_path(line, obj)
    if not path.is_file() or sha(path) != cache_sha:
        raise ValueError(f"Source cache changed before physical validation: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["line"] != line or saved["object_name"] != obj
            or saved["battery"] != BATTERY or saved["validator_projection"] != "lsq"):
        raise ValueError(f"Invalid worker cache: {path}")
    for mode in MODES:
        for batch in range(2):
            if checked_batch(line, obj, mode, batch, cache_sha) is not None:
                continue
            ids = list(range(batch * BATCH, (batch + 1) * BATCH))
            q = torch.as_tensor(saved[f"q_{mode}"])[ids].clone()
            if engine.LINES[line][0] == "inspire":
                distance, _ = driver.score_fresh(obj, q, BATTERY, use_gpu_pipeline=False)
            else:
                distance, _ = driver.score_fresh(obj, q, BATTERY, use_gpu_pipeline=False,
                                                 projection_mode="lsq")
            values = [float(distance[i]) for i in range(BATCH)]
            atomic_json(batch_path(line, obj, mode, batch), {
                "line": line, "object_name": obj, "method": METHOD, "mode": mode,
                "batch_index": batch, "cache_sha256": cache_sha,
                "battery": BATTERY, "validator_projection": "lsq",
                "threshold_m": THRESHOLD, "simulated_envs": BATCH,
                "source_indices": ids,
                "distance_m": [value if math.isfinite(value) else None for value in values],
                "success": [math.isfinite(value) and value <= THRESHOLD for value in values]})
            print(f"checkpointed {line}/{obj}: {mode} batch {batch+1}/2", flush=True)


def recheck(line: str, objects, max_workers=None) -> None:
    source_hash = source_sha(line)
    started = 0
    for obj in objects:
        cache_sha = prepare(line, obj, source_hash)
        dest = result_path(line, obj)
        if dest.is_file():
            record = json.loads(dest.read_text())
            if (record.get("line") != line or record.get("object_name") != obj
                    or record.get("method") != METHOD or record.get("cache_sha256") != cache_sha
                    or record.get("source_sha256") != source_hash
                    or record.get("battery") != BATTERY or record.get("validator_projection") != "lsq"
                    or record.get("revalidation_count") != 1 or record.get("verify_batch_size") != BATCH
                    or any(len(record.get(f"{mode}_success", [])) != 64 for mode in MODES)):
                raise ValueError(f"Existing direct-grasp result differs: {dest}")
            for mode in MODES:
                saved = [item for batch in range(2)
                         for item in checked_batch(line, obj, mode, batch, cache_sha)["success"]]
                if saved != record[f"{mode}_success"]:
                    raise ValueError(f"Saved batch labels disagree with {dest}")
            print(f"reuse one-time raw/native validation {line}/{obj}", flush=True)
            continue
        if any(checked_batch(line, obj, mode, batch, cache_sha) is None
               for mode in MODES for batch in range(2)):
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--line", line,
                   "--object", obj, "--cache-sha", cache_sha]
            log = OUT / line / "physical_batches" / safe(obj) / "worker.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                status = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f"Fresh physical worker failed; resume recheck. Inspect {log}")
            started += 1
            if max_workers is not None and started >= max_workers:
                print("Worker limit reached; rerun recheck to finish/merge this object.", flush=True)
                return
        blocks = {(mode, batch): checked_batch(line, obj, mode, batch, cache_sha)
                  for mode in MODES for batch in range(2)}
        if any(value is None for value in blocks.values()):
            raise RuntimeError(f"Missing raw/native physical batch for {line}/{obj}")
        result = {"line": line, "object_name": obj, "method": METHOD,
                  "hand": engine.LINES[line][0], "generator": engine.LINES[line][1],
                  "source_sha256": source_hash, "coupling_sha256": sha(
                      COUPLING_SPECS[engine.LINES[line][0]]),
                  "cache_sha256": cache_sha, "battery": BATTERY,
                  "validator_projection": "lsq", "verify_batch_size": BATCH,
                  "tail_padded": True, "revalidation_count": 1,
                  "source_indices": list(range(64)), "source_count": 64}
        for mode in MODES:
            result[f"{mode}_batch_sha256"] = [sha(batch_path(line, obj, mode, b)) for b in range(2)]
            result[f"{mode}_success"] = [item for batch in range(2)
                                          for item in blocks[(mode, batch)]["success"]]
            result[f"{mode}_distance_m"] = [item for batch in range(2)
                                             for item in blocks[(mode, batch)]["distance_m"]]
            result[f"{mode}_success_count"] = sum(result[f"{mode}_success"])
        atomic_json(dest, result)
        print(f"{line}/{obj}: Raw {result['raw_success_count']}/64, "
              f"Native {result['native_success_count']}/64", flush=True)


def checked_result(line: str, obj: str, source_hash: str) -> dict:
    path = result_path(line, obj)
    row = json.loads(path.read_text())
    if (row.get("line") != line or row.get("object_name") != obj
            or row.get("method") != METHOD or row.get("source_sha256") != source_hash
            or row.get("coupling_sha256") != sha(COUPLING_SPECS[engine.LINES[line][0]])
            or row.get("cache_sha256") != sha(cache_path(line, obj))
            or row.get("battery") != BATTERY or row.get("validator_projection") != "lsq"
            or row.get("verify_batch_size") != BATCH or row.get("tail_padded") is not True
            or row.get("revalidation_count") != 1
            or row.get("source_indices") != list(range(64))):
        raise ValueError(f"Incompatible direct-grasp result: {path}")
    for mode in MODES:
        if (len(row.get(f"{mode}_success", [])) != 64
                or len(row.get(f"{mode}_distance_m", [])) != 64
                or row.get(f"{mode}_success_count") != sum(row[f"{mode}_success"])
                or row.get(f"{mode}_batch_sha256")
                != [sha(batch_path(line, obj, mode, b)) for b in range(2)]):
            raise ValueError(f"Incomplete or changed {mode} validation: {path}")
        labels = [item for b in range(2)
                  for item in checked_batch(line, obj, mode, b, row["cache_sha256"])["success"]]
        if labels != row[f"{mode}_success"]:
            raise ValueError(f"Physical batch and object labels differ: {path}")
    return row


def summary(line: str) -> dict:
    source_hash = source_sha(line)
    records = {obj: checked_result(line, obj, source_hash) for obj in OBJECTS}
    rng = random.Random(20261221)
    draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    results = {}
    for mode in MODES:
        per_object = [records[obj][f"{mode}_success"] for obj in OBJECTS]
        old = sum(sum(values[:20]) for values in per_object)
        extra = sum(sum(values[20:]) for values in per_object)
        rates = [sum(values) / 64 * 100 for values in per_object]
        boots = [sum(rates[i] for i in draw) / 10 for draw in draws]
        results[mode] = {"old20": {"success": old, "total": 200, "rate_pct": old / 2},
                         "new44": {"success": extra, "total": 440, "rate_pct": extra / 4.4},
                         "overall": {"success": old + extra, "total": 640,
                                     "rate_pct": (old + extra) / 6.4,
                                     "object_cluster_95ci_pct": [engine.shared.metrics.quantile(boots, .025),
                                                                  engine.shared.metrics.quantile(boots, .975)]}}
    delta = [(records[obj]["native_success_count"] - records[obj]["raw_success_count"]) / 64 * 100
             for obj in OBJECTS]
    boots = [sum(delta[i] for i in draw) / 10 for draw in draws]
    report = {"line": line, "method": METHOD, "source_sha256": source_hash,
              "coupling_sha256": sha(COUPLING_SPECS[engine.LINES[line][0]]),
              "battery": BATTERY, "validator_projection": "lsq",
              "verify_batch_size": BATCH, "tail_padded": True, "revalidation_count": 1,
              "source_definition": {"raw": "clip original active joints, then forward-couple passive joints",
                                    "native": "bounded LSQ fit to all original finger joints, then forward-couple"},
              "raw": results["raw"], "native": results["native"],
              "native_minus_raw_pp": sum(delta) / 10,
              "native_minus_raw_object_cluster_95ci_pp": [engine.shared.metrics.quantile(boots, .025),
                                                              engine.shared.metrics.quantile(boots, .975)],
              "per_object": {obj: {"raw_success": records[obj]["raw_success"],
                                   "native_success": records[obj]["native_success"],
                                   "raw_count": records[obj]["raw_success_count"],
                                   "native_count": records[obj]["native_success_count"],
                                   "raw_only": sum(a and not b for a, b in zip(
                                       records[obj]["raw_success"], records[obj]["native_success"])),
                                   "native_only": sum(b and not a for a, b in zip(
                                       records[obj]["raw_success"], records[obj]["native_success"]))}
                             for obj in OBJECTS}}
    dest = OUT / line / "summary.json"
    if dest.exists():
        prior = json.loads(dest.read_text())
        if (prior.get("source_sha256") != source_hash
                or prior.get("coupling_sha256") != report["coupling_sha256"]
                or prior.get("method") != METHOD):
            raise ValueError(f"Refusing to overwrite unrelated summary: {dest}")
    atomic_json(dest, report)
    print(f"{line}: Raw {results['raw']['overall']['success']}/640 = "
          f"{results['raw']['overall']['rate_pct']:.2f}%; "
          f"Native {results['native']['overall']['success']}/640 = "
          f"{results['native']['overall']['rate_pct']:.2f}% | {dest}", flush=True)
    return report


def table() -> None:
    """Add Raw/Native columns to a six-line source/search comparison table."""
    source_roots = {"Sample-288": ROOT / "results/supplement_sample288_six_b64",
                    "CEM-288": ROOT / "results/supplement_cem288_six_independent_b64",
                    "PD-9": ROOT / "results/supplement_pd288_five_b64"}
    headers = ["line", "Raw", "Native", "Sample-288 top1", "Sample-288 multi",
               "CEM-288 top1", "CEM-288 multi", "PD-9 top1", "PD-9 multi"]
    rows, details = [], {}
    for line in LINES:
        raw_native = json.loads((OUT / line / "summary.json").read_text())
        source_hash = source_sha(line)
        if (raw_native.get("source_sha256") != source_hash
                or raw_native.get("method") != METHOD or raw_native.get("battery") != BATTERY
                or raw_native.get("validator_projection") != "lsq"
                or any(raw_native[mode]["overall"]["total"] != 640 for mode in MODES)):
            raise ValueError(f"Wrong raw/native source or validation protocol: {line}")
        values = {"line": line,
                  "Raw": raw_native["raw"]["overall"]["rate_pct"],
                  "Native": raw_native["native"]["overall"]["rate_pct"]}
        details[line] = {"source_sha256": source_hash, "Raw": raw_native["raw"],
                         "Native": raw_native["native"]}
        for method, root in source_roots.items():
            path = (ROOT / "results/supplement_svh_dro_pd288_b64/summary.json"
                    if method == "PD-9" and line == "svh-dro" else root / line / "summary.json")
            if not path.is_file():
                continue
            saved = json.loads(path.read_text())
            if (saved.get("source_sha256") != source_hash or saved.get("battery") != BATTERY
                    or any(saved[metric]["overall"]["total"] != 640 for metric in ("top1", "multi"))):
                raise ValueError(f"Cannot join different source populations/batteries: {path}")
            for metric in ("top1", "multi"):
                values[f"{method} {metric}"] = saved[metric]["overall"]["rate_pct"]
        rows.append(values)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "six_line_table.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    def fmt(row, col):
        return "—" if col not in row else f"{row[col]:.2f}%"
    lines = ["# Six source lines: original-grasp Raw/Native and search recovery",
             "", "Raw = original active joints → clipped Φ; Native = all-joint bounded LSQ → Φ.",
             "Both direct columns use LSQ controller targets on all three hands, one attempt per source,",
             "the standard six-force battery, CPU-state batch32 and 0.02 m threshold.",
             "Older Ability/SVH search columns were validated under their historical controller mode;",
             "do not interpret cross-column differences as an isolated source projection effect.",
             "", "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join([row["line"]] + [fmt(row, col) for col in headers[1:]]) + " |")
    (OUT / "six_line_table.md").write_text("\n".join(lines) + "\n")
    atomic_json(OUT / "six_line_table.json", {"method": METHOD,
                                              "battery": BATTERY, "rows": rows, "sources": details})
    print(f"saved six-line table: {csv_path} and {OUT / 'six_line_table.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "recheck", "summarize", "table", "all", "worker"))
    parser.add_argument("--line", choices=(*LINES, "all"), default="all")
    parser.add_argument("--object", choices=OBJECTS, help="One-object pilot for prepare/recheck")
    parser.add_argument("--max-worker-groups", type=int,
                        help="Pause after this many fresh per-object CPU processes")
    parser.add_argument("--cache-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "worker":
        if args.line == "all" or args.object is None or not args.cache_sha:
            parser.error("worker requires one --line/--object and the cache SHA")
        worker(args.line, args.object, args.cache_sha)
        return
    if args.cache_sha is not None:
        parser.error("--cache-sha applies only to worker")
    if args.stage == "table":
        if args.object or args.max_worker_groups:
            parser.error("table only consumes completed six-line summaries")
        table()
        return
    if args.object and args.stage not in ("prepare", "recheck", "all"):
        parser.error("--object requires prepare/recheck/all")
    if args.max_worker_groups is not None and (
            args.stage not in ("recheck", "all") or args.max_worker_groups < 1):
        parser.error("--max-worker-groups must be positive and requires recheck/all")
    selected = LINES if args.line == "all" else (args.line,)
    objects = OBJECTS if args.object is None else (args.object,)
    for line in selected:
        if args.stage == "check":
            source_sha(line)
        elif args.stage == "prepare":
            input_sha = source_sha(line)
            for obj in objects:
                prepare(line, obj, input_sha)
        elif args.stage == "recheck":
            recheck(line, objects, args.max_worker_groups)
        elif args.stage == "summarize":
            summary(line)
        else:
            recheck(line, objects, args.max_worker_groups)
            if not args.object and args.max_worker_groups is None:
                summary(line)
    if args.stage == "all" and args.line == "all" and args.max_worker_groups is None and not args.object:
        table()


if __name__ == "__main__":
    main()
