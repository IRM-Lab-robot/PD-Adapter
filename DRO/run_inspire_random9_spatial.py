#!/usr/bin/env python3
"""Nine-branch, wide-position nonlearned search for Inspire-DRO/GenDex.

Every branch receives P32×G1 proposals split 11 local / 11 at 3–6 cm /
10 at 6–12 cm from its branch mean. Eight independently sampled random
branch means also span 3–6 cm and 6–12 cm from the projected source wrist.
All 9×32=288 GPU evaluations per source are fresh; no learned model is used.
One new batch32 CPU-state recheck supplies GPU-ranked top-1 and any-candidate
recovery. First-object pilot: `all --line inspire-dro --object contactdb+banana`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DRO import run_pd288_five as pd

spec = importlib.util.spec_from_file_location("random9_spatial_recheck_engine",
                                             ROOT / "DRO/run_sample288_six.py")
if spec is None or spec.loader is None:
    raise ImportError("DRO/run_sample288_six.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

OUT = ROOT / "results/supplement_inspire_random9_spatial_b64"
LINES = ("inspire-dro", "inspire-gen")
OBJECTS = engine.OBJECTS
ARMS = ("default",) + tuple(f"slot{i}" for i in range(8))
BATTERY = engine.BATTERY
P = 32
PART = 8
SEED = 20263044  # frozen generator-independent random-start seed
engine.OUT = OUT
engine.METHOD = "Random-9-Spatial/288"
engine.WORKER_ENTRYPOINT = Path(__file__).resolve()
engine.collect = lambda line, obj, source_sha: collect(line, obj, source_sha)
WARM_OUT = OUT
WARM_METHOD = engine.METHOD
SPATIAL_MODE = "stratified"
SEARCH_TAG = "random9_spatial"


def safe(obj: str) -> str:
    return engine.shared.metrics.safe(obj)


def source_sha(line: str) -> str:
    return engine.shared.check_source(line)


def warm_path(line: str, slot: int) -> Path:
    return WARM_OUT / line / "warmstarts" / f"slot{slot}.pt"


def manifest_path(line: str) -> Path:
    return WARM_OUT / line / "warmstarts/manifest.json"


def search_path(line: str, obj: str, arm: str, part: int) -> Path:
    return (OUT / line / "search" / arm / safe(obj)
            / f"part{part:02d}_s{part*PART:02d}_e{part*PART+PART-1:02d}.pt")


def arm_seed(arm: str, part: int) -> int:
    return 10001 + 1009 * part + 100003 * ARMS.index(arm)


def prepare(line: str) -> None:
    import numpy as np
    import torch

    sha = source_sha(line)
    manifest = manifest_path(line)
    if manifest.exists():
        check_warm(line)
        print(f"reuse {line}: eight random-position warm starts × 640", flush=True)
        return
    sys.path.insert(0, str(ROOT / "src/eval"))
    import generate_pi_phi_warmstart_step12 as projection

    hand = projection.create_hand_model("inspire", device=torch.device("cpu"))
    fingers = hand.get_joint_orders()[6:]
    lower, upper = hand.pk_chain.get_joint_limits()
    lower, upper = np.asarray(lower[6:]), np.asarray(upper[6:])
    active = np.asarray([fingers.index(name) for name in projection.inspire_coupling.active_joints])
    by_slot = {i: [] for i in range(8)}
    for source in pd.sources.source_rows(pd.source_file(line), "inspire", 64):
        obj = source["object_name"]
        projected, command = projection.lsq_project_q_local(
            torch.as_tensor(source["q_dro_order"], dtype=torch.float32), fingers, lower, upper)
        for sid in range(64):
            q = projected[sid].float()
            u = command[sid].numpy()
            # The ID determines the RNG stream; changing iteration order cannot
            # change any previously generated source grasp.
            key = f"{line}|{obj}|{sid}".encode()
            rng = np.random.default_rng(SEED + int(hashlib.sha1(key).hexdigest()[:8], 16))
            for slot in range(8):
                band = 0 if slot < 4 else 1
                lo, hi = ((.03, .06), (.06, .12))[band]
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                radius = rng.uniform(lo ** 3, hi ** 3) ** (1 / 3)
                wrist = q[:6].numpy().copy()
                wrist[:3] += direction * radius
                wrist[3:6] += rng.normal(0, .20 if band == 0 else .30, size=3)
                proposed_u = rng.normal(u, .25).astype(np.float32)
                clipped_u = np.clip(proposed_u, lower[active], upper[active])
                by_slot[slot].append({
                    "object_name": obj, "source_grasp_index": sid,
                    "generator": "DRO" if line.endswith("dro") else "GenDexGrasp",
                    "slot_id": slot, "source_q_seed": q.clone(),
                    "params": torch.tensor(np.concatenate((wrist, clipped_u)), dtype=torch.float32),
                    "source_wrist_distance_m": float(radius),
                    "initial_command_clipped": bool(np.any(proposed_u != clipped_u)),
                    "start_band": "3-6cm" if band == 0 else "6-12cm"})
    hashes = []
    for slot, rows in by_slot.items():
        if len(rows) != 640 or {(r["object_name"], r["source_grasp_index"]) for r in rows} != {
                (obj, sid) for obj in OBJECTS for sid in range(64)}:
            raise ValueError(f"Missing warm-start IDs in {line}/slot{slot}")
        path = warm_path(line, slot)
        payload = {"info": {"line": line, "slot_id": slot, "source_sha256": sha,
                            "method": WARM_METHOD, "random_seed": SEED,
                            "start_shell_m": [(.03, .06), (.06, .12)][slot // 4]}, "seeds": rows}
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            if (existing["info"] != payload["info"] or len(existing["seeds"]) != len(rows)
                    or any(not torch.equal(a["params"], b["params"])
                           for a, b in zip(existing["seeds"], rows))):
                raise ValueError(f"Conflicting partial random warm-start: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".pt.tmp")
            torch.save(payload, temp)
            temp.replace(path)
        hashes.append(engine.sha(path))
    engine.atomic_json(manifest, {"line": line, "method": WARM_METHOD,
                                 "source_sha256": sha, "seed": SEED,
                                 "warm_sha256": hashes, "slots": 8, "sources_per_slot": 640,
                                 "start_shells_m": {"slot0-3": [.03, .06], "slot4-7": [.06, .12]},
                                 "search": "nine branches, each 11 local + 11 mid + 10 far"})
    check_warm(line)
    print(f"prepared {line}: 8×640 independently sampled nonlearned starts", flush=True)


def check_warm(line: str) -> str:
    import torch

    sha = source_sha(line)
    metadata = json.loads(manifest_path(line).read_text())
    if (metadata.get("line") != line or metadata.get("method") != WARM_METHOD
            or metadata.get("source_sha256") != sha or metadata.get("seed") != SEED
            or metadata.get("slots") != 8 or metadata.get("sources_per_slot") != 640):
        raise ValueError(f"Wrong random-start manifest: {line}")
    for slot in range(8):
        path = warm_path(line, slot)
        if not path.is_file() or engine.sha(path) != metadata["warm_sha256"][slot]:
            raise ValueError(f"Missing/changed random slot: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        rows = data["seeds"]
        if (data["info"].get("source_sha256") != sha or data["info"].get("slot_id") != slot
                or len(rows) != 640 or {(r["object_name"], int(r["source_grasp_index"])) for r in rows}
                != {(obj, sid) for obj in OBJECTS for sid in range(64)}):
            raise ValueError(f"Incomplete random warm starts: {path}")
        lo, hi = ((.03, .06), (.06, .12))[slot // 4]
        for row in rows:
            q = torch.as_tensor(row["source_q_seed"])
            p = torch.as_tensor(row["params"])
            actual = float(torch.linalg.vector_norm(p[:3] - q[:3]))
            if (tuple(q.shape) != (18,) or tuple(p.shape) != (12,)
                    or not lo - 1e-6 <= actual <= hi + 1e-6):
                raise ValueError(f"Random start outside its declared shell: {path}")
    return sha


def source_states(line: str, obj: str, part: int):
    import torch

    seeds = torch.load(warm_path(line, 0), map_location="cpu", weights_only=False)["seeds"]
    by_key = {(r["object_name"], int(r["source_grasp_index"])): r for r in seeds}
    return torch.stack([torch.as_tensor(by_key[(obj, sid)]["source_q_seed"])
                        for sid in range(part * PART, (part + 1) * PART)])


def check_search(line: str, obj: str, arm: str, part: int):
    sys.path.insert(0, str(ROOT / "eval"))
    import torch
    from spatial_sampling import LAYERS, layer_slices

    path = search_path(line, obj, arm, part)
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    ids = list(range(part * PART, (part + 1) * PART))
    expected = {"zero_file": str(pd.source_file(line)), "max_grasps": 64,
                "selection_battery": BATTERY, "source_indices": ids,
                "population": P, "generations": 1, "elites": 8, "restarts": 1,
                "batch_source_grasps": PART, "keep_per_grasp": 3,
                "keep_per_object": PART * 3, "cpu_final_candidates": PART * 3,
                "search_pipeline": "gpu", "reuse_gpu_sim": True, "log_rejected": True,
                "allow_non_train": True, "spatial_sampling": SPATIAL_MODE,
                "seed": arm_seed(arm, part), "arm_id": f"{SEARCH_TAG}_{line}_{arm}",
                "warm_start_file": None if arm == "default" else str(warm_path(line, int(arm[4:])))}
    if any(artifact["args"].get(k) != v for k, v in expected.items()):
        raise ValueError(f"Wrong wide Random-9 search settings: {path}")
    if artifact.get("objects") != [obj] or artifact.get("total_source_grasps") != PART:
        raise ValueError(f"Incomplete wide Random-9 search: {path}")
    row = artifact["rows"][0]
    if ([int(x["grasp_index"]) for x in row["grasp_rows"]] != ids
            or not torch.equal(torch.as_tensor(row["q_seed"]), source_states(line, obj, part))):
        raise ValueError(f"Wide Random-9 source mismatch: {path}")
    slices = layer_slices(P)
    for grasp in row["grasp_rows"]:
        if len(grasp["history"]) != 1:
            raise ValueError(f"Missing search history: {path}")
        history = grasp["history"][0]
        if SPATIAL_MODE == "local":
            if history["pass_count"] != grasp["raw_num_pass"]:
                raise ValueError(f"Incomplete local GPU pass history: {path}")
            continue
        mask = torch.as_tensor(history.get("gpu_success_mask"))
        raw = torch.as_tensor(history.get("sampled_params"))
        if (tuple(history.get("spatial_layers", ())) != LAYERS
                or mask.shape != (P,) or raw.shape != (P, 12)
                or sum(history["spatial_layer_pass_counts"]) != int(mask.sum())
                or any(history["spatial_layer_pass_counts"][i] != int(mask[band].sum())
                       for i, band in enumerate(slices))):
            raise ValueError(f"Full spatial sampling record is missing: {path}")
        center = raw[0, :3]
        distances = torch.linalg.vector_norm(raw[:, :3] - center, dim=1)
        if (not torch.all((distances[slices[1]] >= .03 - 1e-6)
                          & (distances[slices[1]] <= .06 + 1e-6))
                or not torch.all((distances[slices[2]] >= .06 - 1e-6)
                              & (distances[slices[2]] <= .12 + 1e-6))):
            raise ValueError(f"Spatial proposal shell mismatch: {path}")
    pd.check_candidate_row(path, row, set(ids), PART * 3, "inspire")
    return row


def search(line: str, objects, arms, chosen_part=None) -> None:
    check_warm(line)
    driver = ROOT / "eval/generate_inspire_train_bsearch.py"
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    for obj in objects:
        for arm in arms:
            for part in ((chosen_part,) if chosen_part is not None else range(8)):
                path = search_path(line, obj, arm, part)
                if path.exists():
                    check_search(line, obj, arm, part)
                    print(f"reuse {line}/{obj}/{arm}/part{part}", flush=True)
                    continue
                ids = list(range(part * PART, (part + 1) * PART))
                path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(driver), "--zero-file", str(pd.source_file(line)),
                       "--objects", obj, "--allow-non-train", "--max-grasps", "64",
                       "--selection-battery", BATTERY, "--population", str(P),
                       "--elites", "8", "--generations", "1", "--restarts", "1",
                       "--batch-source-grasps", str(PART), "--keep-per-grasp", "3",
                       "--keep-per-object", str(PART * 3), "--cpu-final-candidates", str(PART * 3),
                       "--search-pipeline", "gpu", "--reuse-gpu-sim", "--log-rejected",
                       "--spatial-sampling", SPATIAL_MODE, "--seed", str(arm_seed(arm, part)),
                       "--source-indices", *(str(i) for i in ids),
                       "--arm-id", f"{SEARCH_TAG}_{line}_{arm}", "--out", str(path)]
                if arm != "default":
                    cmd.extend(("--warm-start-file", str(warm_path(line, int(arm[4:])))))
                with path.with_suffix(".log").open("w") as stream:
                    process = subprocess.run(cmd, cwd=ROOT, env=env,
                                             stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode or not path.is_file():
                    raise RuntimeError(f"Search failed: inspect {path.with_suffix('.log')}")
                check_search(line, obj, arm, part)
                print(f"searched {line}/{obj}/{arm}/part{part}", flush=True)


def collect(line: str, obj: str, sha: str):
    import torch

    if check_warm(line) != sha:
        raise ValueError("Input source changed during search")
    candidates, hashes = [], []
    for arm in ARMS:
        for part in range(8):
            row = check_search(line, obj, arm, part)
            path = search_path(line, obj, arm, part)
            hashes.append(engine.sha(path))
            for group in ("passing_candidates", "cpu_final_rejected_candidates"):
                for idx, cand in enumerate(row[group]):
                    params = torch.as_tensor(cand["params"], dtype=torch.float32).contiguous()
                    candidates.append({"id": f"{arm}/part{part:02d}/{group}/{idx}",
                                       "source_index": int(cand["source_grasp_index"]),
                                       "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(18),
                                       "gpu_rank_score": float(cand["gpu_rank_score"]),
                                       "search_params_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
                                       "search_cpu_success": bool(cand["cpu_success"])})
    if len(hashes) != 72:
        raise ValueError("Incomplete nine-branch search pool")
    return candidates, hashes


def distance_band(distance: float) -> str:
    if distance < .03:
        return "<3cm"
    if distance < .06:
        return "3-6cm"
    if distance < .12:
        return "6-12cm"
    return ">=12cm"


def audit(line: str, obj: str, record: dict) -> dict:
    """Reconstruct actual source-relative distances for every physical sample."""
    sys.path.insert(0, str(ROOT / "eval"))
    import torch
    from spatial_sampling import LAYERS, layer_slices

    bands = ("<3cm", "3-6cm", "6-12cm", ">=12cm")
    stats = {name: {"samples": 0, "gpu_pass": 0, "rechecked": 0, "cpu_success": 0}
             for name in bands}
    by_candidate_id = {}
    center_stats = {"3-6cm": 0, "6-12cm": 0}
    clipped = 0
    slices = layer_slices(P)
    for arm in ARMS:
        for part in range(8):
            row = check_search(line, obj, arm, part)
            labels = {}
            for grasp in row["grasp_rows"]:
                sid = int(grasp["grasp_index"])
                q = torch.as_tensor(grasp["q_seed"])
                history = grasp["history"][0]
                raw = torch.as_tensor(history["sampled_params"])
                success = torch.as_tensor(history["gpu_success_mask"])
                clipped += history["clamp_sample_count"]
                if arm != "default":
                    center = distance_band(float(torch.linalg.vector_norm(raw[0, :3] - q[:3])))
                    if center not in center_stats:
                        raise ValueError(f"Warm-start shell mismatch: {arm}/{obj}/{sid}")
                    center_stats[center] += 1
                for i in range(P):
                    band = distance_band(float(torch.linalg.vector_norm(raw[i, :3] - q[:3])))
                    stats[band]["samples"] += 1
                    stats[band]["gpu_pass"] += bool(success[i])
                    if success[i]:
                        labels[(sid, raw[i].numpy().tobytes())] = band
                if sum(int(success[b].sum()) for b in slices) != int(success.sum()):
                    raise ValueError("Incomplete spatial GPU mask")
            for group in ("passing_candidates", "cpu_final_rejected_candidates"):
                for idx, cand in enumerate(row[group]):
                    key = (int(cand["source_grasp_index"]),
                           torch.as_tensor(cand["params"], dtype=torch.float32).numpy().tobytes())
                    if key not in labels:
                        raise ValueError(f"Retained candidate not in raw GPU pass mask: {arm}/{obj}")
                    by_candidate_id[f"{arm}/part{part:02d}/{group}/{idx}"] = labels[key]
    for cid, passed in zip(record["candidate_ids"], record["candidate_success"]):
        band = by_candidate_id[cid]
        stats[band]["rechecked"] += 1
        stats[band]["cpu_success"] += bool(passed)
    if sum(item["samples"] for item in stats.values()) != 64 * 288:
        raise ValueError("Expected 64×288 actual physical samples")
    return {"distance_from_projected_source": stats,
            "warm_start_centers": center_stats,
            "command_clipped_samples": clipped,
            "command_clipped_fraction": clipped / (64 * 288),
            "per_branch_proposal_counts": [b.stop - b.start for b in slices]}


def audit_local(line: str, obj: str, record: dict) -> dict:
    """Local search stores pass counts, but not every failed raw proposal."""
    import torch

    bands = ("<3cm", "3-6cm", "6-12cm", ">=12cm")
    kept = {band: {"rechecked": 0, "cpu_success": 0} for band in bands}
    by_id = {}
    gpu_pass = clipped = sampled = 0
    for arm in ARMS:
        for part in range(8):
            pool = check_search(line, obj, arm, part)
            starts = {int(grasp["grasp_index"]): torch.as_tensor(grasp["q_seed"])
                      for grasp in pool["grasp_rows"]}
            for grasp in pool["grasp_rows"]:
                history = grasp["history"][0]
                gpu_pass += int(grasp["raw_num_pass"])
                clipped += int(history["clamp_sample_count"])
                sampled += int(history["clamp_sample_total"])
            for group in ("passing_candidates", "cpu_final_rejected_candidates"):
                for idx, cand in enumerate(pool[group]):
                    sid = int(cand["source_grasp_index"])
                    delta = torch.as_tensor(cand["params"])[:3] - starts[sid][:3]
                    by_id[f"{arm}/part{part:02d}/{group}/{idx}"] = distance_band(
                        float(torch.linalg.vector_norm(delta)))
    if sampled != 64 * 288:
        raise ValueError("Local search did not evaluate all nine P32 branches")
    for cid, passed in zip(record["candidate_ids"], record["candidate_success"]):
        band = by_id[cid]
        kept[band]["rechecked"] += 1
        kept[band]["cpu_success"] += bool(passed)
    return {"raw_failed_proposal_positions_saved": False,
            "distance_from_projected_source_for_retained_candidates": kept,
            "gpu_pass": gpu_pass, "samples": sampled,
            "command_clipped_samples": clipped,
            "command_clipped_fraction": clipped / sampled}


def report(line: str, obj: str) -> dict:
    path = engine.result_path(line, obj)
    row = json.loads(path.read_text())
    sha = check_warm(line)
    candidates, hashes = collect(line, obj, sha)
    selected = engine.shared.metrics.freeze_top1(candidates)
    if (row.get("line") != line or row.get("object_name") != obj
            or row.get("method") != engine.METHOD or row.get("source_sha256") != sha
            or row.get("search_part_sha256") != hashes or row.get("battery") != BATTERY
            or row.get("validator_projection") != "lsq"
            or row.get("gpu_evaluations_per_source") != 288
            or row.get("revalidation_count") != 1 or row.get("verify_batch_size") != 32
            or row.get("tail_padded") is not True
            or row.get("candidate_ids") != [c["id"] for c in candidates]
            or row.get("top1_candidate_indices") != selected
            or row.get("top1_candidate_ids")
            != [candidates[i]["id"] if i is not None else None for i in selected]
            or len(row.get("candidate_success", [])) != len(candidates)):
        raise ValueError(f"Incomplete or wrong Random-9 recheck: {path}")
    top1, multi = engine.shared.metrics.outcomes(candidates, row["candidate_success"], selected)
    if row.get("top1_success") != top1 or row.get("multi_success") != multi:
        raise ValueError(f"Candidate labels and target results disagree: {path}")
    baseline = pd.OUT / line / "revalidation_once" / f"{safe(obj)}.json"
    old = json.loads(baseline.read_text())
    if (old.get("object_name") != obj or old.get("line") != line
            or old.get("source_sha256") != sha or old.get("battery") != BATTERY
            or old.get("verify_batch_size") != 32 or old.get("tail_padded") is not True):
        raise ValueError(f"Wrong reference PD-9 result: {baseline}")
    comparisons = {}
    for name, labels in (("top1", top1), ("multi", multi)):
        previous = old[f"{name}_success"]
        if len(previous) != 64:
            raise ValueError(f"Reference result lacks 64 source targets: {baseline}")
        comparisons[name] = {"random9": sum(labels), "reference_pd9": sum(previous),
                             "total": 64, "random9_old20": sum(labels[:20]),
                             "random9_new44": sum(labels[20:]),
                             "gained_vs_pd9": sum(not a and b for a, b in zip(previous, labels)),
                             "lost_vs_pd9": sum(a and not b for a, b in zip(previous, labels)),
                             "source_success": labels}
    result = {"scope": f"one-object {SPATIAL_MODE} Random-9 sensitivity check",
              "comparison_note": "PD-9 uses historical local search; differences include both initialization and search range",
              "line": line, "object_name": obj, "method": engine.METHOD,
              "source_sha256": sha, "battery": BATTERY, "validator_projection": "lsq",
              "gpu_evaluations_per_source": 288, "candidate_count": len(candidates),
              "recheck_sha256": engine.sha(path), "reference_pd9_sha256": engine.sha(baseline),
              "spatial_audit": audit(line, obj, row) if SPATIAL_MODE == "stratified"
                               else audit_local(line, obj, row), "metrics": comparisons}
    if SPATIAL_MODE == "local":
        wide_path = WARM_OUT / line / "revalidation_once" / f"{safe(obj)}.json"
    if SPATIAL_MODE == "local" and wide_path.is_file():
        wide = json.loads(wide_path.read_text())
        if (wide.get("source_sha256") != sha or wide.get("line") != line
                or wide.get("object_name") != obj or wide.get("battery") != BATTERY
                or wide.get("method") != WARM_METHOD or wide.get("validator_projection") != "lsq"):
            raise ValueError(f"Wrong same-start wide Random-9 control: {wide_path}")
        comparisons = {}
        for key, local_labels in (("top1", top1), ("multi", multi)):
            wide_labels = wide[f"{key}_success"]
            if len(wide_labels) != 64:
                raise ValueError(f"Incomplete wide Random-9 labels: {wide_path}")
            comparisons[key] = {"local": sum(local_labels), "wide": sum(wide_labels),
                                "gained_by_wide": sum(not a and b for a, b in zip(local_labels, wide_labels)),
                                "lost_by_wide": sum(a and not b for a, b in zip(local_labels, wide_labels))}
        result["same_random_starts_comparison"] = {
            "wide_recheck_sha256": engine.sha(wide_path),
            "warm_manifest_sha256": engine.sha(manifest_path(line)),
            "scope": "same random initializations; only per-branch sampling radius differs",
            "metrics": comparisons}
    dest = OUT / line / "reports" / f"{safe(obj)}.json"
    engine.atomic_json(dest, result)
    for name, value in result["metrics"].items():
        print(f"{line}/{obj} {name}: {SPATIAL_MODE} Random-9 {value['random9']}/64, "
              f"historical local PD-9 {value['reference_pd9']}/64", flush=True)
        if "same_random_starts_comparison" in result:
            pair = result["same_random_starts_comparison"]["metrics"][name]
            print(f"same random starts, {name}: local {pair['local']}/64, "
                  f"wide {pair['wide']}/64 (wide gain {pair['gained_by_wide']}, "
                  f"loss {pair['lost_by_wide']})", flush=True)
    print(f"saved {dest}", flush=True)
    return result


def summarize(line: str) -> dict:
    """Summarize the complete 10×64 line, independently of any wide control."""
    missing = [obj for obj in OBJECTS if not engine.result_path(line, obj).is_file()]
    if missing:
        raise FileNotFoundError(f"Recheck all ten objects before summarizing {line}: {missing}")
    reports = [report(line, obj) for obj in OBJECTS]
    source = check_warm(line)
    if any(row["source_sha256"] != source for row in reports):
        raise ValueError(f"Mixed source manifests in {line} summary")
    rng = random.Random(20261221)
    draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    metrics = {}
    for name in ("top1", "multi"):
        values = [row["metrics"][name] for row in reports]
        rates = [value["random9"] * 100 / 64 for value in values]
        differences = [(value["random9"] - value["reference_pd9"]) * 100 / 64
                       for value in values]
        boot = [sum(rates[i] for i in draw) / 10 for draw in draws]
        paired = [sum(differences[i] for i in draw) / 10 for draw in draws]
        old20 = sum(value["random9_old20"] for value in values)
        new44 = sum(value["random9_new44"] for value in values)
        metrics[name] = {
            "old20": {"success": old20, "total": 200, "rate_pct": old20 / 2},
            "new44": {"success": new44, "total": 440, "rate_pct": new44 / 4.4},
            "overall": {"success": old20 + new44, "total": 640,
                        "rate_pct": (old20 + new44) / 6.4,
                        "object_cluster_95ci_pct": [engine.shared.metrics.quantile(boot, .025),
                                                    engine.shared.metrics.quantile(boot, .975)]},
            "vs_local_pd9": {"reference_success": sum(v["reference_pd9"] for v in values),
                             "delta_pp": sum(differences) / 10,
                             "object_cluster_95ci_pp": [engine.shared.metrics.quantile(paired, .025),
                                                          engine.shared.metrics.quantile(paired, .975)]}}
    report_row = {"line": line, "method": engine.METHOD,
                  "scope": "10 held-out objects × 64 source IDs per line",
                  "sampling": SPATIAL_MODE, "source_sha256": source,
                  "warm_manifest_sha256": engine.sha(manifest_path(line)),
                  "battery": BATTERY, "validator_projection": "lsq",
                  "gpu_evaluations_per_source": 288, "revalidation_count": 1,
                  "verify_batch_size": 32, "tail_padded": True,
                  "comparison_note": "Local PD-9 is a descriptive reference; its learned initializations differ",
                  "top1": metrics["top1"], "multi": metrics["multi"],
                  "per_object": {row["object_name"]: row for row in reports}}
    destination = OUT / line / "summary.json"
    engine.atomic_json(destination, report_row)
    for name in ("top1", "multi"):
        value = metrics[name]
        print(f"{line} {SPATIAL_MODE} Random-9 {name}: "
              f"{value['overall']['success']}/640 = {value['overall']['rate_pct']:.2f}% "
              f"(old20 {value['old20']['success']}/200, "
              f"new44 {value['new44']['success']}/440)", flush=True)
    print(f"saved {destination}", flush=True)
    return report_row


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        engine.main()  # Fresh Isaac Gym import precedes torch in the CPU worker.
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "search", "recheck", "report", "summarize", "all"))
    parser.add_argument("--line", choices=(*LINES, "all"), required=True)
    parser.add_argument("--object", choices=OBJECTS,
                        help="One object (default: first test object, contactdb+banana)")
    parser.add_argument("--all-objects", action="store_true",
                        help="Search/recheck all 10 objects and write a full-line summary with `all`")
    parser.add_argument("--arm", choices=ARMS, help="Search one of the nine branches")
    parser.add_argument("--part", type=int, choices=range(8), help="Search eight source IDs")
    parser.add_argument("--max-worker-groups", type=int, help="Pause CPU recheck after N worker groups")
    args = parser.parse_args()
    if args.object is not None and args.all_objects:
        parser.error("Choose either --object or --all-objects")
    if args.part is not None and (args.stage != "search" or args.line == "all" or args.all_objects):
        parser.error("--part requires search, a single --line and one object")
    if args.arm is not None and args.stage != "search":
        parser.error("--arm is for search only")
    if args.max_worker_groups is not None and (
            args.stage not in ("recheck", "all") or args.max_worker_groups < 1):
        parser.error("--max-worker-groups requires recheck/all and a positive number")
    if args.all_objects and args.stage in ("prepare", "check", "report", "summarize"):
        parser.error("--all-objects applies to search/recheck/all; summarize always requires all ten")
    if args.object is not None and args.stage in ("prepare", "check", "summarize"):
        parser.error("--object applies to search/recheck/report/all only")
    objects = OBJECTS if args.all_objects else (args.object or OBJECTS[0],)
    for line in (LINES if args.line == "all" else (args.line,)):
        if args.stage == "prepare":
            prepare(line)
        elif args.stage == "check":
            check_warm(line)
        elif args.stage == "search":
            search(line, objects, (args.arm,) if args.arm else ARMS, args.part)
        elif args.stage == "recheck":
            engine.recheck(line, objects, args.max_worker_groups)
        elif args.stage == "report":
            report(line, objects[0])
        elif args.stage == "summarize":
            summarize(line)
        else:
            prepare(line)
            search(line, objects, ARMS)
            engine.recheck(line, objects, args.max_worker_groups)
            if args.max_worker_groups is None:
                if args.all_objects:
                    summarize(line)
                else:
                    report(line, objects[0])


if __name__ == "__main__":
    main()
