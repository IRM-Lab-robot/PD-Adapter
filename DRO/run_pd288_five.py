#!/usr/bin/env python3
"""Five six-force PD-9 lines on 10×64: reuse old20 searches, search new44.

Each source receives default + eight learned branches × 32 GPU evaluations.
A fixed GPU-score top-1 and candidate-set rate share one CPU-state recheck.
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
from DRO import generate_64 as sources
from DRO import run_svh_cem288 as metrics

OUT = ROOT / "results/supplement_pd288_five_b64"
OBJECTS = metrics.OBJECTS
BATTERY = metrics.BATTERY
ARMS = ("default",) + tuple(f"head{i}" for i in range(8))
DIMS = sources.DIMS
PARAM_DIMS = {"inspire": 12, "ability": 12, "svh": 15}
PART_SIZE = 8
VERIFY_BATCH = 32
WORKER_BATCHES = 4
LINES = {
    "inspire-dro": ("inspire", "DRO", "dro", ROOT / "eval/original_test10_zero_shot_b20.pt"),
    "inspire-gen": ("inspire", "GEN", "gendex", ROOT / "eval/gendex_inspire_mainset_grasps.pt"),
    "svh-gen": ("svh", "GEN", "gendex", ROOT / "results/pd_refiner_svh/gendex_test10/svh_gendex_test10_grasps.pt"),
    "ability-dro": ("ability", "DRO", "dro", ROOT / "results/pd_refiner_ability/ability_test10_zero_shot_b20.pt"),
    "ability-gen": ("ability", "GEN", "gendex", ROOT / "results/pd_refiner_ability/gendex_test10/ability_gendex_test10_grasps.pt"),
}


def hand(line):
    return LINES[line][0]


def model_path(line):
    h = hand(line)
    if h == "inspire":
        return ROOT / "results/supplement_inspire_standard/train/best_pi_phi_v5_multihead.pt"
    return ROOT / f"results/pd_refiner_{h}/pi_phi_mixed_dro_gendex/best_{h}_pi_phi_multihead.pt"


def source_file(line):
    h, gen, _, _ = LINES[line]
    return ROOT / gen / f"{h}.pt"


def old_root(line):
    h, _, g, _ = LINES[line]
    if h == "inspire":
        return ROOT / "results/supplement_inspire_standard/search" / g
    return ROOT / f"results/pd_refiner_{h}/step6_mixed_pi_phi_dual_pareto" / g / "multibasin_arms"


def old_path(line, obj, arm):
    dirname = f"pd_{arm}" if hand(line) == "inspire" and arm != "default" else arm
    return old_root(line) / dirname / f"{metrics.safe(obj)}.pt"


def old_warm(line, head):
    h, _, g, _ = LINES[line]
    if h == "inspire":
        return ROOT / "results/supplement_inspire_standard" / f"pd_{g}_head{head}.pt"
    return (ROOT / f"results/pd_refiner_{h}/step6_mixed_pi_phi_dual_pareto/warmstarts"
            / f"{g}_test10" / f"{h}_mixed_pi_phi_{g}_test10_head{head}.pt")


def new_warm(line, head):
    h, _, g, _ = LINES[line]
    if h == "inspire":
        name = f"pd_{g}_head{head}.pt"
    else:
        name = f"{h}_mixed_pi_phi_{g}_b64_head{head}.pt"
    return OUT / line / "warmstarts" / name


def warm_manifest(line):
    return OUT / line / "warmstarts/manifest.json"


def indices(part):
    if part not in range(6):
        raise ValueError(part)
    return list(range(20 + part * PART_SIZE, min(64, 20 + (part + 1) * PART_SIZE)))


def new_path(line, obj, arm, part):
    ids = indices(part)
    return (OUT / line / "search_new44" / arm / metrics.safe(obj)
            / f"part{part:02d}_s{ids[0]:02d}_e{ids[-1]:02d}.pt")


def result_path(line, obj):
    return OUT / line / "revalidation_once" / f"{metrics.safe(obj)}.json"


def cache_path(line, obj):
    return OUT / line / "recheck_cache" / f"{metrics.safe(obj)}.pt"


def batch_path(line, obj, batch):
    return OUT / line / "revalidation_batches" / metrics.safe(obj) / f"batch{batch:04d}.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def check_source(line):
    import torch

    h, generator, _, oldfile = LINES[line]
    src = source_file(line)
    manifest = src.with_suffix(".json")
    if not src.is_file() or not oldfile.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Missing merged source/manifest/original20: {line}")
    details = json.loads(manifest.read_text())
    if (details.get("generator") != ("DRO" if generator == "DRO" else "GenDexGrasp")
            or details.get("hand") != h or details.get("objects") != list(OBJECTS)
            or details.get("sources_per_object") != 64 or details.get("old_per_object") != 20
            or details.get("added_per_object") != 44
            or details.get("output_sha256") != metrics.sha(src)
            or details.get("old_source_sha256") != metrics.sha(oldfile)):
        raise ValueError(f"Invalid source manifest: {line}")
    complete = sources.source_rows(src, h, 64)
    prior = sources.source_rows(oldfile, h, 20)
    for merged, previous in zip(complete, prior):
        if not torch.equal(torch.as_tensor(merged["q_dro_order"])[:20],
                           torch.as_tensor(previous["q_dro_order"])):
            raise ValueError(f"Old20 grasp order changed for {line}/{merged['object_name']}")
    return details["output_sha256"]


def check_candidate_row(path, row, permitted, cap, h):
    import torch

    n = 0
    for key, label in (("passing_candidates", True), ("cpu_final_rejected_candidates", False)):
        if key not in row:
            raise ValueError(f"Incomplete candidate pool: {path} {key}")
        for cand in row[key]:
            if (int(cand["source_grasp_index"]) not in permitted or bool(cand["cpu_success"]) != label
                    or "gpu_rank_score" not in cand
                    or not math.isfinite(float(cand["gpu_rank_score"]))
                    or torch.as_tensor(cand["q"]).numel() != DIMS[h]
                    or torch.as_tensor(cand["params"]).numel() != PARAM_DIMS[h]):
                raise ValueError(f"Invalid candidate/source/GPU score: {path}")
            n += 1
    if n != int(row["cpu_final_evaluated"]) or n > cap:
        raise ValueError(f"Truncated/incomplete CPU-final candidate pool: {path}")


def check_old(line, obj, arm):
    import torch

    path = old_path(line, obj, arm)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    args = data["args"]
    expected = {"zero_file": str(LINES[line][3]), "selection_battery": BATTERY,
                "max_grasps": 20, "population": 32, "generations": 1,
                "elites": 8, "restarts": 1, "batch_source_grasps": 8,
                "keep_per_grasp": 3, "keep_per_object": 120,
                "cpu_final_candidates": 240, "search_pipeline": "gpu",
                "reuse_gpu_sim": True, "log_rejected": True, "seed": 10001,
                "source_indices": None, "warm_start_file": None if arm == "default" else str(old_warm(line, int(arm[4:])))}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Old20 search setting changed for {path}: {key}={args.get(key)!r}")
    if (data.get("objects") != [obj] or data.get("total_source_grasps") != 20
            or len(data.get("rows", [])) != 1):
        raise ValueError(f"Old20 search is incomplete: {path}")
    row = data["rows"][0]
    if ([int(r["grasp_index"]) for r in row["grasp_rows"]] != list(range(20))
            or torch.as_tensor(row["q_seed"]).shape != (20, DIMS[hand(line)])):
        raise ValueError(f"Old20 source IDs differ: {path}")
    check_candidate_row(path, row, set(range(20)), 60, hand(line))
    return row


def preflight(line):
    import torch

    check_source(line)
    model = model_path(line)
    if not model.is_file():
        raise FileNotFoundError(model)
    if hand(line) == "inspire":
        seed_summary = json.loads((ROOT / "results/supplement_inspire_standard/seed_summary.json").read_text())
        if seed_summary["model_sha256"] != metrics.sha(model):
            raise ValueError("Inspire checkpoint differs from the six-force epoch74 model")
    for head in range(8):
        p = old_warm(line, head)
        if not p.is_file():
            raise FileNotFoundError(p)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        rows = payload["seeds"]
        if (len(rows) != 200 or len({(r["object_name"], int(r["source_grasp_index"])) for r in rows}) != 200
                or {(r["object_name"], int(r["source_grasp_index"])) for r in rows}
                != {(obj, sid) for obj in OBJECTS for sid in range(20)}):
            raise ValueError(f"Incomplete old20 learned proposals: {p}")
        if hand(line) == "inspire":
            if payload["info"].get("model_sha256") != metrics.sha(model):
                raise ValueError(f"Inspire head from a different checkpoint: {p}")
        elif payload["info"].get("model_path") != str(model):
            raise ValueError(f"Mixed hand checkpoint provenance mismatch: {p}")
    n = 0
    for obj in OBJECTS:
        projected = None
        for arm in ARMS:
            row = check_old(line, obj, arm)
            q = torch.as_tensor(row["q_seed"])
            if projected is not None and not torch.equal(q, projected):
                raise ValueError(f"Old20 arm/source mismatch: {line}/{obj}/{arm}")
            projected = q
            n += row["cpu_final_evaluated"]
    print(f"{line}: old20 9 arms complete, {n} recorded CPU-final candidates; model/source checked", flush=True)


def prepare_inspire(line, folder):
    import torch

    sys.path.insert(0, str(ROOT))  # generate_pi_phi_warmstart_step12 imports inspire_coupling
    sys.path.insert(0, str(ROOT / "src/eval"))
    import generate_pi_phi_warmstart_step12 as original

    original.MODEL_PATH = model_path(line)
    model, ckpt = original.load_model()
    if ckpt["K"] != 8:
        raise ValueError("Inspire model must have eight learned heads")
    hand_model = original.create_hand_model("inspire", device=torch.device("cpu"))
    fingers = hand_model.get_joint_orders()[6:]
    lower, upper = hand_model.pk_chain.get_joint_limits()
    out_rows = {head: [] for head in range(8)}
    for row in sources.source_rows(source_file(line), "inspire", 64):
        obj = row["object_name"]
        raw = torch.as_tensor(row["q_dro_order"], dtype=torch.float32)
        q_seed, _ = original.lsq_project_q_local(raw, fingers, lower[6:], upper[6:])
        points = original.load_object_points(obj).unsqueeze(0)
        with torch.no_grad():
            for sid in range(64):
                wrist = q_seed[sid, :6].reshape(1, 6)
                prediction = model(points, q_seed[sid].reshape(1, 18), wrist)
                for head in range(8):
                    out_rows[head].append({"object_name": obj, "source_grasp_index": sid,
                                           "generator": "DRO" if LINES[line][1] == "DRO" else "GenDexGrasp",
                                           "head_id": head,
                                           "params": torch.cat([prediction["pred_w"][0, head],
                                                                prediction["pred_u"][0, head]]).cpu().float(),
                                           "source_q_seed": q_seed[sid].clone()})
    for head, seeds in out_rows.items():
        if len(seeds) != 640:
            raise ValueError(f"Inspire head{head} has {len(seeds)}/640 predictions")
        torch.save({"info": {"model_sha256": metrics.sha(model_path(line)), "head_id": head,
                             "zero_file": str(source_file(line)), "kind": "pd"}, "seeds": seeds},
                   new_warm(line, head))


def prepare(line):
    preflight(line)
    manifest = warm_manifest(line)
    if manifest.exists() or any(new_warm(line, head).exists() for head in range(8)):
        check_warm(line)
        return
    folder = manifest.parent
    folder.mkdir(parents=True, exist_ok=True)
    h, _, g, _ = LINES[line]
    if h == "inspire":
        prepare_inspire(line, folder)
    else:
        script = ROOT / f"results/pd_refiner_{h}" / (
            "generate_step6_mixed_warmstarts.py" if h == "ability" else "generate_svh_mixed_warmstarts.py")
        cmd = [sys.executable, str(script), "--model", str(model_path(line)),
               "--zero-file", str(source_file(line)), "--out-dir", str(folder),
               "--tag", f"{g}_b64", "--max-grasps", "64"]
        with (folder / "prepare.log").open("w") as stream:
            process = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"{line} warm-start generation failed; inspect {folder/'prepare.log'}")
    if any(not new_warm(line, head).is_file() for head in range(8)):
        raise RuntimeError(f"Incomplete {line} warm-start generation: {folder}")
    atomic_json(manifest, {"source_sha256": metrics.sha(source_file(line)),
                           "model_sha256": metrics.sha(model_path(line)),
                           "old_warm_sha256": [metrics.sha(old_warm(line, head)) for head in range(8)],
                           "warm_sha256": [metrics.sha(new_warm(line, head)) for head in range(8)]})
    check_warm(line)


def check_warm(line):
    import torch

    source_sha = check_source(line)
    preflight(line)
    manifest_path = warm_manifest(line)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run prepare first: {manifest_path}")
    meta = json.loads(manifest_path.read_text())
    if (meta.get("source_sha256") != source_sha or meta.get("model_sha256") != metrics.sha(model_path(line))
            or meta.get("old_warm_sha256") != [metrics.sha(old_warm(line, head)) for head in range(8)]):
        raise ValueError(f"Model or source changed after {line} warm generation")
    target_keys = {(obj, sid) for obj in OBJECTS for sid in range(64)}
    for head in range(8):
        p = new_warm(line, head)
        if not p.is_file() or metrics.sha(p) != meta["warm_sha256"][head]:
            raise ValueError(f"Missing/modified warm start: {p}")
        payload = torch.load(p, map_location="cpu", weights_only=False)
        seeds = payload["seeds"]
        index = {(r["object_name"], int(r["source_grasp_index"])): r for r in seeds}
        if (len(seeds) != 640 or set(index) != target_keys
                or payload["info"].get("zero_file") != str(source_file(line))
                or any(torch.as_tensor(r["params"]).numel() != PARAM_DIMS[hand(line)] for r in seeds)):
            raise ValueError(f"Incomplete or wrong-hand warm starts: {p}")
        old = torch.load(old_warm(line, head), map_location="cpu", weights_only=False)
        for r in old["seeds"]:
            key = (r["object_name"], int(r["source_grasp_index"]))
            if not torch.equal(torch.as_tensor(r["params"]), torch.as_tensor(index[key]["params"])):
                raise ValueError(f"Old20 predictions changed: {line}/head{head}/{key}")
            if "source_q_seed" in r and not torch.equal(
                    torch.as_tensor(r["source_q_seed"]), torch.as_tensor(index[key]["source_q_seed"])):
                raise ValueError(f"Old20 projected source changed: {line}/{key}")
    print(f"{line}: eight learned warm-start files each cover 640 targets; old20 unchanged", flush=True)
    return source_sha


def check_new(line, obj, arm, part):
    import torch

    h = hand(line)
    path = new_path(line, obj, arm, part)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    selected = indices(part)
    expected = {"zero_file": str(source_file(line)), "selection_battery": BATTERY,
                "max_grasps": 64, "source_indices": selected,
                "population": 32, "generations": 1, "elites": 8, "restarts": 1,
                "batch_source_grasps": 8, "keep_per_grasp": 3,
                "keep_per_object": 3 * len(selected), "cpu_final_candidates": 240,
                "search_pipeline": "gpu", "reuse_gpu_sim": True, "log_rejected": True,
                "seed": 10001 + 1009 * part,
                "warm_start_file": None if arm == "default" else str(new_warm(line, int(arm[4:])))}
    for key, value in expected.items():
        if data["args"].get(key) != value:
            raise ValueError(f"Wrong new44 search {key}={data['args'].get(key)!r}: {path}")
    if h == "inspire" and (data["args"].get("allow_non_train") is not True
                           or data["args"].get("arm_id") != f"pd_{arm}"):
        raise ValueError(f"Wrong Inspire warm-start branch: {path}")
    if (data.get("objects") != [obj] or data.get("total_source_grasps") != len(selected)
            or len(data.get("rows", [])) != 1):
        raise ValueError(f"Incomplete new44 search: {path}")
    row = data["rows"][0]
    if ([int(r["grasp_index"]) for r in row["grasp_rows"]] != selected
            or torch.as_tensor(row["q_seed"]).shape != (len(selected), DIMS[h])):
        raise ValueError(f"Wrong new44 target indices: {path}")
    check_candidate_row(path, row, set(selected), 3 * len(selected), h)
    return row


def search(line, objects, chosen_arms=None, chosen_part=None):
    check_warm(line)
    driver = ROOT / "eval" / f"generate_{hand(line)}_train_bsearch.py"
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    if not driver.is_file():
        raise FileNotFoundError(driver)
    for obj in objects:
        for arm in (chosen_arms or ARMS):
            for part in ((chosen_part,) if chosen_part is not None else range(6)):
                dest = new_path(line, obj, arm, part)
                if dest.exists():
                    check_new(line, obj, arm, part)
                    print(f"reuse {line}/{obj}/{arm}/part{part}", flush=True)
                    continue
                ids = indices(part)
                dest.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(driver), "--zero-file", str(source_file(line)),
                       "--objects", obj, "--max-grasps", "64", "--selection-battery", BATTERY,
                       "--population", "32", "--elites", "8", "--generations", "1", "--restarts", "1",
                       "--batch-source-grasps", "8", "--keep-per-grasp", "3",
                       "--keep-per-object", str(3 * len(ids)), "--cpu-final-candidates", "240",
                       "--search-pipeline", "gpu", "--reuse-gpu-sim", "--log-rejected",
                       "--seed", str(10001 + 1009 * part), "--source-indices", *(str(i) for i in ids),
                       "--out", str(dest)]
                if arm != "default":
                    cmd.extend(("--warm-start-file", str(new_warm(line, int(arm[4:])))))
                if hand(line) == "inspire":
                    cmd.extend(("--allow-non-train", "--arm-id", f"pd_{arm}"))
                log = dest.with_suffix(".log")
                with log.open("w") as stream:
                    process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode or not dest.is_file():
                    raise RuntimeError(f"Search failed; inspect {log}")
                check_new(line, obj, arm, part)
                print(f"searched {line}/{obj}/{arm}/part{part}: {len(ids)} new targets", flush=True)


def collect(line, obj):
    import torch

    candidates, files = [], []
    old_q, new_q = None, {}
    for arm in ARMS:
        row = check_old(line, obj, arm)
        q = torch.as_tensor(row["q_seed"])
        if old_q is not None and not torch.equal(old_q, q):
            raise ValueError(f"Old20 source grasp mismatch across arms: {line}/{obj}")
        old_q = q
        paths_rows = [("old", old_path(line, obj, arm), row)]
        paths_rows.extend((part, new_path(line, obj, arm, part), check_new(line, obj, arm, part))
                          for part in range(6))
        for part, path, pool in paths_rows:
            if part != "old":
                projected = torch.as_tensor(pool["q_seed"])
                if part in new_q and not torch.equal(new_q[part], projected):
                    raise ValueError(f"New44 source grasp mismatch across arms: {line}/{obj}/part{part}")
                new_q[part] = projected
            files.append({"path": str(path), "sha256": metrics.sha(path)})
            for kind in ("passing_candidates", "cpu_final_rejected_candidates"):
                for i, cand in enumerate(pool[kind]):
                    params = torch.as_tensor(cand["params"], dtype=torch.float32).contiguous()
                    candidates.append({"id": f"{arm}/{part}/{kind}/{i}",
                                       "source_index": int(cand["source_grasp_index"]),
                                       "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(DIMS[hand(line)]),
                                       "gpu_rank_score": float(cand["gpu_rank_score"]),
                                       "search_params_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
                                       "search_cpu_success": bool(cand["cpu_success"])})
    return candidates, files


def ensure_cache(line, obj, candidates, files, top_ids, source_sha, torch):
    path = cache_path(line, obj)
    ids = [c["id"] for c in candidates]
    sources_id = [c["source_index"] for c in candidates]
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("line") != line or saved.get("source_sha256") != source_sha
                or saved.get("model_sha256") != metrics.sha(model_path(line))
                or saved.get("search_files") != files or saved.get("candidate_ids") != ids
                or saved.get("source_indices") != sources_id or saved.get("top1_candidate_ids") != top_ids
                or torch.as_tensor(saved.get("q")).shape != (len(candidates), DIMS[hand(line)])):
            raise ValueError(f"Stale cache, refusing to overwrite: {path}")
    else:
        payload = {"line": line, "object_name": obj, "source_sha256": source_sha,
                   "model_sha256": metrics.sha(model_path(line)), "search_files": files,
                   "battery": BATTERY, "threshold_m": metrics.THRESHOLD,
                   "verify_batch_size": VERIFY_BATCH, "candidate_ids": ids,
                   "source_indices": sources_id, "top1_candidate_ids": top_ids,
                   "q": torch.stack([c["q"] for c in candidates]) if candidates else torch.empty((0, DIMS[hand(line)]))}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
    return metrics.sha(path)


def checked_batch(line, obj, batch, candidates, cache_sha):
    path = batch_path(line, obj, batch)
    if not path.is_file():
        return None
    rec = json.loads(path.read_text())
    block = candidates[batch * VERIFY_BATCH:(batch + 1) * VERIFY_BATCH]
    if (rec.get("line") != line or rec.get("object_name") != obj
            or rec.get("batch_index") != batch or rec.get("cache_sha256") != cache_sha
            or rec.get("candidate_ids") != [c["id"] for c in block]
            or rec.get("source_indices") != [c["source_index"] for c in block]
            or rec.get("battery") != BATTERY or rec.get("threshold_m") != metrics.THRESHOLD
            or rec.get("verify_batch_size") != VERIFY_BATCH or rec.get("simulated_envs") != VERIFY_BATCH
            or len(rec.get("distance_m", [])) != len(block)
            or len(rec.get("candidate_success", [])) != len(block)
            or any(type(x) is not bool for x in rec["candidate_success"])):
        raise ValueError(f"Stale/incomplete batch label: {path}")
    return rec


def worker(line, obj, first, count, expected_sha):
    # Fresh process MUST load IsaacGym/validator before importing torch.
    sys.path.insert(0, str(ROOT / "eval"))
    import importlib
    validator = importlib.import_module(f"generate_{hand(line)}_train_bsearch")
    import torch

    path = cache_path(line, obj)
    if not path.is_file() or metrics.sha(path) != expected_sha:
        raise ValueError(f"Candidate cache changed: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if (cache["line"] != line or cache["object_name"] != obj or cache["battery"] != BATTERY
            or cache["verify_batch_size"] != VERIFY_BATCH):
        raise ValueError(f"Wrong worker cache: {path}")
    n = len(cache["candidate_ids"])
    total_batches = (n + VERIFY_BATCH - 1) // VERIFY_BATCH
    if first not in range(total_batches) or not 1 <= count <= WORKER_BATCHES:
        raise ValueError(f"Invalid short-worker range {first}+{count}/{total_batches}")
    for batch in range(first, min(total_batches, first + count)):
        begin, end = batch * VERIFY_BATCH, min(n, (batch + 1) * VERIFY_BATCH)
        dest = batch_path(line, obj, batch)
        if dest.exists():
            previous = json.loads(dest.read_text())
            if (previous.get("cache_sha256") != expected_sha
                    or previous.get("candidate_ids") != cache["candidate_ids"][begin:end]
                    or previous.get("source_indices") != cache["source_indices"][begin:end]):
                raise ValueError(f"Conflicting saved batch: {dest}")
            continue
        q = cache["q"][begin:end].clone()
        if end - begin < VERIFY_BATCH:
            q = torch.cat([q, q[-1:].expand(VERIFY_BATCH - (end - begin), -1)], dim=0)
        d, _ = validator.score_fresh(obj, q, BATTERY, use_gpu_pipeline=False)
        values = [float(d[i]) for i in range(end - begin)]
        atomic_json(dest, {"line": line, "object_name": obj, "batch_index": batch,
                           "cache_sha256": expected_sha, "battery": BATTERY,
                           "threshold_m": metrics.THRESHOLD,
                           "verify_batch_size": VERIFY_BATCH, "simulated_envs": int(q.shape[0]),
                           "candidate_ids": cache["candidate_ids"][begin:end],
                           "source_indices": cache["source_indices"][begin:end],
                           "distance_m": [v if math.isfinite(v) else None for v in values],
                           "candidate_success": [math.isfinite(v) and v <= metrics.THRESHOLD for v in values]})
        print(f"checkpointed {line}/{obj} CPU batch {batch + 1}/{total_batches}", flush=True)


def recheck(line, objects, max_workers=None):
    import torch  # manager never starts a simulator

    source_sha = check_warm(line)
    started = 0
    for obj in objects:
        candidates, files = collect(line, obj)
        selected = metrics.freeze_top1(candidates)
        selected_ids = [candidates[i]["id"] if i is not None else None for i in selected]
        dest = result_path(line, obj)
        if dest.exists():
            saved = json.loads(dest.read_text())
            if (saved.get("source_sha256") != source_sha or saved.get("model_sha256") != metrics.sha(model_path(line))
                    or saved.get("search_files") != files or saved.get("candidate_ids") != [c["id"] for c in candidates]
                    or saved.get("top1_candidate_ids") != selected_ids or saved.get("battery") != BATTERY
                    or saved.get("verify_batch_size") != VERIFY_BATCH
                    or len(saved.get("candidate_success", [])) != len(candidates)):
                raise ValueError(f"Stale full recheck, refusing to overwrite: {dest}")
            if (saved["top1_success"], saved["multi_success"]) != metrics.outcomes(
                    candidates, saved["candidate_success"], selected):
                raise ValueError(f"Saved candidate labels disagree: {dest}")
            print(f"reuse one-time {line}/{obj} verification", flush=True)
            continue
        cache_sha = ensure_cache(line, obj, candidates, files, selected_ids, source_sha, torch)
        total_batches = (len(candidates) + VERIFY_BATCH - 1) // VERIFY_BATCH
        for first in range(0, total_batches, WORKER_BATCHES):
            last = min(total_batches, first + WORKER_BATCHES)
            if all(checked_batch(line, obj, b, candidates, cache_sha) is not None
                   for b in range(first, last)):
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--line", line,
                   "--object", obj, "--start-batch", str(first), "--batch-count", str(last - first),
                   "--cache-sha", cache_sha]
            log = batch_path(line, obj, first).with_name(f"worker{first:04d}_{last-1:04d}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(f"running {line}/{obj} CPU batches {first}..{last-1}\n")
                stream.flush()
                status = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f"Worker exited {status.returncode}; saved batches reusable. Inspect {log}")
            for b in range(first, last):
                if checked_batch(line, obj, b, candidates, cache_sha) is None:
                    raise RuntimeError(f"Worker omitted batch {b}; inspect {log}")
            print(f"{line}/{obj}: saved batches {first+1}..{last}/{total_batches}", flush=True)
            started += 1
            if max_workers is not None and started >= max_workers:
                print("Short-worker limit reached; recheck again to resume.", flush=True)
                return
        distances, labels, batches = [], [], []
        for b in range(total_batches):
            rec = checked_batch(line, obj, b, candidates, cache_sha)
            if rec is None:
                raise RuntimeError(f"Missing frozen validation batch {line}/{obj}/{b}")
            distances.extend(rec["distance_m"])
            labels.extend(rec["candidate_success"])
            batches.append({"real_candidates": len(rec["candidate_ids"]), "simulated_envs": VERIFY_BATCH})
        if len(labels) != len(candidates):
            raise ValueError(f"Incomplete validation candidate labels {line}/{obj}")
        top1, multi = metrics.outcomes(candidates, labels, selected)
        atomic_json(dest, {"line": line, "hand": hand(line), "generator": LINES[line][1],
                           "object_name": obj, "source_sha256": source_sha,
                           "model_sha256": metrics.sha(model_path(line)),
                           "old20_search_reused": True, "new44_search_only": True,
                           "gpu_evaluations_per_source": 288, "search_files": files,
                           "recheck_cache_sha256": cache_sha, "worker_batches_per_process": WORKER_BATCHES,
                           "battery": BATTERY, "threshold_m": metrics.THRESHOLD,
                           "verify_batch_size": VERIFY_BATCH, "tail_padded": True,
                           "shuffle": False, "revalidation_count": 1,
                           "candidate_count": len(candidates), "cpu_batches": batches,
                           "selection_rule": "minimum search-time gpu_rank_score in persisted CPU-final-evaluated pool; parameter hash breaks ties",
                           "candidate_ids": [c["id"] for c in candidates],
                           "source_indices": [c["source_index"] for c in candidates],
                           "candidate_gpu_rank_scores": [c["gpu_rank_score"] for c in candidates],
                           "candidate_search_params_sha256": [c["search_params_sha256"] for c in candidates],
                           "search_cpu_success": [c["search_cpu_success"] for c in candidates],
                           "distance_m": distances, "candidate_success": labels,
                           "top1_candidate_indices": selected, "top1_candidate_ids": selected_ids,
                           "top1_success": top1, "multi_success": multi,
                           "top1_old20_success": sum(top1[:20]), "top1_new44_success": sum(top1[20:]),
                           "multi_old20_success": sum(multi[:20]), "multi_new44_success": sum(multi[20:])})
        print(f"rechecked {line}/{obj}: top-1 {sum(top1)}/64, multi {sum(multi)}/64", flush=True)


def summarize(line):
    source_sha = check_source(line)
    counts = {name: [0, 0] for name in ("top1", "multi")}
    rates, per_object = {name: [] for name in counts}, {}
    for obj in OBJECTS:
        path = result_path(line, obj)
        if not path.is_file():
            raise FileNotFoundError(f"Recheck all objects first: {path}")
        rec = json.loads(path.read_text())
        if (rec.get("line") != line or rec.get("source_sha256") != source_sha
                or rec.get("model_sha256") != metrics.sha(model_path(line))
                or rec.get("battery") != BATTERY or rec.get("verify_batch_size") != VERIFY_BATCH
                or rec.get("revalidation_count") != 1 or rec.get("tail_padded") is not True
                or any(not Path(x["path"]).is_file() or metrics.sha(Path(x["path"])) != x["sha256"]
                       for x in rec.get("search_files", []))):
            raise ValueError(f"Stale PD-9 result: {path}")
        n = rec["candidate_count"]
        if any(len(rec.get(key, [])) != n for key in ("candidate_ids", "source_indices",
               "candidate_success", "candidate_gpu_rank_scores", "candidate_search_params_sha256")):
            raise ValueError(f"Incomplete candidate-level success labels: {path}")
        candidates = [{"id": cid, "source_index": sid, "gpu_rank_score": float(score),
                       "search_params_sha256": param_hash} for cid, sid, score, param_hash in zip(
                           rec["candidate_ids"], rec["source_indices"],
                           rec["candidate_gpu_rank_scores"], rec["candidate_search_params_sha256"])]
        selected = metrics.freeze_top1(candidates)
        if rec["top1_candidate_indices"] != selected or rec["top1_candidate_ids"] != [
                candidates[i]["id"] if i is not None else None for i in selected]:
            raise ValueError(f"GPU top-1 was not frozen independently of CPU labels: {path}")
        top1, multi = metrics.outcomes(candidates, rec["candidate_success"], selected)
        if rec["top1_success"] != top1 or rec["multi_success"] != multi:
            raise ValueError(f"Candidate/target success keys disagree: {path}")
        per_object[obj] = {"candidate_count": n,
                           "targets_with_gpu_ranked_candidate": sum(i is not None for i in selected)}
        for kind, labels in (("top1", top1), ("multi", multi)):
            a, b = sum(labels[:20]), sum(labels[20:])
            if rec.get(f"{kind}_old20_success") != a or rec.get(f"{kind}_new44_success") != b:
                raise ValueError(f"Old20/new44 identity mismatch: {path}")
            counts[kind][0] += a
            counts[kind][1] += b
            rates[kind].append((a + b) / 64 * 100)
            per_object[obj][kind] = {"old20_success": a, "new44_success": b,
                                     "overall64_success": a + b,
                                     "rate_pct": (a + b) / 64 * 100, "source_success": labels}
    rng = random.Random(20261221)
    resamples = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    outcomes = {}
    for kind in ("top1", "multi"):
        a, b = counts[kind]
        boots = [sum(rates[kind][i] for i in draw) / 10 for draw in resamples]
        outcomes[kind] = {"old20": {"success": a, "total": 200, "rate_pct": a / 2},
                          "new44": {"success": b, "total": 440, "rate_pct": b / 4.4},
                          "overall": {"success": a + b, "total": 640, "rate_pct": (a + b) / 6.4,
                                      "object_cluster_95ci_pct": [metrics.quantile(boots, .025),
                                                                   metrics.quantile(boots, .975)]}}
    diff = [multi - top for top, multi in zip(rates["top1"], rates["multi"])]
    boot_diff = [sum(diff[i] for i in draw) / 10 for draw in resamples]
    report = {"line": line, "method": "PD-9/288", "battery": BATTERY,
              "source_sha256": source_sha, "model_sha256": metrics.sha(model_path(line)),
              "old20_search_reused": True, "new44_search_only": True,
              "metric": "top-1 and any-candidate recovery from the same one-time CPU-state recheck",
              "top1": outcomes["top1"], "multi": outcomes["multi"],
              "multi_minus_top1_pp": outcomes["multi"]["overall"]["rate_pct"] - outcomes["top1"]["overall"]["rate_pct"],
              "multi_minus_top1_object_cluster_95ci_pp": [metrics.quantile(boot_diff, .025),
                                                            metrics.quantile(boot_diff, .975)],
              "per_object": per_object}
    dest = OUT / line / "summary.json"
    atomic_json(dest, report)
    for kind, stats in outcomes.items():
        print(f"{line} PD-9 {kind}: old {stats['old20']['success']}/200 = {stats['old20']['rate_pct']:.2f}% | "
              f"new {stats['new44']['success']}/440 = {stats['new44']['rate_pct']:.2f}% | "
              f"overall {stats['overall']['success']}/640 = {stats['overall']['rate_pct']:.2f}%")
    print(f"saved {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "search", "recheck", "summarize", "worker"))
    parser.add_argument("--line", choices=(*LINES, "all"), default="all")
    parser.add_argument("--object", choices=OBJECTS, help="One-object search/recheck pilot")
    parser.add_argument("--arm", choices=ARMS, help="One search arm for a pilot")
    parser.add_argument("--part", type=int, choices=range(6), help="One new44 search part, requires --object")
    parser.add_argument("--max-worker-groups", type=int, help="Recheck smoke: limit short workers this invocation")
    parser.add_argument("--start-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cache-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "worker":
        if (args.line == "all" or args.object is None or args.start_batch is None
                or args.batch_count is None or not args.cache_sha or args.arm or args.part is not None):
            parser.error("worker requires one --line/--object and its batch/cache arguments")
        worker(args.line, args.object, args.start_batch, args.batch_count, args.cache_sha)
        return
    if args.start_batch is not None or args.batch_count is not None or args.cache_sha is not None:
        parser.error("internal worker arguments only apply to worker")
    if (args.object and args.stage not in ("search", "recheck")) or (
            (args.arm or args.part is not None) and args.stage != "search"):
        parser.error("--object is for search/recheck; --arm/--part are for search only")
    if args.part is not None and (not args.object or args.line == "all"):
        parser.error("--part requires one --line and --object")
    if args.max_worker_groups is not None and (args.stage != "recheck" or args.max_worker_groups < 1):
        parser.error("--max-worker-groups is for recheck and must be positive")
    selected_lines = LINES if args.line == "all" else (args.line,)
    for line in selected_lines:
        if args.stage == "check":
            preflight(line)
        elif args.stage == "prepare":
            prepare(line)
        elif args.stage == "search":
            search(line, (args.object,) if args.object else OBJECTS,
                   (args.arm,) if args.arm else None, args.part)
        elif args.stage == "recheck":
            recheck(line, (args.object,) if args.object else OBJECTS, args.max_worker_groups)
        else:
            summarize(line)


if __name__ == "__main__":
    main()
