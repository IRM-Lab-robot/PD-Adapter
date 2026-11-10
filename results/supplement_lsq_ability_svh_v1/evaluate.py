"""Per-batch checkpointed LSQ CPU-state recheck and paired target statistics."""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pipeline as p

ROOT, OUT = p.ROOT, p.OUT
BATCH = 32
WORKER_BATCHES = 4
THRESHOLD = .02


def result_file(hand, gen, method, obj):
    return OUT / "revalidation_once" / hand / gen / method / f"{p.safe(obj)}.json"


def batch_file(hand, gen, method, obj, index):
    return OUT / "revalidation_batches" / hand / gen / method / p.safe(obj) / f"batch{index:04d}.json"


def cache_file(hand, gen, method, obj):
    return OUT / "recheck_cache" / hand / gen / method / f"{p.safe(obj)}.pt"


def collect(hand, gen, method, obj):
    import torch

    candidates, files = [], []
    q_by_part = {}
    for arm in p.arms(method):
        for part in range(8):
            path = p.search_file(hand, gen, method, arm, obj, part)
            if not path.is_file():
                raise FileNotFoundError(f"Complete LSQ search first: {path}")
            row = p.check_search(path, hand, gen, method, arm, obj, part)
            q = torch.as_tensor(row["q_seed"])
            if part in q_by_part and not torch.equal(q_by_part[part], q):
                raise ValueError(f"Search arms are not paired by source grasp: {path}")
            q_by_part[part] = q
            files.append({"path": str(path), "sha256": p.common.sha(path)})
            for category in ("passing_candidates", "cpu_final_rejected_candidates"):
                for idx, cand in enumerate(row[category]):
                    param = torch.as_tensor(cand["params"], dtype=torch.float32).contiguous()
                    candidates.append({"id": f"{arm}/part{part:02d}/{category}/{idx}",
                                       "source_index": int(cand["source_grasp_index"]),
                                       "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(p.source_audit.DIMS[hand]),
                                       "gpu_rank_score": float(cand["gpu_rank_score"]),
                                       "search_params_sha256": hashlib.sha256(param.numpy().tobytes()).hexdigest(),
                                       "search_cpu_success": bool(cand["cpu_success"])})
    return candidates, files


def frozen_cache(hand, gen, method, obj, candidates, files, source_sha, top_ids, torch):
    path = cache_file(hand, gen, method, obj)
    ids = [c["id"] for c in candidates]
    sources = [c["source_index"] for c in candidates]
    model_sha = p.common.sha(p.model_file(hand)) if method == "pd9" else None
    if path.exists():
        prior = torch.load(path, map_location="cpu", weights_only=False)
        if (prior.get("source_sha256") != source_sha or prior.get("model_sha256") != model_sha
                or prior.get("search_files") != files or prior.get("candidate_ids") != ids
                or prior.get("source_indices") != sources or prior.get("top1_candidate_ids") != top_ids
                or prior.get("projection_mode") != "lsq"
                or torch.as_tensor(prior.get("q")).shape != (len(ids), p.source_audit.DIMS[hand])):
            raise ValueError(f"Frozen cache changed, refusing overwrite: {path}")
    else:
        payload = {"hand": hand, "gen": gen, "method": method, "object_name": obj,
                   "battery": p.BATTERY, "projection_mode": "lsq", "verify_batch_size": BATCH,
                   "source_sha256": source_sha, "model_sha256": model_sha,
                   "search_files": files, "candidate_ids": ids, "source_indices": sources,
                   "top1_candidate_ids": top_ids,
                   "q": torch.stack([c["q"] for c in candidates]) if candidates
                   else torch.empty((0, p.source_audit.DIMS[hand]))}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
    return p.common.sha(path)


def checked_batch(hand, gen, method, obj, number, candidates, cache_sha):
    path = batch_file(hand, gen, method, obj, number)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    group = candidates[number * BATCH:(number + 1) * BATCH]
    if (record.get("hand") != hand or record.get("gen") != gen or record.get("method") != method
            or record.get("object_name") != obj or record.get("batch_index") != number
            or record.get("cache_sha256") != cache_sha or record.get("projection_mode") != "lsq"
            or record.get("battery") != p.BATTERY or record.get("threshold_m") != THRESHOLD
            or record.get("verify_batch_size") != BATCH or record.get("simulated_envs") != BATCH
            or record.get("candidate_ids") != [c["id"] for c in group]
            or record.get("source_indices") != [c["source_index"] for c in group]
            or len(record.get("candidate_success", [])) != len(group)
            or len(record.get("distance_m", [])) != len(group)
            or any(type(v) is not bool for v in record["candidate_success"])):
        raise ValueError(f"Corrupt/mismatched physical batch: {path}")
    return record


def worker(hand, gen, method, obj, first, count, expected_sha):
    # Critical: Isaac Gym and the validator are imported before torch.
    sys.path.insert(0, str(ROOT / "eval"))
    import importlib
    validator = importlib.import_module(f"generate_{hand}_train_bsearch")
    import torch

    path = cache_file(hand, gen, method, obj)
    if not path.is_file() or p.common.sha(path) != expected_sha:
        raise ValueError(f"Frozen candidate file changed before worker started: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if (cache["hand"] != hand or cache["gen"] != gen or cache["method"] != method
            or cache["object_name"] != obj or cache["projection_mode"] != "lsq"
            or cache["battery"] != p.BATTERY or cache["verify_batch_size"] != BATCH):
        raise ValueError(f"Wrong physical worker input: {path}")
    n = len(cache["candidate_ids"])
    total = (n + BATCH - 1) // BATCH
    if first not in range(total) or not 1 <= count <= WORKER_BATCHES:
        raise ValueError(f"Invalid physical batch range {first}+{count}/{total}")
    for number in range(first, min(first + count, total)):
        start, end = number * BATCH, min((number + 1) * BATCH, n)
        dest = batch_file(hand, gen, method, obj, number)
        if dest.exists():
            old = json.loads(dest.read_text())
            if (old.get("cache_sha256") != expected_sha
                    or old.get("candidate_ids") != cache["candidate_ids"][start:end]
                    or old.get("source_indices") != cache["source_indices"][start:end]
                    or old.get("projection_mode") != "lsq"):
                raise ValueError(f"Conflicting physical batch already exists: {dest}")
            continue
        q = cache["q"][start:end].clone()
        if len(q) < BATCH:
            q = torch.cat([q, q[-1:].expand(BATCH - len(q), -1)], dim=0)
        distances, _ = validator.score_fresh(obj, q, p.BATTERY, use_gpu_pipeline=False,
                                             projection_mode="lsq")
        measured = [float(distances[i]) for i in range(end - start)]
        p.atomic_json(dest, {"hand": hand, "gen": gen, "method": method, "object_name": obj,
                             "batch_index": number, "cache_sha256": expected_sha,
                             "projection_mode": "lsq", "battery": p.BATTERY,
                             "threshold_m": THRESHOLD, "verify_batch_size": BATCH,
                             "simulated_envs": int(q.shape[0]),
                             "candidate_ids": cache["candidate_ids"][start:end],
                             "source_indices": cache["source_indices"][start:end],
                             "distance_m": [x if math.isfinite(x) else None for x in measured],
                             "candidate_success": [math.isfinite(x) and x <= THRESHOLD for x in measured]})
        print(f"saved LSQ batch {hand}/{gen}/{method}/{obj} {number + 1}/{total}", flush=True)


def recheck(hand, gen, method, objects, worker_limit=None):
    import torch  # Manager only reads candidates; short workers import Isaac Gym.

    p.check_inputs(hand)
    source_sha = p.common.sha(p.source_file(hand, gen))
    launched = 0
    for obj in objects:
        candidates, files = collect(hand, gen, method, obj)
        chosen = p.common.freeze_top1(candidates)
        top_ids = [candidates[i]["id"] if i is not None else None for i in chosen]
        dest = result_file(hand, gen, method, obj)
        if dest.exists():
            result = json.loads(dest.read_text())
            if (result.get("search_files") != files or result.get("source_sha256") != source_sha
                    or result.get("projection_mode") != "lsq"
                    or result.get("candidate_ids") != [c["id"] for c in candidates]
                    or result.get("top1_candidate_ids") != top_ids
                    or len(result.get("candidate_success", [])) != len(candidates)):
                raise ValueError(f"Stale full LSQ result: {dest}")
            if (result["top1_success"], result["multi_success"]) != p.common.outcomes(
                    candidates, result["candidate_success"], chosen):
                raise ValueError(f"Stored labels disagree with target success: {dest}")
            print(f"reuse {hand}/{gen}/{method}/{obj} full LSQ recheck", flush=True)
            continue
        cache_sha = frozen_cache(hand, gen, method, obj, candidates, files, source_sha, top_ids, torch)
        total = (len(candidates) + BATCH - 1) // BATCH
        for first in range(0, total, WORKER_BATCHES):
            last = min(first + WORKER_BATCHES, total)
            if all(checked_batch(hand, gen, method, obj, i, candidates, cache_sha) is not None
                   for i in range(first, last)):
                continue
            cmd = [sys.executable, str(OUT / "run.py"), "worker", "--hand", hand, "--gen", gen,
                   "--method", method, "--object", obj,
                   "--batch-start", str(first), "--batch-count", str(last - first),
                   "--cache-sha", cache_sha]
            log = batch_file(hand, gen, method, obj, first).with_name(f"worker{first:04d}_{last-1:04d}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(f"run LSQ {hand}/{gen}/{method}/{obj} batch {first}..{last-1}\n")
                stream.flush()
                ret = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if ret.returncode:
                raise RuntimeError(f"Physical worker failed ({ret.returncode}); re-run to resume. Inspect {log}")
            if any(checked_batch(hand, gen, method, obj, i, candidates, cache_sha) is None
                   for i in range(first, last)):
                raise RuntimeError(f"Worker failed to persist batches: {log}")
            print(f"{hand}/{gen}/{method}/{obj}: physical batches {first + 1}..{last}/{total}", flush=True)
            launched += 1
            if worker_limit is not None and launched >= worker_limit:
                print("Requested short worker limit reached; recheck again to resume.", flush=True)
                return
        distances, labels, batch_info = [], [], []
        for number in range(total):
            rec = checked_batch(hand, gen, method, obj, number, candidates, cache_sha)
            if rec is None:
                raise RuntimeError(f"Missing LSQ physical batch {number} for {obj}")
            distances.extend(rec["distance_m"])
            labels.extend(rec["candidate_success"])
            batch_info.append({"real_candidates": len(rec["candidate_ids"]), "simulated_envs": BATCH})
        if len(labels) != len(candidates):
            raise ValueError(f"Incorrect LSQ candidate label count: {hand}/{gen}/{method}/{obj}")
        top1, multi = p.common.outcomes(candidates, labels, chosen)
        p.atomic_json(dest, {"hand": hand, "gen": gen, "method": method, "object_name": obj,
                             "source_sha256": source_sha,
                             "model_sha256": p.common.sha(p.model_file(hand)) if method == "pd9" else None,
                             "search_files": files, "cache_sha256": cache_sha,
                             "gpu_evaluations_per_source": 288, "battery": p.BATTERY,
                             "projection_mode": "lsq", "verify_batch_size": BATCH,
                             "threshold_m": THRESHOLD, "tail_padded": True,
                             "shuffle": False, "revalidation_count": 1,
                             "candidate_count": len(candidates), "cpu_batches": batch_info,
                             "selection_rule": "lowest search-time GPU score; raw parameter hash breaks ties",
                             "candidate_ids": [c["id"] for c in candidates],
                             "source_indices": [c["source_index"] for c in candidates],
                             "candidate_gpu_scores": [c["gpu_rank_score"] for c in candidates],
                             "candidate_param_hashes": [c["search_params_sha256"] for c in candidates],
                             "search_cpu_success": [c["search_cpu_success"] for c in candidates],
                             "distance_m": distances, "candidate_success": labels,
                             "top1_candidate_indices": chosen, "top1_candidate_ids": top_ids,
                             "top1_success": top1, "multi_success": multi,
                             "top1_old20_success": sum(top1[:20]), "top1_new44_success": sum(top1[20:]),
                             "multi_old20_success": sum(multi[:20]), "multi_new44_success": sum(multi[20:])})
        print(f"LSQ {hand}/{gen}/{method}/{obj}: top-1 {sum(top1)}/64; multi {sum(multi)}/64", flush=True)


def summarize(hand, gen, methods=None, write=True):
    p.check_inputs(hand)
    source_sha = p.common.sha(p.source_file(hand, gen))
    missing = {method: [obj for obj in p.TEST10 if not result_file(hand, gen, method, obj).is_file()]
               for method in p.METHODS}
    if methods is None:
        selected = tuple(method for method in p.METHODS if not missing[method])
        if not selected:
            raise FileNotFoundError(f"No complete 10-object LSQ method for {hand}/{gen}; missing: {missing}")
    else:
        selected = tuple(methods)
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(p.METHODS):
            raise ValueError(f"Invalid methods for summarize: {selected}")
        for method in selected:
            if missing[method]:
                raise FileNotFoundError(f"{hand}/{gen}/{method} needs all 10 objects; missing: {missing[method]}")
    model_path = p.model_file(hand)
    model_sha = p.common.sha(model_path) if model_path.is_file() else None
    if "pd9" in selected and model_sha is None:
        raise FileNotFoundError(f"PD-9 model checkpoint is missing: {model_path}")
    labels = {method: {} for method in selected}
    stats = {}
    for method in selected:
        old, extra, rates = {name: 0 for name in ("top1", "multi")}, {
            name: 0 for name in ("top1", "multi")}, {name: [] for name in ("top1", "multi")}
        per_object, candidate_total = {}, 0
        for obj in p.TEST10:
            path = result_file(hand, gen, method, obj)
            if not path.is_file():
                raise FileNotFoundError(f"Finish all three methods/objects before summary: {path}")
            row = json.loads(path.read_text())
            if (row.get("source_sha256") != source_sha or row.get("projection_mode") != "lsq"
                    or row.get("battery") != p.BATTERY or row.get("verify_batch_size") != BATCH
                    or row.get("revalidation_count") != 1 or row.get("tail_padded") is not True
                    or row.get("method") != method or row.get("object_name") != obj
                    or row.get("hand") != hand or row.get("gen") != gen):
                raise ValueError(f"Invalid LSQ physical protocol: {path}")
            if method == "pd9" and row.get("model_sha256") != model_sha:
                raise ValueError(f"PD-9 result checkpoint no longer matches training model: {path}")
            files = row["search_files"]
            if (len(files) != 8 * len(p.arms(method)) or
                    any(not Path(f["path"]).is_file() or p.common.sha(Path(f["path"])) != f["sha256"]
                        for f in files)):
                raise ValueError(f"Search pool changed since LSQ recheck: {path}")
            n = row["candidate_count"]
            if any(len(row.get(k, [])) != n for k in ("candidate_ids", "source_indices",
                    "candidate_success", "candidate_gpu_scores", "candidate_param_hashes")):
                raise ValueError(f"Incomplete candidate-level LSQ data: {path}")
            candidates = [{"id": cid, "source_index": sid, "gpu_rank_score": float(score),
                           "search_params_sha256": hsh} for cid, sid, score, hsh in zip(
                               row["candidate_ids"], row["source_indices"],
                               row["candidate_gpu_scores"], row["candidate_param_hashes"])]
            top_indices = p.common.freeze_top1(candidates)
            if (row["top1_candidate_indices"] != top_indices or row["top1_candidate_ids"] != [
                    candidates[i]["id"] if i is not None else None for i in top_indices]):
                raise ValueError(f"Top-1 was not frozen on GPU-only ranking: {path}")
            top1, multi = p.common.outcomes(candidates, row["candidate_success"], top_indices)
            if row["top1_success"] != top1 or row["multi_success"] != multi:
                raise ValueError(f"Candidate labels disagree with target outcomes: {path}")
            candidate_total += n
            per_object[obj] = {"candidate_count": n,
                               "targets_with_candidate": sum(i is not None for i in top_indices)}
            for label, success in (("top1", top1), ("multi", multi)):
                a, b = sum(success[:20]), sum(success[20:])
                if row[f"{label}_old20_success"] != a or row[f"{label}_new44_success"] != b:
                    raise ValueError(f"Invalid old20/new44 accounting in {path}")
                old[label] += a
                extra[label] += b
                rates[label].append((a + b) / 64 * 100)
                labels[method].setdefault(label, {})[obj] = success
                per_object[obj][label] = {"old20_success": a, "new44_success": b,
                                         "overall64_success": a + b, "rate_pct": (a + b) / 64 * 100,
                                         "success": success}
        rng = random.Random(20261221)
        draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
        rate_stats = {}
        for label in ("top1", "multi"):
            total = old[label] + extra[label]
            boots = [sum(rates[label][i] for i in draw) / 10 for draw in draws]
            rate_stats[label] = {"old20": {"success": old[label], "total": 200, "rate_pct": old[label] / 2},
                                 "new44": {"success": extra[label], "total": 440, "rate_pct": extra[label] / 4.4},
                                 "overall": {"success": total, "total": 640, "rate_pct": total / 6.4,
                                             "object_cluster_95ci_pct": [p.common.quantile(boots, .025),
                                                                          p.common.quantile(boots, .975)]}}
        stats[method] = {"search_samples_per_target": 288,
                         "search_cpu_final_candidates_per_target": candidate_total / 640,
                         "top1": rate_stats["top1"], "multi": rate_stats["multi"],
                         "per_object": per_object}
    comparisons = {}
    rng = random.Random(20261221)
    draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    for reference in ("cem288", "sample288"):
        if "pd9" not in stats or reference not in stats:
            continue
        by_metric = {}
        for metric in ("top1", "multi"):
            object_deltas = [sum(int(a) - int(b) for a, b in zip(labels["pd9"][metric][obj],
                                                                 labels[reference][metric][obj])) / 64 * 100
                             for obj in p.TEST10]
            boots = [sum(object_deltas[i] for i in draw) / 10 for draw in draws]
            by_metric[metric] = {"delta_pp": sum(object_deltas) / 10,
                                 "object_cluster_95ci_pp": [p.common.quantile(boots, .025),
                                                             p.common.quantile(boots, .975)],
                                 "per_object_delta_pp": dict(zip(p.TEST10, object_deltas))}
        comparisons[f"pd9_minus_{reference}"] = by_metric
    result = {"hand": hand, "gen": gen, "battery": p.BATTERY,
              "projection_mode_search_gpu_cpu_final_and_recheck": "lsq",
              "revalidation_count": 1, "verify_batch_size": BATCH, "tail_padded": True,
              "training_model_sha256": model_sha,
              "source_sha256": source_sha, "num_objects": 10, "sources_per_object": 64,
              "summary_scope": "available_complete_methods" if methods is None else "requested_methods",
              "completed_methods": list(selected),
              "incomplete_methods": [method for method in p.METHODS if missing[method]],
              "missing_revalidation_objects": {method: missing[method] for method in p.METHODS if missing[method]},
              "methods": stats, "comparisons": comparisons}
    name = f"{hand}_{gen}" if methods is None else f"{hand}_{gen}_{'_'.join(selected)}"
    dest = OUT / "summary" / f"{name}.json"
    if write:
        p.atomic_json(dest, result)
    for method in selected:
        print(f"{hand}/{gen}/{method}: top-1 {stats[method]['top1']['overall']['rate_pct']:.2f}% "
              f"multi {stats[method]['multi']['overall']['rate_pct']:.2f}%", flush=True)
    print(f"{'saved' if write else 'validated without writing'} {dest}", flush=True)
    return result
