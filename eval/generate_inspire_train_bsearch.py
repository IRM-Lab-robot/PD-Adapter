import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DRO_ROOT))

import numpy as np
from spatial_sampling import LAYERS, layer_slices, sample_stratified

from holdout_isaac_battery import BATTERIES, HoldoutValidator, run_holdout
import inspire_coupling
from native_inspire_cem_search import lsq_project_q, params_to_q
from utils.hand_model import create_hand_model

import torch


def load_warm_start_file(path_arg):
    if not path_arg:
        return None
    path = Path(path_arg)
    if not path.is_absolute():
        path = ROOT / path
    data = torch.load(path, map_location="cpu")
    rows = data.get("seeds", data.get("rows", [])) if isinstance(data, dict) else data
    warm = {}
    for row in rows:
        object_name = str(row["object_name"])
        source_idx = int(row["source_grasp_index"])
        params = torch.as_tensor(row["params"], dtype=torch.float32).view(12).clone()
        key = (object_name, source_idx)
        if key in warm:
            raise ValueError(f"Duplicate warm-start key {key}; pass a generator-specific warm-start file")
        warm[key] = params.numpy().astype(np.float32)
    return {"path": str(path), "map": warm, "count": len(warm)}


def warm_start_for(args, object_name, grasp_index):
    warm = getattr(args, "_warm_start_map", None)
    if not warm:
        return None
    return warm.get((object_name, int(grasp_index)))


def annotate_gpu_pass(row, object_name, source_idx, arm_id, rng_seed, validation_order_idx, entered_cpu_final, cpu_final_reason):
    out = dict(row)
    out["candidate_id"] = f"{arm_id}:{object_name}:{int(source_idx)}:{int(validation_order_idx)}"
    out["arm_id"] = arm_id
    out["object_name"] = object_name
    out["source_grasp_index"] = int(source_idx)
    out["rng_seed"] = int(rng_seed)
    out["validation_order_idx"] = int(validation_order_idx)
    out["gpu_rank_score"] = float(out.get("gpu_rank_score", out.get("rank_score", out.get("score", float("inf")))))
    out["entered_cpu_final"] = bool(entered_cpu_final)
    out["cpu_final_reason"] = cpu_final_reason
    return out


def maybe_print_warm_start(args, object_name, grasp_index, params):
    if getattr(args, "_warm_start_printed", False):
        return
    if params is None:
        return
    print(
        f"warm_start_used object={object_name} source={int(grasp_index)} "
        f"mean_first3={[round(float(x), 6) for x in np.asarray(params).reshape(12)[:3]]} "
        f"u_first3={[round(float(x), 6) for x in np.asarray(params).reshape(12)[6:9]]}",
        flush=True,
    )
    args._warm_start_printed = True


def selected_source_indices(args, num_sources):
    if args.source_indices is None:
        return list(range(num_sources))
    out = []
    for idx in args.source_indices:
        value = int(idx)
        if value < 0 or value >= num_sources:
            raise ValueError(f"source index {value} out of range [0, {num_sources})")
        out.append(value)
    return out


def active_limits(finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    lo = []
    hi = []
    for active_name in inspire_coupling.active_joints:
        active_idx = name2finger[active_name]
        lo.append(float(lower[active_idx]))
        hi.append(float(upper[active_idx]))
    return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


def clamp_stats(samples, u_lower, u_upper):
    u = samples[..., 6:]
    out = np.logical_or(u < u_lower.reshape((1,) * (u.ndim - 1) + (6,)), u > u_upper.reshape((1,) * (u.ndim - 1) + (6,)))
    sample_out = np.any(out, axis=-1)
    return int(sample_out.sum()), int(sample_out.size), float(sample_out.mean()) if sample_out.size else 0.0


def score_fresh(object_name, q_batch, battery_name, use_gpu_pipeline=False):
    success, _, distance = run_holdout(
        object_name,
        q_batch.cpu(),
        projection_mode="lsq",
        battery=BATTERIES[battery_name],
        use_gpu_pipeline=use_gpu_pipeline,
    )
    return distance, success


class ReusableGpuScorer:
    def __init__(self, object_name, batch_size, battery_name):
        self.object_name = object_name
        self.batch_size = batch_size
        self.battery_name = battery_name
        data_urdf = DRO_ROOT / "data" / "data_urdf"
        object_type, object_id = object_name.split("+")
        hand = create_hand_model("inspire")
        self.sim = HoldoutValidator(
            robot_name="inspire",
            joint_orders=hand.get_joint_orders(),
            batch_size=batch_size,
            gpu=0,
            is_filter=False,
            use_gui=False,
            projection_mode="lsq",
            battery=BATTERIES[battery_name],
            use_gpu_pipeline=True,
        )
        self.sim.set_asset(
            robot_path=str(data_urdf / "robot"),
            robot_file="inspire/inspire_hand_right_extended.urdf",
            object_path=str(data_urdf / "object"),
            object_file=f"{object_type}/{object_id}/coacd_decomposed_object_one_link.urdf",
        )
        self.sim.create_envs()

    def score(self, q_batch):
        if q_batch.shape[0] != self.batch_size:
            return score_fresh(self.object_name, q_batch, self.battery_name, use_gpu_pipeline=True)
        self.sim.set_actor_pose_dof(q_batch.cpu())
        success, _, distance = self.sim.run_sim()
        return distance, success

    def destroy(self):
        self.sim.destroy()


def run_grasp_search(object_name, q0, u0, args, finger_names, lower, upper, rng, gpu_scorer=None, warm_start_params=None):
    init_std = args.init_std_scale * np.array([args.wrist_xyz_std] * 3 + [args.wrist_rot_std] * 3 + [args.u_std] * 6, dtype=np.float32)
    min_std = np.array([0.001] * 3 + [0.01] * 3 + [0.02] * 6, dtype=np.float32)
    base_mean = np.concatenate([q0[:6].numpy(), u0], axis=0).astype(np.float32)
    if warm_start_params is not None:
        base_mean = np.asarray(warm_start_params, dtype=np.float32).reshape(12)
    pass_rows = []
    best_row = None
    all_history = []
    u_lower, u_upper = active_limits(finger_names, lower, upper)

    for restart in range(args.restarts):
        mean = base_mean.copy()
        if restart > 0:
            mean = rng.normal(base_mean, init_std).astype(np.float32)
        std = init_std.copy()
        for generation in range(args.generations):
            if getattr(args, "spatial_sampling", "local") == "stratified":
                samples = sample_stratified(mean[None, :], std[None, :], args.population, rng)[0]
            else:
                samples = rng.normal(mean[None, :], std[None, :], size=(args.population, 12)).astype(np.float32)
                samples[0] = mean
            clamp_count, clamp_total, clamp_fraction = clamp_stats(samples, u_lower, u_upper)
            q_candidates = params_to_q(samples, finger_names, lower, upper)
            if args.search_pipeline == "gpu":
                if gpu_scorer is not None:
                    score, success = gpu_scorer.score(q_candidates)
                else:
                    score, success = score_fresh(object_name, q_candidates, args.selection_battery, use_gpu_pipeline=True)
                trusted_score = score.clone()
            else:
                score, success = score_fresh(object_name, q_candidates, args.selection_battery, use_gpu_pipeline=False)
                trusted_score = score
            order = torch.argsort(score)[: args.elites].cpu().numpy()
            elite = samples[order]
            mean = elite.mean(axis=0).astype(np.float32)
            std = np.maximum(elite.std(axis=0).astype(np.float32), min_std)

            for idx in torch.nonzero(success, as_tuple=False).flatten().tolist():
                pass_rows.append(
                    {
                        "restart": restart,
                        "generation": generation,
                        "score": float(trusted_score[idx]),
                        "rank_score": float(score[idx]),
                        "label_source": "gpu_rank_untrusted" if args.search_pipeline == "gpu" else "cpu_trusted",
                        "q": q_candidates[idx].clone(),
                        "params": torch.tensor(samples[idx], dtype=torch.float32),
                    }
                )

            gen_best = int(torch.argmin(score).item())
            current_best = {
                "restart": restart,
                "generation": generation,
                "score": float(trusted_score[gen_best]),
                "rank_score": float(score[gen_best]),
                "success": bool(success[gen_best]),
                "label_source": "gpu_rank_untrusted" if args.search_pipeline == "gpu" else "cpu_trusted",
                "q": q_candidates[gen_best].clone(),
                "params": torch.tensor(samples[gen_best], dtype=torch.float32),
            }
            if best_row is None or current_best["score"] < best_row["score"]:
                best_row = current_best
            all_history.append(
                {
                    "restart": restart,
                    "generation": generation,
                    "best_score": float(score[gen_best]),
                    "best_success": bool(success[gen_best]),
                    "mean_score": float(score.mean()),
                    "elite_mean_score": float(score[order].mean()),
                        "pass_count": int(success.sum()),
                        "clamp_sample_count": clamp_count,
                        "clamp_sample_total": clamp_total,
                        "clamp_sample_fraction": clamp_fraction,
                    }
            )
            if getattr(args, "spatial_sampling", "local") == "stratified":
                bands = layer_slices(args.population)
                all_history[-1].update({
                    "spatial_layers": LAYERS,
                    "spatial_layer_pass_counts": [int(success[band].sum()) for band in bands],
                    "sampled_params": torch.from_numpy(samples.copy()),
                    "gpu_success_mask": success.cpu().clone(),
                })
    return best_row, pass_rows, all_history


def run_object(row, args, finger_names, lower, upper):
    object_name = row["object_name"]
    q_seed, u_seed = lsq_project_q(row["q_dro_order"][: args.max_grasps], finger_names, lower, upper)
    source_indices = selected_source_indices(args, q_seed.shape[0])
    q_seed = q_seed[source_indices]
    u_seed = u_seed[source_indices]
    if args.search_pipeline == "gpu" and args.batch_source_grasps > 1:
        return run_object_batched_gpu(object_name, q_seed, u_seed, source_indices, args, finger_names, lower, upper)
    object_seed = int(hashlib.sha1(object_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.seed + object_seed % 100000)
    grasp_rows = []
    all_pass = []
    full_gpu_pass = []
    start_time = time.time()
    object_rng_seed = args.seed + object_seed % 100000
    gpu_scorer = ReusableGpuScorer(object_name, args.population, args.selection_battery) if args.search_pipeline == "gpu" and args.reuse_gpu_sim else None
    try:
        for local_source_idx, grasp_index in enumerate(source_indices):
            best_row, pass_rows, history = run_grasp_search(
                object_name,
                q_seed[local_source_idx],
                u_seed[local_source_idx],
                args,
                finger_names,
                lower,
                upper,
                rng,
                gpu_scorer=gpu_scorer,
                warm_start_params=warm_start_for(args, object_name, grasp_index),
            )
            maybe_print_warm_start(args, object_name, grasp_index, warm_start_for(args, object_name, grasp_index))
            pass_rows = sorted(pass_rows, key=lambda x: x["score"])
            raw_num_pass = len(pass_rows)
            for validation_order_idx, pass_row in enumerate(pass_rows):
                pass_row["_validation_order_idx"] = validation_order_idx
            if args.keep_per_grasp > 0:
                if args.persist_all_gpu_pass:
                    for truncated in pass_rows[args.keep_per_grasp:]:
                        full_gpu_pass.append(annotate_gpu_pass(truncated, object_name, grasp_index, args.arm_id, object_rng_seed, truncated["_validation_order_idx"], False, "truncated_by_keep_per_grasp"))
                pass_rows = pass_rows[: args.keep_per_grasp]
            for pass_row in pass_rows:
                pass_row["source_grasp_index"] = grasp_index
            all_pass.extend(pass_rows)
            grasp_rows.append(
                {
                    "grasp_index": grasp_index,
                    "q_seed": q_seed[local_source_idx],
                    "best": best_row,
                    "num_pass": len(pass_rows),
                    "raw_num_pass": raw_num_pass,
                    "history": history,
                }
            )
            print(f"{object_name} grasp {grasp_index}: best_score={best_row['score']:.4f} best_success={best_row['success']} pass_candidates={len(pass_rows)} raw_pass_candidates={raw_num_pass}", flush=True)
    finally:
        if gpu_scorer is not None:
            gpu_scorer.destroy()
    all_pass = sorted(all_pass, key=lambda x: x["score"])
    retained_before_cpu = list(all_pass)
    cpu_final_evaluated = 0
    cpu_final_rejected = 0
    cpu_final_rejected_candidates = []
    if args.search_pipeline == "gpu":
        if args.cpu_final_candidates > 0:
            all_pass = all_pass[: args.cpu_final_candidates]
            if args.persist_all_gpu_pass:
                selected_ids = {id(row) for row in all_pass}
                for row in retained_before_cpu:
                    if id(row) not in selected_ids:
                        full_gpu_pass.append(annotate_gpu_pass(row, object_name, row["source_grasp_index"], args.arm_id, object_rng_seed, row.get("_validation_order_idx", -1), False, "truncated_by_cpu_final_candidates"))
        if all_pass:
            q_final = torch.stack([row["q"] for row in all_pass], dim=0)
            cpu_score, cpu_success = score_fresh(object_name, q_final, args.selection_battery, use_gpu_pipeline=False)
            filtered = []
            for idx, pass_row in enumerate(all_pass):
                pass_row["gpu_rank_score"] = pass_row["rank_score"]
                pass_row["score"] = float(cpu_score[idx])
                pass_row["cpu_success"] = bool(cpu_success[idx])
                pass_row["label_source"] = "cpu_final_trusted"
                if bool(cpu_success[idx]):
                    filtered.append(pass_row)
                elif args.log_rejected:
                    pass_row["label_source"] = "cpu_final_rejected"
                    cpu_final_rejected_candidates.append(pass_row)
                if args.persist_all_gpu_pass:
                    full_gpu_pass.append(annotate_gpu_pass(pass_row, object_name, pass_row["source_grasp_index"], args.arm_id, object_rng_seed, pass_row.get("_validation_order_idx", idx), True, "cpu_final_success" if bool(cpu_success[idx]) else "cpu_final_rejected"))
            cpu_final_evaluated = len(all_pass)
            cpu_final_rejected = len(all_pass) - len(filtered)
            all_pass = sorted(filtered, key=lambda x: x["score"])
    if args.keep_per_object > 0:
        all_pass = all_pass[: args.keep_per_object]
    result = {
        "object_name": object_name,
        "q_seed": q_seed,
        "grasp_rows": grasp_rows,
        "passing_candidates": all_pass,
        "num_passing_candidates": len(all_pass),
        "elapsed_sec": time.time() - start_time,
        "cpu_final_evaluated": cpu_final_evaluated,
        "cpu_final_rejected": cpu_final_rejected,
        "cpu_final_rejected_candidates": cpu_final_rejected_candidates,
    }
    if args.persist_all_gpu_pass:
        result["all_gpu_pass_candidates"] = full_gpu_pass
        result["all_gpu_pass_count"] = len(full_gpu_pass)
    return result


def run_object_batched_gpu(object_name, q_seed, u_seed, source_indices, args, finger_names, lower, upper):
    object_seed = int(hashlib.sha1(object_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.seed + object_seed % 100000)
    init_std = args.init_std_scale * np.array([args.wrist_xyz_std] * 3 + [args.wrist_rot_std] * 3 + [args.u_std] * 6, dtype=np.float32)
    min_std = np.array([0.001] * 3 + [0.01] * 3 + [0.02] * 6, dtype=np.float32)
    u_lower, u_upper = active_limits(finger_names, lower, upper)
    grasp_rows = []
    all_pass = []
    full_gpu_pass = []
    start_time = time.time()
    object_rng_seed = args.seed + object_seed % 100000

    for chunk_start in range(0, q_seed.shape[0], args.batch_source_grasps):
        chunk_end = min(chunk_start + args.batch_source_grasps, q_seed.shape[0])
        chunk_size = chunk_end - chunk_start
        base_means = []
        chunk_source_indices = source_indices[chunk_start:chunk_end]
        for local_idx, grasp_index in enumerate(chunk_source_indices):
            warm_params = warm_start_for(args, object_name, grasp_index)
            if warm_params is None:
                selected_idx = chunk_start + local_idx
                base_means.append(np.concatenate([q_seed[selected_idx, :6].numpy(), u_seed[selected_idx]], axis=0).astype(np.float32))
            else:
                maybe_print_warm_start(args, object_name, grasp_index, warm_params)
                base_means.append(np.asarray(warm_params, dtype=np.float32).reshape(12))
        base_means = np.stack(base_means, axis=0)
        means = base_means.copy()
        stds = np.repeat(init_std[None, :], chunk_size, axis=0)
        chunk_pass_rows = [[] for _ in range(chunk_size)]
        best_rows = [None for _ in range(chunk_size)]
        histories = [[] for _ in range(chunk_size)]
        gpu_scorer = ReusableGpuScorer(object_name, chunk_size * args.population, args.selection_battery) if args.reuse_gpu_sim else None
        try:
            for restart in range(args.restarts):
                if restart > 0:
                    means = rng.normal(base_means, init_std[None, :]).astype(np.float32)
                    stds = np.repeat(init_std[None, :], chunk_size, axis=0)
                for generation in range(args.generations):
                    if getattr(args, "spatial_sampling", "local") == "stratified":
                        samples = sample_stratified(means, stds, args.population, rng)
                    else:
                        samples = rng.normal(
                            means[:, None, :],
                            stds[:, None, :],
                            size=(chunk_size, args.population, 12),
                        ).astype(np.float32)
                        samples[:, 0, :] = means
                    flat_samples = samples.reshape(chunk_size * args.population, 12)
                    q_candidates = params_to_q(flat_samples, finger_names, lower, upper)
                    if gpu_scorer is not None:
                        score_flat, success_flat = gpu_scorer.score(q_candidates)
                    else:
                        score_flat, success_flat = score_fresh(object_name, q_candidates, args.selection_battery, use_gpu_pipeline=True)
                    score = score_flat.reshape(chunk_size, args.population)
                    success = success_flat.reshape(chunk_size, args.population)
                    q_view = q_candidates.reshape(chunk_size, args.population, -1)
                    for local_idx in range(chunk_size):
                        clamp_count, clamp_total, clamp_fraction = clamp_stats(samples[local_idx], u_lower, u_upper)
                        order = torch.argsort(score[local_idx])[: args.elites].cpu().numpy()
                        elite = samples[local_idx, order]
                        means[local_idx] = elite.mean(axis=0).astype(np.float32)
                        stds[local_idx] = np.maximum(elite.std(axis=0).astype(np.float32), min_std)
                        for candidate_idx in torch.nonzero(success[local_idx], as_tuple=False).flatten().tolist():
                            chunk_pass_rows[local_idx].append(
                                {
                                    "restart": restart,
                                    "generation": generation,
                                    "score": float(score[local_idx, candidate_idx]),
                                    "rank_score": float(score[local_idx, candidate_idx]),
                                    "label_source": "gpu_rank_untrusted",
                                    "q": q_view[local_idx, candidate_idx].clone(),
                                    "params": torch.tensor(samples[local_idx, candidate_idx], dtype=torch.float32),
                                }
                            )
                        gen_best = int(torch.argmin(score[local_idx]).item())
                        current_best = {
                            "restart": restart,
                            "generation": generation,
                            "score": float(score[local_idx, gen_best]),
                            "rank_score": float(score[local_idx, gen_best]),
                            "success": bool(success[local_idx, gen_best]),
                            "label_source": "gpu_rank_untrusted",
                            "q": q_view[local_idx, gen_best].clone(),
                            "params": torch.tensor(samples[local_idx, gen_best], dtype=torch.float32),
                        }
                        if best_rows[local_idx] is None or current_best["score"] < best_rows[local_idx]["score"]:
                            best_rows[local_idx] = current_best
                        histories[local_idx].append(
                            {
                                "restart": restart,
                                "generation": generation,
                                "best_score": float(score[local_idx, gen_best]),
                                "best_success": bool(success[local_idx, gen_best]),
                                "mean_score": float(score[local_idx].mean()),
                                "elite_mean_score": float(score[local_idx, order].mean()),
                                "pass_count": int(success[local_idx].sum()),
                                "clamp_sample_count": clamp_count,
                                "clamp_sample_total": clamp_total,
                                "clamp_sample_fraction": clamp_fraction,
                            }
                        )
                        if getattr(args, "spatial_sampling", "local") == "stratified":
                            bands = layer_slices(args.population)
                            histories[local_idx][-1].update({
                                "spatial_layers": LAYERS,
                                "spatial_layer_pass_counts": [int(success[local_idx, band].sum())
                                                              for band in bands],
                                "sampled_params": torch.from_numpy(samples[local_idx].copy()),
                                "gpu_success_mask": success[local_idx].cpu().clone(),
                            })
        finally:
            if gpu_scorer is not None:
                gpu_scorer.destroy()

        for local_idx, grasp_index in enumerate(chunk_source_indices):
            selected_idx = chunk_start + local_idx
            pass_rows = sorted(chunk_pass_rows[local_idx], key=lambda x: x["score"])
            raw_num_pass = len(pass_rows)
            for validation_order_idx, pass_row in enumerate(pass_rows):
                pass_row["_validation_order_idx"] = validation_order_idx
            if args.keep_per_grasp > 0:
                if args.persist_all_gpu_pass:
                    for truncated in pass_rows[args.keep_per_grasp:]:
                        full_gpu_pass.append(annotate_gpu_pass(truncated, object_name, grasp_index, args.arm_id, object_rng_seed, truncated["_validation_order_idx"], False, "truncated_by_keep_per_grasp"))
                pass_rows = pass_rows[: args.keep_per_grasp]
            for pass_row in pass_rows:
                pass_row["source_grasp_index"] = grasp_index
            all_pass.extend(pass_rows)
            grasp_rows.append(
                {
                    "grasp_index": grasp_index,
                    "q_seed": q_seed[selected_idx],
                    "best": best_rows[local_idx],
                    "num_pass": len(pass_rows),
                    "raw_num_pass": raw_num_pass,
                    "history": histories[local_idx],
                }
            )
            print(
                f"{object_name} grasp {grasp_index}: best_score={best_rows[local_idx]['score']:.4f} "
                f"best_success={best_rows[local_idx]['success']} pass_candidates={len(pass_rows)} raw_pass_candidates={raw_num_pass}",
                flush=True,
            )

    all_pass = sorted(all_pass, key=lambda x: x["score"])
    retained_before_cpu = list(all_pass)
    cpu_final_evaluated = 0
    cpu_final_rejected = 0
    cpu_final_rejected_candidates = []
    if args.cpu_final_candidates > 0:
        all_pass = all_pass[: args.cpu_final_candidates]
        if args.persist_all_gpu_pass:
            selected_ids = {id(row) for row in all_pass}
            for row in retained_before_cpu:
                if id(row) not in selected_ids:
                    full_gpu_pass.append(annotate_gpu_pass(row, object_name, row["source_grasp_index"], args.arm_id, object_rng_seed, row.get("_validation_order_idx", -1), False, "truncated_by_cpu_final_candidates"))
    if all_pass:
        q_final = torch.stack([row["q"] for row in all_pass], dim=0)
        cpu_score, cpu_success = score_fresh(object_name, q_final, args.selection_battery, use_gpu_pipeline=False)
        filtered = []
        for idx, pass_row in enumerate(all_pass):
            pass_row["gpu_rank_score"] = pass_row["rank_score"]
            pass_row["score"] = float(cpu_score[idx])
            pass_row["cpu_success"] = bool(cpu_success[idx])
            pass_row["label_source"] = "cpu_final_trusted"
            if bool(cpu_success[idx]):
                filtered.append(pass_row)
            elif args.log_rejected:
                pass_row["label_source"] = "cpu_final_rejected"
                cpu_final_rejected_candidates.append(pass_row)
            if args.persist_all_gpu_pass:
                full_gpu_pass.append(annotate_gpu_pass(pass_row, object_name, pass_row["source_grasp_index"], args.arm_id, object_rng_seed, pass_row.get("_validation_order_idx", idx), True, "cpu_final_success" if bool(cpu_success[idx]) else "cpu_final_rejected"))
        cpu_final_evaluated = len(all_pass)
        cpu_final_rejected = len(all_pass) - len(filtered)
        all_pass = sorted(filtered, key=lambda x: x["score"])
    if args.keep_per_object > 0:
        all_pass = all_pass[: args.keep_per_object]
    result = {
        "object_name": object_name,
        "q_seed": q_seed,
        "grasp_rows": grasp_rows,
        "passing_candidates": all_pass,
        "num_passing_candidates": len(all_pass),
        "elapsed_sec": time.time() - start_time,
        "cpu_final_evaluated": cpu_final_evaluated,
        "cpu_final_rejected": cpu_final_rejected,
        "cpu_final_rejected_candidates": cpu_final_rejected_candidates,
        "batch_source_grasps": args.batch_source_grasps,
    }
    if args.persist_all_gpu_pass:
        result["all_gpu_pass_candidates"] = full_gpu_pass
        result["all_gpu_pass_count"] = len(full_gpu_pass)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-file", type=str, default="inspire_train40_zero_shot_b60.pt")
    parser.add_argument("--split", type=str, default="inspire_train_test_split.json")
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--allow-non-train", action="store_true")
    parser.add_argument("--max-grasps", type=int, default=60)
    parser.add_argument("--selection-battery", type=str, default="staged_standard_diag_rot12")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--keep-per-object", type=int, default=80)
    parser.add_argument("--keep-per-grasp", type=int, default=0)
    parser.add_argument("--search-pipeline", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--reuse-gpu-sim", action="store_true")
    parser.add_argument("--batch-source-grasps", type=int, default=1)
    parser.add_argument("--cpu-final-candidates", type=int, default=240)
    parser.add_argument("--log-rejected", action="store_true")
    parser.add_argument("--warm-start-file", type=str, default=None)
    parser.add_argument("--persist-all-gpu-pass", action="store_true")
    parser.add_argument("--arm-id", type=str, default="unknown")
    parser.add_argument("--source-indices", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--init-std-scale", type=float, default=1.0)
    parser.add_argument("--wrist-xyz-std", type=float, default=0.01)
    parser.add_argument("--wrist-rot-std", type=float, default=0.10)
    parser.add_argument("--u-std", type=float, default=0.25)
    parser.add_argument("--spatial-sampling", choices=("local", "stratified"), default="local")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    warm_start = load_warm_start_file(args.warm_start_file)
    args._warm_start_map = warm_start["map"] if warm_start is not None else None
    if warm_start is not None:
        print(f"loaded warm-start file {warm_start['path']} seeds={warm_start['count']}", flush=True)

    split = json.loads((ROOT / "eval" / args.split).read_text())
    train_objects = set(split["train"])
    requested = args.objects or split["train"]
    test_overlap = set(requested) - train_objects
    if test_overlap and not args.allow_non_train:
        raise ValueError(f"Requested objects are not in train split: {sorted(test_overlap)}")

    hand = create_hand_model("inspire", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)

    zero_rows = torch.load(ROOT / "eval" / args.zero_file, map_location="cpu")
    row_by_object = {row["object_name"]: row for row in zero_rows}
    missing = [name for name in requested if name not in row_by_object]
    if missing:
        raise ValueError(f"Missing zero-shot rows: {missing}")

    out_rows = [run_object(row_by_object[name], args, finger_names, lower, upper) for name in requested]
    total_pass = sum(row["num_passing_candidates"] for row in out_rows)
    total_sources = sum(len(row["grasp_rows"]) for row in out_rows)
    result = {
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "objects": requested,
        "total_source_grasps": total_sources,
        "total_passing_candidates": total_pass,
        "rows": out_rows,
    }
    out_arg = Path(args.out)
    if out_arg.parts and out_arg.parts[0] == "eval":
        out_arg = Path(*out_arg.parts[1:])
    out = ROOT / "eval" / out_arg
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"passing candidates {total_pass} from {total_sources} source grasps")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
