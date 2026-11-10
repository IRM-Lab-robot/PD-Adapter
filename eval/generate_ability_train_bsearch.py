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

from holdout_isaac_battery import BATTERIES, HoldoutValidator
from utils.hand_model import create_hand_model

import torch


ACTIVE_JOINTS = ["index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q1", "thumb_q2"]
COUPLING = {
    "index_q2": {"driver": "index_q1", "multiplier": 1.05851325, "offset": 0.0},
    "middle_q2": {"driver": "middle_q1", "multiplier": 1.05851325, "offset": 0.0},
    "ring_q2": {"driver": "ring_q1", "multiplier": 1.05851325, "offset": 0.0},
    "pinky_q2": {"driver": "pinky_q1", "multiplier": 1.05851325, "offset": 0.0},
}


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
    lo = [float(lower[name2finger[name]]) for name in ACTIVE_JOINTS]
    hi = [float(upper[name2finger[name]]) for name in ACTIVE_JOINTS]
    return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


def clamp_stats(samples, u_lower, u_upper):
    u = samples[..., 6:]
    out = np.logical_or(u < u_lower.reshape((1,) * (u.ndim - 1) + (6,)), u > u_upper.reshape((1,) * (u.ndim - 1) + (6,)))
    sample_out = np.any(out, axis=-1)
    return int(sample_out.sum()), int(sample_out.size), float(sample_out.mean()) if sample_out.size else 0.0


def forward_coupling(u6, name2finger, lower, upper):
    q10 = np.zeros_like(lower, dtype=np.float32)
    for value, name in zip(u6, ACTIVE_JOINTS):
        q10[name2finger[name]] = value
    for passive_name, spec in COUPLING.items():
        q10[name2finger[passive_name]] = spec["multiplier"] * q10[name2finger[spec["driver"]]] + spec["offset"]
    return np.clip(q10, lower, upper)


def lsq_project_q(q_batch, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q_rows = []
    u_rows = []
    for q in q_batch.detach().cpu().numpy().astype(np.float32):
        q_finger = q[6:]
        u6 = []
        for active_name in ACTIVE_JOINTS:
            terms = [(1.0, 0.0, name2finger[active_name])]
            for passive_name, spec in COUPLING.items():
                if spec["driver"] == active_name:
                    terms.append((float(spec["multiplier"]), float(spec["offset"]), name2finger[passive_name]))
            numerator = sum(a * (float(q_finger[idx]) - b) for a, b, idx in terms)
            denominator = sum(a * a for a, _, _ in terms)
            active_idx = name2finger[active_name]
            u6.append(float(np.clip(numerator / denominator, lower[active_idx], upper[active_idx])))
        q10 = forward_coupling(np.asarray(u6, dtype=np.float32), name2finger, lower, upper)
        q_rows.append(np.concatenate([q[:6], q10], axis=0))
        u_rows.append(u6)
    return torch.tensor(np.stack(q_rows), dtype=torch.float32), np.asarray(u_rows, dtype=np.float32)


def params_to_q(params, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q_rows = []
    for row in params.astype(np.float32):
        wrist = row[:6]
        u6 = row[6:].copy()
        for active_i, active_name in enumerate(ACTIVE_JOINTS):
            active_idx = name2finger[active_name]
            u6[active_i] = np.clip(u6[active_i], lower[active_idx], upper[active_idx])
        q10 = forward_coupling(u6, name2finger, lower, upper)
        q_rows.append(np.concatenate([wrist, q10], axis=0))
    return torch.tensor(np.stack(q_rows), dtype=torch.float32)


def load_warm_start_file(path):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    seeds = payload["seeds"] if isinstance(payload, dict) and "seeds" in payload else payload
    out = {}
    for seed in seeds:
        object_name = str(seed["object_name"])
        source_idx = int(seed["source_grasp_index"])
        params = torch.as_tensor(seed["params"], dtype=torch.float32).view(12).numpy().astype(np.float32)
        out[(object_name, source_idx)] = params
    return out


def warm_start_means_for_object(object_name, source_indices, default_means, warm_starts):
    if warm_starts is None:
        return default_means
    rows = []
    missing = []
    for local_idx, source_idx in enumerate(source_indices):
        key = (object_name, int(source_idx))
        if key not in warm_starts:
            missing.append(key)
            rows.append(default_means[local_idx])
        else:
            rows.append(warm_starts[key])
    if missing:
        raise ValueError(f"Missing warm-start seeds for {missing[:5]} (total {len(missing)})")
    return np.stack(rows, axis=0).astype(np.float32)


def score_fresh(object_name, q_batch, battery_name, use_gpu_pipeline=False, projection_mode="none"):
    data_urdf = DRO_ROOT / "data" / "data_urdf"
    object_type, object_id = object_name.split("+")
    hand = create_hand_model("ability")
    sim = HoldoutValidator(
        robot_name="ability",
        joint_orders=hand.get_joint_orders(),
        batch_size=q_batch.shape[0],
        gpu=0,
        is_filter=False,
        use_gui=False,
        projection_mode=projection_mode,
        battery=BATTERIES[battery_name],
        use_gpu_pipeline=use_gpu_pipeline,
    )
    try:
        sim.set_asset(
            robot_path=str(data_urdf / "robot"),
            robot_file="ability/ability_hand_right_large_no_fsr_extended.urdf",
            object_path=str(data_urdf / "object"),
            object_file=f"{object_type}/{object_id}/coacd_decomposed_object_one_link.urdf",
        )
        sim.create_envs()
        sim.set_actor_pose_dof(q_batch.cpu())
        success, _, distance = sim.run_sim()
        return distance, success
    finally:
        sim.destroy()


class ReusableGpuScorer:
    def __init__(self, object_name, batch_size, battery_name, projection_mode="none"):
        self.object_name = object_name
        self.batch_size = batch_size
        self.battery_name = battery_name
        self.projection_mode = projection_mode
        data_urdf = DRO_ROOT / "data" / "data_urdf"
        object_type, object_id = object_name.split("+")
        hand = create_hand_model("ability")
        self.sim = HoldoutValidator(
            robot_name="ability",
            joint_orders=hand.get_joint_orders(),
            batch_size=batch_size,
            gpu=0,
            is_filter=False,
            use_gui=False,
            projection_mode=projection_mode,
            battery=BATTERIES[battery_name],
            use_gpu_pipeline=True,
        )
        self.sim.set_asset(
            robot_path=str(data_urdf / "robot"),
            robot_file="ability/ability_hand_right_large_no_fsr_extended.urdf",
            object_path=str(data_urdf / "object"),
            object_file=f"{object_type}/{object_id}/coacd_decomposed_object_one_link.urdf",
        )
        self.sim.create_envs()

    def score(self, q_batch):
        if q_batch.shape[0] != self.batch_size:
            return score_fresh(self.object_name, q_batch, self.battery_name, use_gpu_pipeline=True,
                               projection_mode=self.projection_mode)
        self.sim.set_actor_pose_dof(q_batch.cpu())
        success, _, distance = self.sim.run_sim()
        return distance, success

    def destroy(self):
        self.sim.destroy()


def run_object_batched_gpu(object_name, q_seed, u_seed, source_indices, args, finger_names, lower, upper, warm_starts=None):
    object_seed = int(hashlib.sha1(object_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.seed + object_seed % 100000)
    init_std = args.init_std_scale * np.array([args.wrist_xyz_std] * 3 + [args.wrist_rot_std] * 3 + [args.u_std] * 6, dtype=np.float32)
    min_std = np.array([0.001] * 3 + [0.01] * 3 + [0.02] * 6, dtype=np.float32)
    u_lower, u_upper = active_limits(finger_names, lower, upper)
    grasp_rows = []
    all_pass = []
    start_time = time.time()

    for chunk_start in range(0, q_seed.shape[0], args.batch_source_grasps):
        chunk_end = min(chunk_start + args.batch_source_grasps, q_seed.shape[0])
        chunk_size = chunk_end - chunk_start
        chunk_source_indices = source_indices[chunk_start:chunk_end]
        default_means = np.stack([np.concatenate([q_seed[chunk_start + i, :6].numpy(), u_seed[chunk_start + i]], axis=0).astype(np.float32) for i in range(chunk_size)], axis=0)
        base_means = warm_start_means_for_object(object_name, chunk_source_indices, default_means, warm_starts)
        means = base_means.copy()
        stds = np.repeat(init_std[None, :], chunk_size, axis=0)
        chunk_pass_rows = [[] for _ in range(chunk_size)]
        best_rows = [None for _ in range(chunk_size)]
        histories = [[] for _ in range(chunk_size)]
        gpu_scorer = ReusableGpuScorer(object_name, chunk_size * args.population, args.selection_battery,
                                      projection_mode=getattr(args, "validator_projection", "none")) if args.reuse_gpu_sim else None
        try:
            for restart in range(args.restarts):
                if restart > 0:
                    means = rng.normal(base_means, init_std[None, :]).astype(np.float32)
                    stds = np.repeat(init_std[None, :], chunk_size, axis=0)
                for generation in range(args.generations):
                    if getattr(args, "spatial_sampling", "local") == "stratified":
                        samples = sample_stratified(means, stds, args.population, rng)
                    else:
                        samples = rng.normal(means[:, None, :], stds[:, None, :], size=(chunk_size, args.population, 12)).astype(np.float32)
                        samples[:, 0, :] = means
                    flat_samples = samples.reshape(chunk_size * args.population, 12)
                    q_candidates = params_to_q(flat_samples, finger_names, lower, upper)
                    if gpu_scorer is not None:
                        score_flat, success_flat = gpu_scorer.score(q_candidates)
                    else:
                        score_flat, success_flat = score_fresh(
                            object_name, q_candidates, args.selection_battery, use_gpu_pipeline=True,
                            projection_mode=getattr(args, "validator_projection", "none"))
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
                            chunk_pass_rows[local_idx].append({
                                "restart": restart,
                                "generation": generation,
                                "score": float(score[local_idx, candidate_idx]),
                                "rank_score": float(score[local_idx, candidate_idx]),
                                "label_source": "gpu_rank_untrusted",
                                "q": q_view[local_idx, candidate_idx].clone(),
                                "params": torch.tensor(samples[local_idx, candidate_idx], dtype=torch.float32),
                            })
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
                        histories[local_idx].append({
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
                        })
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
            if args.keep_per_grasp > 0:
                pass_rows = pass_rows[: args.keep_per_grasp]
            for pass_row in pass_rows:
                pass_row["source_grasp_index"] = grasp_index
            all_pass.extend(pass_rows)
            grasp_rows.append({"grasp_index": grasp_index, "q_seed": q_seed[selected_idx], "best": best_rows[local_idx], "num_pass": len(pass_rows), "raw_num_pass": raw_num_pass, "history": histories[local_idx]})
            print(f"{object_name} grasp {grasp_index}: best_score={best_rows[local_idx]['score']:.4f} best_success={best_rows[local_idx]['success']} pass_candidates={len(pass_rows)} raw_pass_candidates={raw_num_pass}", flush=True)

    all_pass = sorted(all_pass, key=lambda x: x["score"])
    cpu_final_evaluated = 0
    cpu_final_rejected = 0
    cpu_final_rejected_candidates = []
    if args.cpu_final_candidates > 0:
        all_pass = all_pass[: args.cpu_final_candidates]
    if all_pass:
        q_final = torch.stack([row["q"] for row in all_pass], dim=0)
        cpu_score, cpu_success = score_fresh(
            object_name, q_final, args.selection_battery, use_gpu_pipeline=False,
            projection_mode=getattr(args, "validator_projection", "none"))
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
        cpu_final_evaluated = len(all_pass)
        cpu_final_rejected = len(all_pass) - len(filtered)
        all_pass = sorted(filtered, key=lambda x: x["score"])
    if args.keep_per_object > 0:
        all_pass = all_pass[: args.keep_per_object]
    return {
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


def run_object(row, args, finger_names, lower, upper, warm_starts=None):
    object_name = row["object_name"]
    q_seed, u_seed = lsq_project_q(row["q_dro_order"][: args.max_grasps], finger_names, lower, upper)
    source_indices = selected_source_indices(args, q_seed.shape[0])
    q_seed = q_seed[source_indices]
    u_seed = u_seed[source_indices]
    return run_object_batched_gpu(object_name, q_seed, u_seed, source_indices, args, finger_names, lower, upper, warm_starts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-file", type=str, default=str(ROOT / "results/pd_refiner_ability/ability_test10_zero_shot_b20.pt"))
    parser.add_argument("--objects", nargs="+", default=["ycb+baseball", "contactdb+mug", "contactdb+banana"])
    parser.add_argument("--max-grasps", type=int, default=20)
    parser.add_argument("--selection-battery", type=str, default="standard_six_force05_steps6")
    parser.add_argument("--validator-projection", choices=("none", "lsq"), default="none",
                        help="Project controller pre-open/close targets in the validator; legacy default is none")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--keep-per-object", type=int, default=120)
    parser.add_argument("--keep-per-grasp", type=int, default=3)
    parser.add_argument("--search-pipeline", choices=["gpu"], default="gpu")
    parser.add_argument("--reuse-gpu-sim", action="store_true")
    parser.add_argument("--batch-source-grasps", type=int, default=8)
    parser.add_argument("--cpu-final-candidates", type=int, default=240)
    parser.add_argument("--log-rejected", action="store_true")
    parser.add_argument("--warm-start-file", type=str, default=None)
    parser.add_argument("--source-indices", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--init-std-scale", type=float, default=1.0)
    parser.add_argument("--wrist-xyz-std", type=float, default=0.01)
    parser.add_argument("--wrist-rot-std", type=float, default=0.10)
    parser.add_argument("--u-std", type=float, default=0.25)
    parser.add_argument("--spatial-sampling", choices=("local", "stratified"), default="local")
    parser.add_argument("--out", type=str, default=str(ROOT / "results/pd_refiner_ability/bsearch_pilot/ability_test10_pilot_bsearch.pt"))
    args = parser.parse_args()
    if not args.log_rejected:
        print("warning: --log-rejected not set; rejected tensors will not be retained", flush=True)

    hand = create_hand_model("ability", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)

    zero_rows = torch.load(args.zero_file, map_location="cpu")
    row_by_object = {row["object_name"]: row for row in zero_rows}
    missing = [name for name in args.objects if name not in row_by_object]
    if missing:
        raise ValueError(f"Missing zero-shot rows: {missing}")

    warm_starts = load_warm_start_file(args.warm_start_file)
    out_rows = [run_object(row_by_object[name], args, finger_names, lower, upper, warm_starts) for name in args.objects]
    total_pass = sum(row["num_passing_candidates"] for row in out_rows)
    total_sources = sum(len(row["grasp_rows"]) for row in out_rows)
    source_success = sum(len({int(c["source_grasp_index"]) for c in row["passing_candidates"]}) for row in out_rows)
    result = {
        "args": {k: v for k, v in vars(args).items()},
        "objects": args.objects,
        "total_source_grasps": total_sources,
        "total_passing_candidates": total_pass,
        "source_success_count": source_success,
        "source_success_rate": float(source_success / total_sources) if total_sources else 0.0,
        "rows": out_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(json.dumps({"saved": str(out), "total_pass": total_pass, "total_sources": total_sources, "source_success_rate": result["source_success_rate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
