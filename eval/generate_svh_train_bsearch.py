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


ACTIVE_JOINTS = [
    "right_hand_Thumb_Flexion",
    "right_hand_Thumb_Opposition",
    "right_hand_Index_Finger_Distal",
    "right_hand_Index_Finger_Proximal",
    "right_hand_Middle_Finger_Proximal",
    "right_hand_Middle_Finger_Distal",
    "right_hand_Ring_Finger",
    "right_hand_Pinky",
    "right_hand_Finger_Spread",
]

COUPLING = {
    "right_hand_j5": {"driver": "right_hand_Thumb_Opposition", "multiplier": 1.0, "offset": 0.0},
    "right_hand_j3": {"driver": "right_hand_Thumb_Flexion", "multiplier": 1.01511, "offset": 0.0},
    "right_hand_j4": {"driver": "right_hand_Thumb_Flexion", "multiplier": 1.44889, "offset": 0.0},
    "right_hand_j14": {"driver": "right_hand_Index_Finger_Distal", "multiplier": 1.045, "offset": 0.0},
    "right_hand_j15": {"driver": "right_hand_Middle_Finger_Distal", "multiplier": 1.0454, "offset": 0.0},
    "right_hand_j12": {"driver": "right_hand_Ring_Finger", "multiplier": 1.3588, "offset": 0.0},
    "right_hand_j16": {"driver": "right_hand_Ring_Finger", "multiplier": 1.42093, "offset": 0.0},
    "right_hand_j13": {"driver": "right_hand_Pinky", "multiplier": 1.3588, "offset": 0.0},
    "right_hand_j17": {"driver": "right_hand_Pinky", "multiplier": 1.42307, "offset": 0.0},
    "right_hand_index_spread": {"driver": "right_hand_Finger_Spread", "multiplier": 0.5, "offset": 0.0},
    "right_hand_ring_spread": {"driver": "right_hand_Finger_Spread", "multiplier": 0.5, "offset": 0.0},
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


def active_limits(joint_names, lower, upper):
    name2idx = {name: i for i, name in enumerate(joint_names)}
    lo = [float(lower[name2idx[name]]) for name in ACTIVE_JOINTS]
    hi = [float(upper[name2idx[name]]) for name in ACTIVE_JOINTS]
    return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


def clamp_stats(samples, u_lower, u_upper):
    u = samples[..., 6:]
    out = np.logical_or(u < u_lower.reshape((1,) * (u.ndim - 1) + (9,)), u > u_upper.reshape((1,) * (u.ndim - 1) + (9,)))
    sample_out = np.any(out, axis=-1)
    return int(sample_out.sum()), int(sample_out.size), float(sample_out.mean()) if sample_out.size else 0.0


def forward_coupling(u9, joint_names, lower, upper):
    name2idx = {name: i for i, name in enumerate(joint_names)}
    q20 = np.zeros_like(lower, dtype=np.float32)
    values = {}
    for active_i, active_name in enumerate(ACTIVE_JOINTS):
        active_idx = name2idx[active_name]
        value = float(np.clip(u9[active_i], lower[active_idx], upper[active_idx]))
        values[active_name] = value
        q20[active_idx] = value
    for passive_name, spec in COUPLING.items():
        passive_idx = name2idx[passive_name]
        value = spec["multiplier"] * values[spec["driver"]] + spec["offset"]
        q20[passive_idx] = float(np.clip(value, lower[passive_idx], upper[passive_idx]))
    return q20


def lsq_project_q(q_batch, joint_names, lower, upper):
    name2idx = {name: i for i, name in enumerate(joint_names)}
    q_rows = []
    u_rows = []
    for q in q_batch.detach().cpu().numpy().astype(np.float32):
        q_finger = q[6:]
        u9 = []
        for active_name in ACTIVE_JOINTS:
            terms = [(1.0, 0.0, name2idx[active_name])]
            for passive_name, spec in COUPLING.items():
                if spec["driver"] == active_name:
                    terms.append((float(spec["multiplier"]), float(spec["offset"]), name2idx[passive_name]))
            numerator = sum(a * (float(q_finger[idx]) - b) for a, b, idx in terms)
            denominator = sum(a * a for a, _, _ in terms)
            active_idx = name2idx[active_name]
            u9.append(float(np.clip(numerator / denominator, lower[active_idx], upper[active_idx])))
        q20 = forward_coupling(np.asarray(u9, dtype=np.float32), joint_names, lower, upper)
        q_rows.append(np.concatenate([q[:6], q20], axis=0))
        u_rows.append(u9)
    return torch.tensor(np.stack(q_rows), dtype=torch.float32), np.asarray(u_rows, dtype=np.float32)


def params_to_q(params, joint_names, lower, upper):
    q_rows = []
    for row in params.astype(np.float32):
        wrist = row[:6]
        u9 = row[6:].copy()
        q20 = forward_coupling(u9, joint_names, lower, upper)
        q_rows.append(np.concatenate([wrist, q20], axis=0))
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
        params = torch.as_tensor(seed["params"], dtype=torch.float32).view(15).numpy().astype(np.float32)
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
    hand = create_hand_model("svh")
    sim = HoldoutValidator(
        robot_name="svh",
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
            robot_file="svh/svh_right_extended.urdf",
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
        hand = create_hand_model("svh")
        self.sim = HoldoutValidator(
            robot_name="svh",
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
            robot_file="svh/svh_right_extended.urdf",
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


def run_object_batched_gpu(object_name, q_seed, u_seed, source_indices, args, joint_names, lower, upper, warm_starts=None):
    object_seed = int(hashlib.sha1(object_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.seed + object_seed % 100000)
    init_std = args.init_std_scale * np.array([args.wrist_xyz_std] * 3 + [args.wrist_rot_std] * 3 + [args.u_std] * 9, dtype=np.float32)
    min_std = np.array([0.001] * 3 + [0.01] * 3 + [0.02] * 9, dtype=np.float32)
    u_lower, u_upper = active_limits(joint_names, lower, upper)
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
                        samples = rng.normal(means[:, None, :], stds[:, None, :], size=(chunk_size, args.population, 15)).astype(np.float32)
                        samples[:, 0, :] = means
                    flat_samples = samples.reshape(chunk_size * args.population, 15)
                    q_candidates = params_to_q(flat_samples, joint_names, lower, upper)
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


def run_object(row, args, joint_names, lower, upper, warm_starts=None):
    object_name = row["object_name"]
    q_seed, u_seed = lsq_project_q(row["q_dro_order"][: args.max_grasps], joint_names, lower, upper)
    source_indices = selected_source_indices(args, q_seed.shape[0])
    q_seed = q_seed[source_indices]
    u_seed = u_seed[source_indices]
    return run_object_batched_gpu(object_name, q_seed, u_seed, source_indices, args, joint_names, lower, upper, warm_starts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-file", type=str, default=str(ROOT / "results/pd_refiner_svh/svh_test10_zero_shot_b20.pt"))
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
    parser.add_argument("--out", type=str, default=str(ROOT / "results/pd_refiner_svh/bsearch_pilot/svh_test10_pilot_bsearch.pt"))
    args = parser.parse_args()
    if not args.log_rejected:
        print("warning: --log-rejected not set; rejected tensors will not be retained", flush=True)

    hand = create_hand_model("svh", device=torch.device("cpu"))
    joint_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)

    zero_rows = torch.load(args.zero_file, map_location="cpu")
    row_by_object = {row["object_name"]: row for row in zero_rows}
    missing = [name for name in args.objects if name not in row_by_object]
    if missing:
        raise ValueError(f"Missing zero-shot rows: {missing}")
    warm_starts = load_warm_start_file(args.warm_start_file)
    out_rows = [run_object(row_by_object[name], args, joint_names, lower, upper, warm_starts) for name in args.objects]
    total_pass = sum(row["num_passing_candidates"] for row in out_rows)
    total_sources = sum(len(row["grasp_rows"]) for row in out_rows)
    source_success = sum(len({int(c["source_grasp_index"]) for c in row["passing_candidates"]}) for row in out_rows)
    result = {
        "args": {k: v for k, v in vars(args).items()},
        "objects": args.objects,
        "robot_name": "svh",
        "params_dim": 15,
        "active_joints": ACTIVE_JOINTS,
        "joint_order": hand.get_joint_orders(),
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
