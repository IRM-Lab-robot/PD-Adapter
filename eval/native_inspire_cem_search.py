import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DRO_ROOT))

from holdout_isaac_battery import BATTERIES, HoldoutValidator, run_holdout

import numpy as np
import torch

import inspire_coupling
from utils.hand_model import create_hand_model


def lsq_project_q(q_batch, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q_rows = []
    u_rows = []
    for q in q_batch.detach().cpu().numpy().astype(np.float32):
        q_finger = q[6:]
        u6 = []
        for active_name in inspire_coupling.active_joints:
            terms = [(1.0, 0.0, name2finger[active_name])]
            for passive_name, spec in inspire_coupling.coupling.items():
                if spec["driver"] == active_name:
                    terms.append((float(spec["multiplier"]), float(spec["offset"]), name2finger[passive_name]))
            numerator = sum(a * (float(q_finger[idx]) - b) for a, b, idx in terms)
            denominator = sum(a * a for a, _, _ in terms)
            active_idx = name2finger[active_name]
            u6.append(float(np.clip(numerator / denominator, lower[active_idx], upper[active_idx])))
        q12 = inspire_coupling.forward_coupling(np.asarray(u6, dtype=np.float32), name2finger, lower, upper)
        q_rows.append(np.concatenate([q[:6], q12], axis=0))
        u_rows.append(u6)
    return torch.tensor(np.stack(q_rows), dtype=torch.float32), np.asarray(u_rows, dtype=np.float32)


def params_to_q(params, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q_rows = []
    for row in params.astype(np.float32):
        wrist = row[:6]
        u6 = row[6:]
        for active_i, active_name in enumerate(inspire_coupling.active_joints):
            active_idx = name2finger[active_name]
            u6[active_i] = np.clip(u6[active_i], lower[active_idx], upper[active_idx])
        q12 = inspire_coupling.forward_coupling(u6, name2finger, lower, upper)
        q_rows.append(np.concatenate([wrist, q12], axis=0))
    return torch.tensor(np.stack(q_rows), dtype=torch.float32)


class ReusableBatteryEvaluator:
    def __init__(self, object_name, population, battery_names):
        self.object_name = object_name
        self.population = population
        self.battery_names = battery_names
        self.hand = create_hand_model("inspire")
        data_urdf = DRO_ROOT / "data" / "data_urdf"
        object_type, object_id = object_name.split("+")
        first_battery = BATTERIES[battery_names[0]]
        self.sim = HoldoutValidator(
            robot_name="inspire",
            joint_orders=self.hand.get_joint_orders(),
            batch_size=population,
            gpu=0,
            is_filter=False,
            use_gui=False,
            projection_mode="lsq",
            battery=first_battery,
        )
        self.sim.set_asset(
            robot_path=str(data_urdf / "robot"),
            robot_file="inspire/inspire_hand_right_extended.urdf",
            object_path=str(data_urdf / "object"),
            object_file=f"{object_type}/{object_id}/coacd_decomposed_object_one_link.urdf",
        )
        self.sim.create_envs()

    def score(self, q_batch):
        distances = []
        successes = []
        for battery_name in self.battery_names:
            self.sim.battery = BATTERIES[battery_name]
            self.sim.set_actor_pose_dof(q_batch.cpu())
            success, _, distance = self.sim.run_sim()
            distances.append(distance)
            successes.append(success)
        distance_stack = torch.stack(distances, dim=0)
        success_stack = torch.stack(successes, dim=0)
        return distance_stack.max(dim=0).values, success_stack.all(dim=0), distance_stack

    def destroy(self):
        self.sim.destroy()


class FreshBatteryEvaluator:
    def __init__(self, object_name, population, battery_names):
        self.object_name = object_name
        self.population = population
        self.battery_names = battery_names

    def score(self, q_batch):
        distances = []
        successes = []
        for battery_name in self.battery_names:
            success, _, distance = run_holdout(self.object_name, q_batch.cpu(), "lsq", BATTERIES[battery_name])
            distances.append(distance)
            successes.append(success)
        distance_stack = torch.stack(distances, dim=0)
        success_stack = torch.stack(successes, dim=0)
        return distance_stack.max(dim=0).values, success_stack.all(dim=0), distance_stack

    def destroy(self):
        return None


def run_object_cem(object_row, args, finger_names, lower, upper):
    object_name = object_row["object_name"]
    q_lsq, u_lsq = lsq_project_q(object_row["q_dro_order"], finger_names, lower, upper)
    evaluator_cls = FreshBatteryEvaluator if args.evaluator == "fresh" else ReusableBatteryEvaluator
    evaluator = evaluator_cls(object_name, args.population, args.selection_batteries)
    rng = np.random.default_rng(args.seed)
    init_std = np.array([args.wrist_xyz_std] * 3 + [args.wrist_rot_std] * 3 + [args.u_std] * 6, dtype=np.float32)
    min_std = np.array([0.001] * 3 + [0.01] * 3 + [0.02] * 6, dtype=np.float32)
    rows = []
    try:
        for grasp_idx in range(q_lsq.shape[0]):
            mean = np.concatenate([q_lsq[grasp_idx, :6].numpy(), u_lsq[grasp_idx]], axis=0).astype(np.float32)
            std = init_std.copy()
            best_score = float("inf")
            best_q = None
            best_success = False
            history = []
            for gen in range(args.generations):
                samples = rng.normal(mean[None, :], std[None, :], size=(args.population, 12)).astype(np.float32)
                samples[0] = mean
                q_candidates = params_to_q(samples, finger_names, lower, upper)
                score, success, distance_stack = evaluator.score(q_candidates)
                order = torch.argsort(score)[: args.elites].cpu().numpy()
                elite = samples[order]
                mean = elite.mean(axis=0).astype(np.float32)
                std = np.maximum(elite.std(axis=0).astype(np.float32), min_std)
                gen_best = int(torch.argmin(score).item())
                if float(score[gen_best]) < best_score:
                    best_score = float(score[gen_best])
                    best_q = q_candidates[gen_best].clone()
                    best_success = bool(success[gen_best])
                history.append(
                    {
                        "generation": gen,
                        "best_score": float(score[gen_best]),
                        "best_success": bool(success[gen_best]),
                        "mean_score": float(score.mean()),
                        "elite_mean_score": float(score[order].mean()),
                    }
                )
            rows.append(
                {
                    "grasp_index": grasp_idx,
                    "q_lsq": q_lsq[grasp_idx],
                    "q_selected": best_q,
                    "best_score": best_score,
                    "best_success": best_success,
                    "history": history,
                }
            )
            print(f"{object_name} grasp {grasp_idx}: best_score={best_score:.4f} success={best_success}")
    finally:
        evaluator.destroy()
    q_selected = torch.stack([r["q_selected"] for r in rows], dim=0)
    return {
        "object_name": object_name,
        "q_lsq": q_lsq,
        "q_selected": q_selected,
        "rows": rows,
        "args": vars(args),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="inspire_shape_scale15_zero_shot_grasps.pt")
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--selection-batteries", nargs="+", default=["diag8_force1_steps6", "rot45_force1_steps6"])
    parser.add_argument("--evaluator", choices=["fresh", "reusable"], default="fresh")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--wrist-xyz-std", type=float, default=0.005)
    parser.add_argument("--wrist-rot-std", type=float, default=0.05)
    parser.add_argument("--u-std", type=float, default=0.15)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    hand = create_hand_model("inspire", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)

    data = torch.load(ROOT / "eval" / args.input, map_location="cpu")
    wanted = set(args.objects)
    data = [row for row in data if row["object_name"] in wanted]
    out_rows = [run_object_cem(row, args, finger_names, lower, upper) for row in data]
    out = ROOT / "eval" / args.out
    torch.save(out_rows, out)
    total_success = sum(sum(1 for r in row["rows"] if r["best_success"]) for row in out_rows)
    total = sum(len(row["rows"]) for row in out_rows)
    print(f"CEM selection success {total_success}/{total} = {total_success / total:.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
