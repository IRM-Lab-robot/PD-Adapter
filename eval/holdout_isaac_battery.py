import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DRO_ROOT))

from isaacgym import gymapi
from isaacgym import gymtorch

from native_inspire_path_b import CoupledInspireValidator

import argparse
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import inspire_coupling
from utils.hand_model import create_hand_model


BATTERIES = {
    "standard_six_force05_steps6": {
        "directions": [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ],
        "force_multiplier": 1.0,
        "seconds": 6,
    },
    "diag8_force1_steps6": {
        "directions": [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        "force_multiplier": 2.0,
        "seconds": 6,
    },
    "diag8_force075_steps6": {
        "directions": [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        "force_multiplier": 1.5,
        "seconds": 6,
    },
    "rot45_force1_steps6": {
        "directions": None,
        "force_multiplier": 2.0,
        "seconds": 6,
    },
    "rot45_force075_steps6": {
        "directions": None,
        "force_multiplier": 1.5,
        "seconds": 6,
    },
    "diag8_force1_steps8": {
        "directions": [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        "force_multiplier": 2.0,
        "seconds": 8,
    },
}


def rotated_six_directions():
    theta = math.pi / 4.0
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    base = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    return (base @ rot.T).tolist()


BATTERIES["combined20_force05_steps6"] = {
    "directions": BATTERIES["standard_six_force05_steps6"]["directions"]
    + BATTERIES["diag8_force1_steps6"]["directions"]
    + rotated_six_directions(),
    "force_multiplier": 1.0,
    "seconds": 6,
}

BATTERIES["staged_standard_diag_rot12"] = {
    "stages": [
        {
            "directions": BATTERIES["standard_six_force05_steps6"]["directions"],
            "force_multiplier": 1.0,
            "seconds": 6,
        },
        {
            "directions": BATTERIES["diag8_force1_steps6"]["directions"],
            "force_multiplier": 1.5,
            "seconds": 3,
        },
        {
            "directions": rotated_six_directions(),
            "force_multiplier": 1.5,
            "seconds": 3,
        },
    ]
}


class HoldoutValidator(CoupledInspireValidator):
    def __init__(self, *args, battery, **kwargs):
        super().__init__(*args, **kwargs)
        self.battery = battery

    def run_sim(self):
        for _ in range(self.grasp_step):
            self.gym.simulate(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        start_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()

        stage_distances = []
        stages = self.battery.get("stages", [self.battery])
        for stage in stages:
            directions = stage["directions"]
            if directions is None:
                directions = rotated_six_directions()
            tensor_device = gymtorch.wrap_tensor(self._rigid_body_states).device
            directions = torch.tensor(directions, dtype=torch.float32, device=tensor_device)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            object_force = self.object_force.to(device=tensor_device)
            force_multiplier = float(stage["force_multiplier"])
            seconds = int(stage["seconds"])

            force_list = []
            for direction in directions:
                force_tensor = torch.zeros([len(self.envs), self.rigid_body_num, 3], device=tensor_device)
                force_tensor[:, 0, :] = force_multiplier * object_force[:, None] * direction[None, :]
                force_list.append(force_tensor)

            total_steps = self.steps_per_sec * seconds
            for step in range(total_steps):
                force_idx = min(step * len(force_list) // total_steps, len(force_list) - 1)
                self.gym.apply_rigid_body_force_tensors(
                    self.sim,
                    gymtorch.unwrap_tensor(force_list[force_idx]),
                    None,
                    gymapi.ENV_SPACE,
                )
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            stage_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()
            stage_distances.append((stage_pos - start_pos).norm(dim=-1))

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        end_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()
        distance = torch.stack(stage_distances, dim=0).max(dim=0).values if stage_distances else (end_pos - start_pos).norm(dim=-1)
        success = distance <= 0.02

        object_pose = gymtorch.wrap_tensor(self._rigid_body_states).clone()[::self.rigid_body_num, :7]
        object_transform = np.eye(4)[np.newaxis].repeat(self.batch_size, axis=0)
        object_transform[:, :3, 3] = object_pose[:, :3].cpu().numpy()
        object_transform[:, :3, :3] = Rotation.from_quat(object_pose[:, 3:7].cpu().numpy()).as_matrix()

        self.gym.refresh_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(self._dof_states).clone().reshape(len(self.envs), -1, 2)[:, :, 0]
        robot_transform = np.eye(4)[np.newaxis].repeat(self.batch_size, axis=0)
        robot_transform[:, :3, 3] = dof_states[:, :3].cpu().numpy()
        robot_transform[:, :3, :3] = Rotation.from_euler("XYZ", dof_states[:, 3:6].cpu().numpy()).as_matrix()
        robot_transform = np.linalg.inv(object_transform) @ robot_transform
        dof_states[:, :3] = torch.tensor(robot_transform[:, :3, 3], device=dof_states.device)
        dof_states[:, 3:6] = torch.tensor(
            Rotation.from_matrix(robot_transform[:, :3, :3]).as_euler("XYZ"),
            device=dof_states.device,
        )
        q_isaac = dof_states[:, self.isaac2urdf_order].to(torch.device("cpu"))
        return success.cpu().bool(), q_isaac, distance.cpu()


def run_holdout(object_name, q_batch, projection_mode, battery, use_gpu_pipeline=False):
    data_urdf = DRO_ROOT / "data" / "data_urdf"
    object_type, object_id = object_name.split("+")
    hand = create_hand_model("inspire")
    simulator = HoldoutValidator(
        robot_name="inspire",
        joint_orders=hand.get_joint_orders(),
        batch_size=q_batch.shape[0],
        gpu=0,
        is_filter=False,
        use_gui=False,
        projection_mode=projection_mode,
        battery=battery,
        use_gpu_pipeline=use_gpu_pipeline,
    )
    simulator.set_asset(
        robot_path=str(data_urdf / "robot"),
        robot_file="inspire/inspire_hand_right_extended.urdf",
        object_path=str(data_urdf / "object"),
        object_file=f"{object_type}/{object_id}/coacd_decomposed_object_one_link.urdf",
    )
    simulator.create_envs()
    simulator.set_actor_pose_dof(q_batch.cpu())
    success, q_isaac, distance = simulator.run_sim()
    simulator.destroy()
    return success, q_isaac, distance


def project_lsq(q_batch, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    rows = []
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
        rows.append(np.concatenate([q[:6], q12], axis=0))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def load_lines(zero_file, search_file=None, dfc_file=None, cem_file=None, objects=None):
    hand = create_hand_model("inspire", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)

    zero = torch.load(ROOT / "eval" / zero_file, map_location="cpu")
    if objects:
        wanted = set(objects)
        zero = [r for r in zero if r["object_name"] in wanted]
    dfc_by_object = {}
    search_by_object = {}
    cem_by_object = {}
    if dfc_file:
        dfc = torch.load(ROOT / "eval" / dfc_file, map_location="cpu")
        dfc_by_object = {r["object_name"]: r["q_optimized"] for r in dfc}
    if search_file:
        search = torch.load(ROOT / "eval" / search_file, map_location="cpu")
        search_by_object = {r["object_name"]: r["q_optimized"] for r in search}
    if cem_file:
        cem = torch.load(ROOT / "eval" / cem_file, map_location="cpu")
        cem_by_object = {r["object_name"]: r["q_selected"] for r in cem}

    rows = []
    for item in zero:
        object_name = item["object_name"]
        q_a = item["q_dro_order"].float()
        q_lsq = project_lsq(q_a, finger_names, lower, upper)
        rows.append(
            {
                "object_name": object_name,
                "A": {"q": q_a, "projection_mode": "none"},
                "B-lsq": {"q": q_lsq, "projection_mode": "lsq"},
            }
        )
        if object_name in dfc_by_object:
            rows[-1]["DFC-freewrist"] = {"q": dfc_by_object[object_name].float(), "projection_mode": "lsq"}
        if object_name in search_by_object:
            rows[-1]["B-search"] = {"q": search_by_object[object_name].float(), "projection_mode": "lsq"}
        if object_name in cem_by_object:
            rows[-1]["B-search-CEM"] = {"q": cem_by_object[object_name].float(), "projection_mode": "lsq"}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batteries", nargs="+", default=list(BATTERIES.keys()))
    parser.add_argument("--zero-file", type=str, default="inspire_zero_shot_grasps.pt")
    parser.add_argument("--search-file", type=str, default="inspire_native_random_search_k32_selected_for_validate.pt")
    parser.add_argument("--dfc-file", type=str, default="inspire_u6_dfc_freewrist_60.pt")
    parser.add_argument("--cem-file", type=str, default="")
    parser.add_argument("--objects", nargs="+", default=[])
    parser.add_argument("--lines", nargs="+", default=["A", "B-lsq", "DFC-freewrist", "B-search"])
    parser.add_argument("--out", type=str, default="inspire_holdout_battery_results.pt")
    parser.add_argument("--use-gpu-pipeline", action="store_true")
    args = parser.parse_args()

    dfc_file = args.dfc_file if "DFC-freewrist" in args.lines else None
    search_file = args.search_file if "B-search" in args.lines else None
    cem_file = args.cem_file if "B-search-CEM" in args.lines else None
    line_rows = load_lines(args.zero_file, search_file=search_file, dfc_file=dfc_file, cem_file=cem_file, objects=args.objects)
    results = {}
    for battery_name in args.batteries:
        battery = BATTERIES[battery_name]
        results[battery_name] = {}
        print(f"battery {battery_name}: {battery}")
        for line_name in args.lines:
            line_results = []
            for object_row in line_rows:
                object_name = object_row["object_name"]
                line = object_row[line_name]
                success, q_isaac, distance = run_holdout(
                    object_name,
                    line["q"],
                    projection_mode=line["projection_mode"],
                    battery=battery,
                    use_gpu_pipeline=args.use_gpu_pipeline,
                )
                line_results.append(
                    {
                        "object_name": object_name,
                        "success": success,
                        "q_isaac": q_isaac,
                        "distance": distance,
                    }
                )
                print(f"{battery_name} {line_name} {object_name}: {int(success.sum())}/{success.numel()}")
            total_success = sum(int(r["success"].sum()) for r in line_results)
            total = sum(r["success"].numel() for r in line_results)
            results[battery_name][line_name] = {
                "rows": line_results,
                "summary": {"success": total_success, "total": total, "rate": total_success / total},
            }
            print(f"{battery_name} {line_name}: {total_success}/{total} = {total_success / total:.4f}")

    out = ROOT / "eval" / args.out
    torch.save({"batteries": {k: BATTERIES[k] for k in args.batteries}, "results": results}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
