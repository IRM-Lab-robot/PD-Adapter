import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DRO_ROOT))

from validation.isaac_validator import IsaacValidator

import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
from utils.controller import controller
from utils.hand_model import create_hand_model
import inspire_coupling
from command_target_projection import project_finger_targets_lsq


class CoupledInspireValidator(IsaacValidator):
    def __init__(self, *args, projection_mode="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_mode = projection_mode
        self.residual_rows = []
        self._target_lsq_limits = None

    def _project_finger_crude(self, q_finger, name2finger, lower, upper):
        u6 = np.asarray(
            [q_finger[name2finger[name]] for name in inspire_coupling.active_joints],
            dtype=np.float32,
        )
        return inspire_coupling.forward_coupling(u6, name2finger, lower, upper)

    def _project_finger_lsq(self, q_finger, name2finger, lower, upper):
        u6 = []
        for active_name in inspire_coupling.active_joints:
            terms = [(1.0, 0.0, name2finger[active_name])]
            for passive_name, spec in inspire_coupling.coupling.items():
                if spec["driver"] == active_name:
                    terms.append((float(spec["multiplier"]), float(spec["offset"]), name2finger[passive_name]))

            numerator = sum(a * (float(q_finger[idx]) - b) for a, b, idx in terms)
            denominator = sum(a * a for a, _, _ in terms)
            value = numerator / denominator
            active_idx = name2finger[active_name]
            value = float(np.clip(value, lower[active_idx], upper[active_idx]))
            u6.append(value)

        return inspire_coupling.forward_coupling(np.asarray(u6, dtype=np.float32), name2finger, lower, upper)

    def _project_finger(self, q_finger, name2finger, lower, upper):
        if self.projection_mode == "crude":
            return self._project_finger_crude(q_finger, name2finger, lower, upper)
        if self.projection_mode == "lsq":
            return self._project_finger_lsq(q_finger, name2finger, lower, upper)
        return q_finger.copy()

    def _project_inspire_batch(self, q_batch):
        if self.robot_name not in ["inspire", "inspire_left", "inspire_left_v2"] or self.projection_mode == "none":
            return q_batch

        q_out = q_batch.clone()
        joint_orders = self.joint_orders
        finger_names = joint_orders[6:]
        name2finger = {name: i for i, name in enumerate(finger_names)}

        lower, upper = create_hand_model(self.robot_name, device=q_batch.device).pk_chain.get_joint_limits()
        lower = np.asarray(lower[6:], dtype=np.float32)
        upper = np.asarray(upper[6:], dtype=np.float32)

        for batch_idx in range(q_batch.shape[0]):
            q_finger = q_batch[batch_idx, 6:].detach().cpu().numpy().astype(np.float32)
            projected = self._project_finger(q_finger, name2finger, lower, upper)
            q_out[batch_idx, 6:] = torch.tensor(projected, dtype=q_batch.dtype, device=q_batch.device)

        return q_out

    def _project_hand_batch(self, q_batch):
        if self.robot_name in ("inspire", "inspire_left", "inspire_left_v2"):
            return self._project_inspire_batch(q_batch)
        if self.projection_mode == "none":
            return q_batch
        if self.projection_mode != "lsq" or self.robot_name not in ("ability", "svh"):
            raise ValueError(f"Unsupported target projection: {self.robot_name}/{self.projection_mode}")
        if self._target_lsq_limits is None:
            hand = create_hand_model(self.robot_name, device=torch.device("cpu"))
            lower, upper = hand.pk_chain.get_joint_limits()
            self._target_lsq_limits = (np.asarray(lower[6:], dtype=np.float32),
                                       np.asarray(upper[6:], dtype=np.float32))
        lower, upper = self._target_lsq_limits
        finger_names = self.joint_orders[6:]
        if q_batch.ndim != 2 or q_batch.shape[1] != 6 + len(finger_names):
            raise ValueError(f"Invalid {self.robot_name} controller target shape: {tuple(q_batch.shape)}")
        projected = project_finger_targets_lsq(self.robot_name,
                                               q_batch[:, 6:].detach().cpu().numpy(),
                                               finger_names, lower, upper)
        out = q_batch.clone()
        out[:, 6:] = torch.as_tensor(projected, dtype=out.dtype, device=out.device)
        return out

    def coupling_residual(self, q_batch, mode="crude"):
        joint_orders = self.joint_orders
        finger_names = joint_orders[6:]
        name2finger = {name: i for i, name in enumerate(finger_names)}
        lower, upper = create_hand_model(self.robot_name, device=q_batch.device).pk_chain.get_joint_limits()
        lower = np.asarray(lower[6:], dtype=np.float32)
        upper = np.asarray(upper[6:], dtype=np.float32)
        rows = []
        for batch_idx in range(q_batch.shape[0]):
            q_finger = q_batch[batch_idx, 6:].detach().cpu().numpy().astype(np.float32)
            if mode == "lsq":
                projected = self._project_finger_lsq(q_finger, name2finger, lower, upper)
            else:
                projected = self._project_finger_crude(q_finger, name2finger, lower, upper)
            passive_diffs = {}
            passive_diffs_raw = []
            for passive_name, spec in inspire_coupling.coupling.items():
                passive_idx = name2finger[passive_name]
                expected = projected[passive_idx]
                diff = float(q_finger[passive_idx] - expected)
                passive_diffs[passive_name] = diff
                passive_diffs_raw.append(diff)
            diffs12 = q_finger - projected
            passive_diffs_raw = np.array(passive_diffs_raw, dtype=np.float32)
            rows.append(
                {
                    "l2": float(np.linalg.norm(diffs12)),
                    "mean_abs": float(np.mean(np.abs(diffs12))),
                    "max_abs": float(np.max(np.abs(diffs12))),
                    "passive_l2": float(np.linalg.norm(passive_diffs_raw)),
                    "passive_mean_abs": float(np.mean(np.abs(passive_diffs_raw))),
                    "passive_max_abs": float(np.max(np.abs(passive_diffs_raw))),
                    "per_joint": passive_diffs,
                }
            )
        return rows

    def set_actor_pose_dof(self, q):
        self.gym.prepare_sim(self.sim)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(_root_state)
        root_state[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device=root_state.device)
        self.gym.set_actor_root_state_tensor(self.sim, _root_state)

        outer_q, inner_q = controller(self.robot_name, q)

        if self.robot_name in ["inspire", "inspire_left", "inspire_left_v2"]:
            residual_mode = "lsq" if self.projection_mode == "lsq" else "crude"
            self.residual_rows = self.coupling_residual(inner_q, mode=residual_mode)

        outer_q = self._project_hand_batch(outer_q)
        inner_q = self._project_hand_batch(inner_q)

        if self.use_gpu_pipeline:
            self.gym.refresh_dof_state_tensor(self.sim)
            dof_states = gymtorch.wrap_tensor(self._dof_states).view(len(self.envs), -1, 2)
            outer_q = outer_q.to(device=dof_states.device, dtype=dof_states.dtype)
            inner_q = inner_q.to(device=dof_states.device, dtype=dof_states.dtype)
            dof_states[:, :, 0] = outer_q[:, self.urdf2isaac_order]
            dof_states[:, :, 1] = 0
            self.gym.set_dof_state_tensor(self.sim, self._dof_states)
            targets = inner_q[:, self.urdf2isaac_order].contiguous()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets.view(-1)))
            return

        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            robot_handle = self.robot_handles[env_idx]

            dof_states_initial = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_initial["pos"] = outer_q[env_idx, self.urdf2isaac_order]
            dof_states_initial["vel"] = 0
            self.gym.set_actor_dof_states(env, robot_handle, dof_states_initial, gymapi.STATE_ALL)

            dof_states_target = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_target["pos"] = inner_q[env_idx, self.urdf2isaac_order]
            dof_states_target["vel"] = 0
            self.gym.set_actor_dof_position_targets(env, robot_handle, dof_states_target["pos"])


def run_native_inspire(object_name, q_batch, projection_mode):
    data_urdf = DRO_ROOT / "data" / "data_urdf"
    object_type, object_id = object_name.split("+")
    hand = create_hand_model("inspire")
    simulator = CoupledInspireValidator(
        robot_name="inspire",
        joint_orders=hand.get_joint_orders(),
        batch_size=q_batch.shape[0],
        gpu=0,
        is_filter=False,
        use_gui=False,
        projection_mode=projection_mode,
    )
    simulator.set_asset(
        robot_path=str(data_urdf / "robot"),
        robot_file="inspire/inspire_hand_right_extended.urdf",
        object_path=str(data_urdf / "object"),
        object_file=f"{object_type}/{object_id}/coacd_decomposed_object_one_link.urdf",
    )
    simulator.create_envs()
    simulator.set_actor_pose_dof(q_batch.cpu())
    success, q_isaac = simulator.run_sim()
    residual_rows = simulator.residual_rows
    simulator.destroy()
    return success.cpu().bool(), q_isaac.cpu(), residual_rows


def summarize_residuals(rows):
    l2 = np.array([r["l2"] for r in rows], dtype=np.float32)
    mean_abs = np.array([r["mean_abs"] for r in rows], dtype=np.float32)
    max_abs = np.array([r["max_abs"] for r in rows], dtype=np.float32)
    return {
        "l2_mean": float(l2.mean()),
        "l2_p50": float(np.percentile(l2, 50)),
        "l2_p90": float(np.percentile(l2, 90)),
        "l2_max": float(l2.max()),
        "mean_abs_mean": float(mean_abs.mean()),
        "max_abs_mean": float(max_abs.mean()),
    }


def summarize_success(rows):
    return sum(int(r["success"].sum()) for r in rows), sum(r["success"].numel() for r in rows)


def main():
    eval_dir = Path(__file__).resolve().parent
    data = torch.load(eval_dir / "inspire_zero_shot_grasps.pt", map_location="cpu")

    rows_a = []
    rows_crude = []
    rows_lsq = []
    residual_crude_all = []
    residual_lsq_all = []
    for item in data:
        object_name = item["object_name"]
        q = item["q_dro_order"]

        success_a, _, residual_crude_rows = run_native_inspire(object_name, q, projection_mode="none")
        success_crude, _, _ = run_native_inspire(object_name, q, projection_mode="crude")
        success_lsq, _, residual_lsq_rows = run_native_inspire(object_name, q, projection_mode="lsq")

        rows_a.append({"object_name": object_name, "success": success_a})
        rows_crude.append({"object_name": object_name, "success": success_crude})
        rows_lsq.append({"object_name": object_name, "success": success_lsq})
        for idx, residual in enumerate(residual_crude_rows):
            residual_crude_all.append(
                {
                    "object_name": object_name,
                    "index": idx,
                    "path_a_success": bool(success_a[idx]),
                    "path_b_crude_success": bool(success_crude[idx]),
                    **residual,
                }
            )
        for idx, residual in enumerate(residual_lsq_rows):
            residual_lsq_all.append(
                {
                    "object_name": object_name,
                    "index": idx,
                    "path_a_success": bool(success_a[idx]),
                    "path_b_lsq_success": bool(success_lsq[idx]),
                    **residual,
                }
            )

        print(
            f"{object_name}: A {int(success_a.sum())}/{success_a.numel()} | "
            f"B-crude {int(success_crude.sum())}/{success_crude.numel()} | "
            f"B-lsq {int(success_lsq.sum())}/{success_lsq.numel()}"
        )

    total_a, total = summarize_success(rows_a)
    total_crude, _ = summarize_success(rows_crude)
    total_lsq, _ = summarize_success(rows_lsq)
    print(
        f"total: A {total_a}/{total} = {total_a / total:.4f} | "
        f"B-crude {total_crude}/{total} = {total_crude / total:.4f} | "
        f"B-lsq {total_lsq}/{total} = {total_lsq / total:.4f}"
    )

    residual_crude_summary = summarize_residuals(residual_crude_all)
    residual_lsq_summary = summarize_residuals(residual_lsq_all)
    l2_lsq = np.array([r["l2"] for r in residual_lsq_all], dtype=np.float32)
    b_lsq_fail = np.array([not r["path_b_lsq_success"] for r in residual_lsq_all], dtype=np.float32)
    corr_lsq = float(np.corrcoef(l2_lsq, b_lsq_fail)[0, 1]) if np.std(b_lsq_fail) > 0 and np.std(l2_lsq) > 0 else float("nan")
    print("residual crude summary:", residual_crude_summary)
    print("residual lsq summary:", residual_lsq_summary)
    print(f"corr(lsq l2 residual, B-lsq failure): {corr_lsq:.4f}")

    out = eval_dir / "inspire_native_path_ab_results.pt"
    torch.save(
        {
            "path_a": rows_a,
            "path_b_crude": rows_crude,
            "path_b_lsq": rows_lsq,
            "residuals": residual_crude_all,
            "residuals_crude": residual_crude_all,
            "residuals_lsq": residual_lsq_all,
            "residual_crude_summary": residual_crude_summary,
            "residual_lsq_summary": residual_lsq_summary,
            "corr_lsq_l2_b_failure": corr_lsq,
        },
        out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
