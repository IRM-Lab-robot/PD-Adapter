#!/usr/bin/env python3
"""Export real-robot execution manifest and protocol templates."""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/user/yangbo_program/new")
DRO = Path("/home/user/yangbo_program/DRO-Grasp")
OUT = ROOT / "results" / "realrobot_step3"
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(DRO))

import inspire_coupling  # noqa: E402
from utils.hand_model import create_hand_model  # noqa: E402


SEED = 20260803
MY = np.diag([1.0, -1.0, 1.0])
METHODS = ["Raw", "naive_phi", "PD"]
FINAL_OBJECTS = [
    "plum",
    "lemon",
    "rubiks_cube",
    "apple",
    "orange",
    "potted_meat_can",
    "baseball",
    "mustard_bottle",
    "water_bottle",
    "banana",
]


def q_to_left_pose(q18: list[float]) -> dict[str, list[float]]:
    q = np.asarray(q18, dtype=np.float64)
    t_right = q[:3]
    r_right = Rotation.from_euler("XYZ", q[3:6]).as_matrix()
    t_left = MY @ t_right
    r_left = MY @ r_right @ MY
    quat_xyzw = Rotation.from_matrix(r_left).as_quat()
    return {
        "position_xyz_m": [float(x) for x in t_left.tolist()],
        "quaternion_xyzw": [float(x) for x in quat_xyzw.tolist()],
    }


def active_limits() -> tuple[np.ndarray, np.ndarray, list[int]]:
    hand = create_hand_model("inspire")
    joint_names = hand.get_joint_orders()[6:]
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower[6:], dtype=np.float64)
    upper = np.asarray(upper[6:], dtype=np.float64)
    name2idx = {name: idx for idx, name in enumerate(joint_names)}
    idxs = [name2idx[name] for name in inspire_coupling.active_joints]
    return lower[idxs], upper[idxs], idxs


LOWER_U, UPPER_U, ACTIVE_IDXS = active_limits()


def normalize_u(u: list[float]) -> list[float]:
    arr = np.asarray(u, dtype=np.float64)
    clipped = np.clip(arr, LOWER_U, UPPER_U)
    denom = np.maximum(UPPER_U - LOWER_U, 1e-9)
    return [float(x) for x in ((clipped - LOWER_U) / denom).tolist()]


def raw_u_from_q(raw_q: list[float]) -> list[float]:
    q = np.asarray(raw_q, dtype=np.float64)
    return [float(x) for x in np.clip(q[6:][ACTIVE_IDXS], LOWER_U, UPPER_U).tolist()]


def method_payload(record: dict, method: str) -> tuple[list[float], list[float], dict[str, object]]:
    if method == "Raw":
        q = record["raw_q_star"]
        u = raw_u_from_q(q)
        expected = record["method_outcomes_reference_only"]["raw_success"]
        mapping = "Raw q* contains 12 finger DOFs; hardware SDK accepts 6 active commands. Export uses projection_mode=none-compatible crude active-joint extraction: active q entries are clipped to active joint limits and passive DOFs are not commanded."
    elif method == "naive_phi":
        q = record["naive_q_proj"]
        u = [float(x) for x in record["naive_u_proj"]]
        expected = record["method_outcomes_reference_only"]["naive_success"]
        mapping = "LSQ projection to 6 active Inspire commands, matching naive-Phi simulation baseline."
    elif method == "PD":
        q = record["pd_q"]
        u = [float(x) for x in record["pd_params_wrist_u"][6:12]]
        expected = record["method_outcomes_reference_only"]["pd_success"]
        mapping = "PD best candidate wrist+u; first 6 params are wrist, last 6 are active Inspire commands."
    else:
        raise ValueError(method)
    return q, u, {"expected_success_sim_reference_only": bool(expected), "mapping_note": mapping}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    selected = json.loads((ROOT / "results" / "realrobot_step2" / "selected_targets.json").read_text())
    selected_by_object = {obj["object"]: obj for obj in selected["objects"]}
    trials = []
    csv_rows = []
    trial_index = 0
    rng = random.Random(SEED)

    for block_index, object_name in enumerate(FINAL_OBJECTS, start=1):
        obj = selected_by_object[object_name]
        block_trials = []
        for rec in obj["records"]:
            target_id = int(rec["target_id"])
            for method in METHODS:
                q, u_rad, expected = method_payload(rec, method)
                left_pose = q_to_left_pose(q)
                u_norm = normalize_u(u_rad)
                trial = {
                    "trial_id": f"RR3-{trial_index + 1:03d}",
                    "object_block_index": block_index,
                    "object": object_name,
                    "object_name_internal": obj["object_name"],
                    "target_id": target_id,
                    "method": method,
                    "left_wrist_pose": left_pose,
                    "pregrasp_u_normalized": [0.0] * 6,
                    "target_u_radians_model": [float(x) for x in u_rad],
                    "target_u_normalized": u_norm,
                    "command_sequence": [
                        {"step": "pregrasp_open", "u_normalized": [0.0] * 6, "hold_sec": 0.5},
                        {"step": "move_to_left_wrist_pose", "pose": left_pose},
                        {"step": "close_to_target", "u_normalized": u_norm, "hold_sec": 0.5},
                        {"step": "lift_10cm_and_hold", "hold_sec": 5.0},
                    ],
                    **expected,
                }
                block_trials.append(trial)
        rng.shuffle(block_trials)
        for trial in block_trials:
            trial_index += 1
            trial["trial_id"] = f"RR3-{trial_index:03d}"
            trials.append(trial)
            csv_rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "object": trial["object"],
                    "target_id": trial["target_id"],
                    "method": trial["method"],
                    "left_wrist_position_xyz_m": json.dumps(trial["left_wrist_pose"]["position_xyz_m"]),
                    "left_wrist_quaternion_xyzw": json.dumps(trial["left_wrist_pose"]["quaternion_xyzw"]),
                    "pregrasp_u_normalized": json.dumps(trial["pregrasp_u_normalized"]),
                    "target_u_normalized": json.dumps(trial["target_u_normalized"]),
                    "target_u_radians_model": json.dumps(trial["target_u_radians_model"]),
                    "sim_expected_success_reference_only": trial["expected_success_sim_reference_only"],
                    "mapping_note": trial["mapping_note"],
                }
            )

    distribution = []
    for object_name in FINAL_OBJECTS:
        rows = [r for r in trials if r["object"] == object_name]
        distribution.append(
            {
                "object": object_name,
                "Raw": sum(1 for r in rows if r["method"] == "Raw"),
                "naive_phi": sum(1 for r in rows if r["method"] == "naive_phi"),
                "PD": sum(1 for r in rows if r["method"] == "PD"),
                "total": len(rows),
                "selected_target_count": len({r["target_id"] for r in rows}),
            }
        )

    shortfall = 150 - len(trials)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step": "realrobot_step3",
        "constraints": "Pure export/documentation; no new simulation, no retraining, no re-search.",
        "randomization": {"seed": SEED, "scheme": "object blocks fixed in final list order; within each object block, method x target trials fully randomized"},
        "coordinate_convention": {"source": "right-hand simulation q=[xyz,rpyXYZ,12 finger DOFs]", "mirror": "My=diag(1,-1,1); t_left=My*t_right; R_left=My*R_right*My", "quaternion_order": "xyzw", "frame_note": "left wrist pose is expressed in the same object/table frame as simulation after mirror transform"},
        "command_convention": {"sdk_units": "TODO: convert normalized u to vendor SDK units after checking Inspire left-hand SDK documentation", "exported_units": "target_u_normalized in [0,1] plus target_u_radians_model for traceability", "pregrasp": "all-zero normalized open command", "active_joint_order": list(inspire_coupling.active_joints)},
        "global_calibration": {"thumb_model_choice": "TODO after calibration card", "thumb_yaw_bias": 0.0, "thumb_yaw_bias_units": "model radians before SDK conversion", "status": "pending_real_robot_calibration"},
        "final_objects": FINAL_OBJECTS,
        "expected_trials_nominal": 150,
        "exported_trials": len(trials),
        "shortfall_trials": shortfall,
        "shortfall_reason": "Under strict pre-registered PD-margin rule, potted_meat_can has 4 targets and all backup objects have fewer than 5 targets; no target was fabricated or selected using Raw/naive outcomes.",
        "distribution": distribution,
        "trials": trials,
    }
    (OUT / "execution_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_csv(
        OUT / "execution_trials.csv",
        csv_rows,
        ["trial_id", "object", "target_id", "method", "left_wrist_position_xyz_m", "left_wrist_quaternion_xyzw", "pregrasp_u_normalized", "target_u_normalized", "target_u_radians_model", "sim_expected_success_reference_only", "mapping_note"],
    )
    record_rows = [
        {
            "trial_id": r["trial_id"],
            "time": "",
            "operator": "",
            "object_placement_id": "",
            "reposed": "",
            "result_success_failure": "",
            "failure_mode_code": "",
            "notes": "",
        }
        for r in trials
    ]
    write_csv(OUT / "realrobot_record_template.csv", record_rows, list(record_rows[0].keys()))
    physical_rows = [
        {"object": obj, "physical_source": "USER_FILL", "mesh_deviation": "USER_FILL", "measured_mass_g": "", "note": "ContactDB substitute; instance differs from mesh" if obj in {"water_bottle", "banana"} else ""}
        for obj in FINAL_OBJECTS
    ]
    write_csv(OUT / "physical_object_table.csv", physical_rows, ["object", "physical_source", "mesh_deviation", "measured_mass_g", "note"])
    calibration_card = """# Inspire Left-Hand Thumb Calibration Card\n\n1. Mount the camera at the fixed calibration viewpoint and do not move it during the three captures.\n2. Put the left hand in the calibration fixture or marked table pose with no object.\n3. Send three normalized command vectors in this order:\n   - open: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n   - half_closed: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]\n   - thumb_opposition: [1.0, 0.8, 0.2, 0.2, 0.2, 0.2]\n4. For each command, wait 1 s after motion stops and take one photo. Save filenames with timestamp and command name.\n5. Overlay or compare the observed thumb tip against two FK predictions: mirrored right-hand model and vendor left-hand URDF.\n6. Choose the model with smaller thumb-tip image-plane error across the three photos.\n7. If the vendor left-hand URDF is closer, estimate a deployment-layer thumb yaw bias by minimizing the three-photo thumb-tip projection error. Record the bias in `execution_manifest.json/global_calibration/thumb_yaw_bias` before running table X trials.\n8. If the mirrored model is closer, keep thumb yaw bias at 0 and record `thumb_model_choice=mirrored_model`.\n9. Do not run grasp trials until the calibration decision and any bias are written into the manifest copy used by the robot operator.\n"""
    (OUT / "thumb_calibration_card.md").write_text(calibration_card)
    criteria_card = """# Real-Robot Success And Retry Criteria\n\nSuccess: the object leaves the table by at least 10 cm and remains held for at least 5 s.\n\nRetry rule: only hardware fault trials may be retried. Grasp failures are not retried. Every retry must keep the original trial ID and add a retry note.\n\nFailure codes:\nA = no contact / empty grasp\nB = slips after contact before lift\nC = drops during lift\nD = drops during 5 s hold\nE = collision or safety stop\nF = hardware fault retry\n\nInitial placement: each object uses a fixed marked pose. Photograph the placement before the first trial of each object block and whenever reposed.\n"""
    (OUT / "success_criteria_card.md").write_text(criteria_card)
    print(json.dumps({"exported_trials": len(trials), "shortfall_trials": shortfall, "distribution": distribution, "first5": csv_rows[:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
