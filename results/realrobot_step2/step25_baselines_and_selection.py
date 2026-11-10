#!/usr/bin/env python3
"""Step 2.5 Raw / naive-Phi / PD baselines and pre-registered target selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/user/yangbo_program/new")
sys.path.insert(0, str(ROOT / "eval"))

# holdout_isaac_battery imports IsaacGym before torch; keep this import first.
from holdout_isaac_battery import BATTERIES, project_lsq, run_holdout  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.hand_model import create_hand_model  # noqa: E402


OUT_DIR = ROOT / "results" / "realrobot_step2"
EVAL = ROOT / "eval"
BATTERY = BATTERIES["staged_standard_diag_rot12"]

OBJECTS = [
    {
        "display": "baseball",
        "object_name": "ycb+baseball",
        "zero_file": "original_test10_zero_shot_b20.pt",
        "pd_file": "bsearch_k_retrain/K8_natural_seed20262001/dro_default_p32/ycb_baseball.pt",
        "target_count": 20,
    },
    {
        "display": "rubiks_cube",
        "object_name": "ycb+rubiks_cube",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_rubiks_cube.pt",
        "target_count": 8,
    },
    {
        "display": "potted_meat_can",
        "object_name": "ycb+potted_meat_can",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_potted_meat_can.pt",
        "target_count": 8,
    },
    {
        "display": "tomato_soup_can",
        "object_name": "ycb+tomato_soup_can_repaired",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8_coacd25/ycb_tomato_soup_can_repaired.pt",
        "target_count": 8,
    },
    {
        "display": "water_bottle",
        "object_name": "contactdb+water_bottle",
        "zero_file": "original_test10_zero_shot_b20.pt",
        "pd_file": "bsearch_k_retrain/K8_natural_seed20262001/dro_default_p32/contactdb_water_bottle.pt",
        "target_count": 20,
    },
    {
        "display": "mustard_bottle",
        "object_name": "ycb+mustard_bottle",
        "zero_file": "original_test10_zero_shot_b20.pt",
        "pd_file": "bsearch_k_retrain/K8_natural_seed20262001/dro_default_p32/ycb_mustard_bottle.pt",
        "target_count": 20,
    },
    {
        "display": "banana",
        "object_name": "contactdb+banana",
        "zero_file": "original_test10_zero_shot_b20.pt",
        "pd_file": "bsearch_k_retrain/K8_natural_seed20262001/dro_default_p32/contactdb_banana.pt",
        "target_count": 20,
    },
    {
        "display": "apple",
        "object_name": "ycb+apple",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_apple.pt",
        "target_count": 8,
    },
    {
        "display": "lemon",
        "object_name": "ycb+lemon",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_lemon.pt",
        "target_count": 8,
    },
    {
        "display": "plum",
        "object_name": "ycb+plum",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_plum.pt",
        "target_count": 8,
    },
    {
        "display": "orange",
        "object_name": "ycb+orange",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_orange.pt",
        "target_count": 8,
    },
    {
        "display": "tennis_ball",
        "object_name": "ycb+tennis_ball",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8_coacd25/ycb_tennis_ball.pt",
        "target_count": 8,
    },
    {
        "display": "pudding_box",
        "object_name": "ycb+pudding_box",
        "zero_file": "realrobot_step2_dro_zero_shot_b20.pt",
        "pd_file": "realrobot_step2/pdrefiner_short_p32_g1_r1_s8/ycb_pudding_box.pt",
        "target_count": 8,
    },
]


def tensor_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def load_zero_row(zero_file: str, object_name: str) -> dict:
    rows = torch.load(EVAL / zero_file, map_location="cpu")
    for row in rows:
        if row["object_name"] == object_name:
            return row
    raise KeyError((zero_file, object_name))


def load_pd_row(pd_file: str) -> dict:
    data = torch.load(EVAL / pd_file, map_location="cpu")
    if isinstance(data, dict) and "rows" in data:
        return data["rows"][0]
    raise TypeError(f"unexpected PD file format: {pd_file}")


def best_pd_by_source(row: dict) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for cand in row.get("passing_candidates", []):
        source = int(cand["source_grasp_index"])
        score = float(cand.get("score", cand.get("rank_score", float("inf"))))
        if source not in best or score < float(best[source].get("score", float("inf"))):
            best[source] = cand
    return best


def validate_batch(object_name: str, q: torch.Tensor, projection_mode: str) -> tuple[list[bool], list[float]]:
    if q.numel() == 0:
        return [], []
    success, _, distance = run_holdout(object_name, q.float().cpu(), projection_mode, BATTERY, use_gpu_pipeline=False)
    return [bool(x) for x in success.tolist()], [float(x) for x in distance.float().tolist()]


def active_u_from_q(q_batch: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    hand = create_hand_model("inspire", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)
    q_lsq = project_lsq(q_batch, finger_names, lower, upper)
    name2finger = {name: i for i, name in enumerate(finger_names)}
    u_rows = []
    import inspire_coupling

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
        u_rows.append(u6)
    return q_lsq, np.asarray(u_rows, dtype=np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    selected = {
        "selection_rule": "For each object, sort targets by PD best-candidate margin = 0.02 - validated_distance descending; select top 5 without consulting Raw/naive outcomes.",
        "objects": [],
    }

    for spec in OBJECTS:
        display = spec["display"]
        object_name = spec["object_name"]
        target_count = int(spec["target_count"])
        zero_row = load_zero_row(spec["zero_file"], object_name)
        raw_q = zero_row["q_dro_order"].float()[:target_count].cpu()
        q_lsq, u_lsq = active_u_from_q(raw_q)
        pd_row = load_pd_row(spec["pd_file"])
        pd_best = best_pd_by_source(pd_row)

        print(f"validating {display} Raw n={target_count}", flush=True)
        raw_success, raw_distance = validate_batch(object_name, raw_q, "none")
        print(f"validating {display} naive n={target_count}", flush=True)
        naive_success, naive_distance = validate_batch(object_name, q_lsq, "lsq")

        pd_indices = [idx for idx in range(target_count) if idx in pd_best]
        pd_q = torch.stack([pd_best[idx]["q"].float().cpu() for idx in pd_indices], dim=0) if pd_indices else torch.empty((0, 18))
        print(f"validating {display} PD n={len(pd_indices)}", flush=True)
        pd_success_sub, pd_distance_sub = validate_batch(object_name, pd_q, "lsq")
        pd_success = [False] * target_count
        pd_distance: list[float | None] = [None] * target_count
        for local, idx in enumerate(pd_indices):
            pd_success[idx] = pd_success_sub[local]
            pd_distance[idx] = pd_distance_sub[local]

        per_target = []
        for idx in range(target_count):
            pd_margin = None if pd_distance[idx] is None else 0.02 - float(pd_distance[idx])
            per_target.append(
                {
                    "target_id": idx,
                    "raw_success": raw_success[idx],
                    "raw_distance": raw_distance[idx],
                    "naive_success": naive_success[idx],
                    "naive_distance": naive_distance[idx],
                    "pd_success": pd_success[idx],
                    "pd_distance": pd_distance[idx],
                    "pd_margin": pd_margin,
                }
            )
        selected_targets = sorted(
            [row for row in per_target if row["pd_margin"] is not None],
            key=lambda row: float(row["pd_margin"]),
            reverse=True,
        )[:5]
        selected_records = []
        for row in selected_targets:
            idx = int(row["target_id"])
            cand = pd_best[idx]
            selected_records.append(
                {
                    "target_id": idx,
                    "pd_margin": row["pd_margin"],
                    "pd_distance": row["pd_distance"],
                    "raw_q_star": tensor_list(raw_q[idx]),
                    "naive_q_proj": tensor_list(q_lsq[idx]),
                    "naive_u_proj": [float(x) for x in u_lsq[idx].tolist()],
                    "pd_q": tensor_list(cand["q"]),
                    "pd_params_wrist_u": tensor_list(cand["params"]),
                    "method_outcomes_reference_only": {
                        "raw_success": row["raw_success"],
                        "naive_success": row["naive_success"],
                        "pd_success": row["pd_success"],
                    },
                }
            )
        baseline_rows.append(
            {
                "object": display,
                "object_name": object_name,
                "target_count": target_count,
                "raw_success_count": int(sum(raw_success)),
                "naive_success_count": int(sum(naive_success)),
                "pd_success_count": int(sum(pd_success)),
                "pd_available_targets": len(pd_indices),
                "raw_success_rate": float(sum(raw_success) / target_count),
                "naive_success_rate": float(sum(naive_success) / target_count),
                "pd_success_rate": float(sum(pd_success) / target_count),
                "per_target": per_target,
            }
        )
        selected["objects"].append(
            {
                "object": display,
                "object_name": object_name,
                "selected_count": len(selected_records),
                "selected_target_ids": [r["target_id"] for r in selected_records],
                "records": selected_records,
                "insufficient_reason": "fewer than 5 PD candidates" if len(selected_records) < 5 else "",
            }
        )

    summary = {"battery": "staged_standard_diag_rot12", "rows": baseline_rows}
    (OUT_DIR / "step25_baselines_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "selected_targets.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": [{k: v for k, v in row.items() if k != "per_target"} for row in baseline_rows]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
