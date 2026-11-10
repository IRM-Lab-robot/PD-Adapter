#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


NEW_ROOT = Path("/home/user/yangbo_program/new")
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
TRAIN_SCRIPT = NEW_ROOT / "src/training/train_svh_pi_phi_multihead.py"
DEFAULT_MODEL = NEW_ROOT / "results/pd_refiner_svh/pi_phi_mixed_dro_gendex/best_svh_pi_phi_multihead.pt"

sys.path.insert(0, str(DRO_ROOT))

from utils.hand_model import create_hand_model  # noqa: E402


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


def active_limits(finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    lo = [float(lower[name2finger[name]]) for name in ACTIVE_JOINTS]
    hi = [float(upper[name2finger[name]]) for name in ACTIVE_JOINTS]
    return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


def forward_coupling(u9, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q20 = np.zeros_like(lower, dtype=np.float32)
    values = {}
    for active_i, active_name in enumerate(ACTIVE_JOINTS):
        idx = name2finger[active_name]
        value = float(np.clip(u9[active_i], lower[idx], upper[idx]))
        values[active_name] = value
        q20[idx] = value
    for passive_name, spec in COUPLING.items():
        idx = name2finger[passive_name]
        q20[idx] = float(np.clip(spec["multiplier"] * values[spec["driver"]] + spec["offset"], lower[idx], upper[idx]))
    return q20


def lsq_project_q(q_batch, finger_names, lower, upper):
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q_rows = []
    for q in q_batch.detach().cpu().numpy().astype(np.float32):
        q_finger = q[6:]
        u9 = []
        for active_name in ACTIVE_JOINTS:
            terms = [(1.0, 0.0, name2finger[active_name])]
            for passive_name, spec in COUPLING.items():
                if spec["driver"] == active_name:
                    terms.append((float(spec["multiplier"]), float(spec["offset"]), name2finger[passive_name]))
            numerator = sum(a * (float(q_finger[idx]) - b) for a, b, idx in terms)
            denominator = sum(a * a for a, _, _ in terms)
            active_idx = name2finger[active_name]
            u9.append(float(np.clip(numerator / denominator, lower[active_idx], upper[active_idx])))
        q20 = forward_coupling(np.asarray(u9, dtype=np.float32), finger_names, lower, upper)
        q_rows.append(np.concatenate([q[:6], q20], axis=0))
    return torch.tensor(np.stack(q_rows), dtype=torch.float32)


def load_training_module():
    spec = importlib.util.spec_from_file_location("svh_pi_phi_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_path):
    mod = load_training_module()
    ckpt = torch.load(model_path, map_location="cpu")
    finger_names = list(ckpt["finger_names"])
    hand = create_hand_model("svh", device=torch.device("cpu"))
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    q_lower = torch.as_tensor(lower_np[6:], dtype=torch.float32)
    q_upper = torch.as_tensor(upper_np[6:], dtype=torch.float32)
    coupling = mod.SVHCoupling(finger_names, q_lower, q_upper)
    model = mod.SVHPiPhiMultiHead(
        coupling,
        torch.tensor(ckpt["active_lower"], dtype=torch.float32),
        torch.tensor(ckpt["active_upper"], dtype=torch.float32),
        torch.tensor(ckpt["delta_scale"], dtype=torch.float32),
        int(ckpt["K"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--zero-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--max-grasps", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt, mod = load_model(args.model)
    hand = create_hand_model("svh", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower_np[6:], dtype=np.float32)
    upper = np.asarray(upper_np[6:], dtype=np.float32)
    u_lower, u_upper = active_limits(finger_names, lower, upper)
    zero_rows = torch.load(args.zero_file, map_location="cpu")

    by_head = {head_id: [] for head_id in range(int(ckpt["K"]))}
    diagnostics = []
    with torch.no_grad():
        for row in zero_rows:
            object_name = str(row["object_name"])
            q_seed = lsq_project_q(row["q_dro_order"][: args.max_grasps], finger_names, lower, upper)
            object_points = mod.base.load_object_points(object_name).unsqueeze(0)
            for source_idx in range(q_seed.shape[0]):
                w_star = q_seed[source_idx, :6].view(1, 6)
                q_star = q_seed[source_idx, 6:].view(1, 20)
                target_grasp = torch.cat([w_star, q_star], dim=1)
                out = model(object_points, target_grasp, w_star)
                params_by_head = []
                for head_id in range(int(ckpt["K"])):
                    pred_w = out["pred_w"][0, head_id].cpu().float()
                    pred_u = out["pred_u"][0, head_id].cpu().float()
                    pred_q = out["pred_q"][0, head_id].cpu().float()
                    params = torch.cat([pred_w, pred_u]).float()
                    params_by_head.append(params)
                    by_head[head_id].append({
                        "object_name": object_name,
                        "source_grasp_index": int(source_idx),
                        "head_id": int(head_id),
                        "params": params,
                        "pred_w": pred_w,
                        "pred_u": pred_u,
                        "pred_q": pred_q,
                        "q_svh": torch.cat([pred_w, pred_q]).float(),
                        "source_q_seed": q_seed[source_idx].float(),
                    })
                pair_vals = []
                for i in range(len(params_by_head)):
                    for j in range(i + 1, len(params_by_head)):
                        pair_vals.append(float(torch.linalg.vector_norm(params_by_head[i] - params_by_head[j]).item()))
                u_all = torch.stack([p[6:] for p in params_by_head], dim=0)
                diagnostics.append({
                    "object_name": object_name,
                    "source_grasp_index": int(source_idx),
                    "all_u_in_limits": bool(torch.all(u_all >= torch.tensor(u_lower) - 1e-6) and torch.all(u_all <= torch.tensor(u_upper) + 1e-6)),
                    "u_min": float(u_all.min().item()),
                    "u_max": float(u_all.max().item()),
                    "pairwise_l2_mean": float(np.mean(pair_vals)),
                })

    files = {}
    for head_id, seeds in by_head.items():
        path = args.out_dir / f"svh_mixed_pi_phi_{args.tag}_head{head_id}.pt"
        torch.save({"info": {"head_id": head_id, "model_path": str(args.model), "zero_file": str(args.zero_file), "format": "params=[pred_w6,pred_u9]"}, "seeds": seeds}, path)
        files[f"head{head_id}"] = str(path)
    l2 = np.asarray([d["pairwise_l2_mean"] for d in diagnostics], dtype=np.float32)
    summary = {
        "model_path": str(args.model),
        "zero_file": str(args.zero_file),
        "tag": args.tag,
        "num_heads": int(ckpt["K"]),
        "num_sources": len(diagnostics),
        "files": files,
        "all_seed_u_in_active_limits": all(d["all_u_in_limits"] for d in diagnostics),
        "u_lower": u_lower.tolist(),
        "u_upper": u_upper.tolist(),
        "head_pairwise_l2_per_source": {"min": float(l2.min()), "mean": float(l2.mean()), "median": float(np.median(l2)), "max": float(l2.max())},
        "first5": diagnostics[:5],
    }
    summary_path = args.out_dir / f"svh_mixed_pi_phi_{args.tag}_allhead_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
