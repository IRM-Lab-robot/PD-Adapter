#!/usr/bin/env python3
"""Train SVH pi_phi 8-head refiner from B-search teacher positives."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


NEW_ROOT = Path("/home/user/yangbo_program/new")
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
ABILITY_TRAIN = NEW_ROOT / "src/training/train_ability_pi_phi_multihead.py"
DATA_DIR = NEW_ROOT / "results/pd_refiner_svh/bsearch_train40_p64_g4_r1_m20"
OUT_DIR = NEW_ROOT / "results/pd_refiner_svh/pi_phi"
REPORT_PATH = NEW_ROOT / "报告4.txt"

sys.path.insert(0, str(DRO_ROOT))

from utils.hand_model import create_hand_model  # noqa: E402


spec = importlib.util.spec_from_file_location("ability_pi_phi_train", ABILITY_TRAIN)
if spec is None or spec.loader is None:
    raise ImportError(ABILITY_TRAIN)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SEED = base.SEED
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
LR = base.LR
WEIGHT_DECAY = base.WEIGHT_DECAY
K = base.K
DELTA_SCALE = base.DELTA_SCALE

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


def svh_limits() -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hand = create_hand_model("svh", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    q_lower = torch.as_tensor(lower_np[6:], dtype=torch.float32)
    q_upper = torch.as_tensor(upper_np[6:], dtype=torch.float32)
    name2finger = {name: i for i, name in enumerate(finger_names)}
    u_lower = torch.tensor([float(q_lower[name2finger[name]]) for name in ACTIVE_JOINTS], dtype=torch.float32)
    u_upper = torch.tensor([float(q_upper[name2finger[name]]) for name in ACTIVE_JOINTS], dtype=torch.float32)
    return finger_names, q_lower, q_upper, u_lower, u_upper


class SVHCoupling(nn.Module):
    def __init__(self, finger_names: list[str], q_lower: torch.Tensor, q_upper: torch.Tensor) -> None:
        super().__init__()
        self.finger_names = finger_names
        self.name2finger = {name: i for i, name in enumerate(finger_names)}
        self.register_buffer("q_lower", q_lower.float())
        self.register_buffer("q_upper", q_upper.float())

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(u.shape[:-1] + (20,), dtype=u.dtype, device=u.device)
        for active_i, active_name in enumerate(ACTIVE_JOINTS):
            q[..., self.name2finger[active_name]] = u[..., active_i]
        for passive_name, spec in COUPLING.items():
            q[..., self.name2finger[passive_name]] = float(spec["multiplier"]) * q[..., self.name2finger[spec["driver"]]] + float(spec["offset"])
        return torch.minimum(torch.maximum(q, self.q_lower), self.q_upper)


class SVHPiPhiMultiHead(nn.Module):
    def __init__(self, coupling: SVHCoupling, u_lower: torch.Tensor, u_upper: torch.Tensor, delta_scale: torch.Tensor, k: int = K) -> None:
        super().__init__()
        self.k = k
        self.object_encoder = base.PointNetEncoder()
        self.target_branch = nn.Sequential(nn.Linear(26, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(256 + 128, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 15)) for _ in range(k)])
        self.coupling = coupling
        self.register_buffer("u_lower", u_lower.float())
        self.register_buffer("u_upper", u_upper.float())
        self.register_buffer("delta_scale", delta_scale.float())

    def forward(self, object_points: torch.Tensor, target_grasp: torch.Tensor, w_star: torch.Tensor) -> dict[str, torch.Tensor]:
        object_feature = self.object_encoder(object_points)
        target_feature = self.target_branch(target_grasp)
        fused = self.fusion(torch.cat([object_feature, target_feature], dim=1))
        raw = torch.stack([head(fused) for head in self.heads], dim=1)
        delta_w = self.delta_scale.view(1, 1, 6) * torch.tanh(raw[..., :6])
        pred_w = w_star.unsqueeze(1) + delta_w
        pred_u = self.u_lower.view(1, 1, 9) + (self.u_upper - self.u_lower).view(1, 1, 9) * torch.sigmoid(raw[..., 6:])
        pred_q = self.coupling(pred_u)
        return {"pred_w": pred_w, "pred_u": pred_u, "pred_q": pred_q, "delta_w": delta_w}


def build_groups(data_dirs: list[tuple[str, Path]], coupling: SVHCoupling, u_lower: torch.Tensor, u_upper: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    num_targets = 0
    num_recomputed_q = 0
    num_raw_u_out = 0
    by_generator = Counter()
    groups_by_generator = Counter()
    object_points_cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for generator, data_dir in data_dirs:
            for path in sorted(data_dir.glob("*.pt")):
                data = torch.load(path, map_location="cpu")
                for row in data["rows"]:
                    object_name = str(row["object_name"])
                    q_seed = torch.as_tensor(row["q_seed"], dtype=torch.float32)
                    if object_name not in object_points_cache:
                        object_points_cache[object_name] = base.load_object_points(object_name)
                    for candidate in row["passing_candidates"]:
                        if not bool(candidate.get("cpu_success", False)) or candidate.get("label_source") != "cpu_final_trusted":
                            continue
                        source_idx = int(candidate["source_grasp_index"])
                        params = torch.as_tensor(candidate["params"], dtype=torch.float32).view(15)
                        raw_u = params[6:].clone()
                        u_target = torch.minimum(torch.maximum(raw_u, u_lower), u_upper)
                        q_target = coupling(u_target.unsqueeze(0)).squeeze(0).cpu()
                        q_candidate = torch.as_tensor(candidate["q"], dtype=torch.float32).view(26)
                        if bool(torch.any(raw_u < u_lower - 1e-6) or torch.any(raw_u > u_upper + 1e-6)):
                            num_raw_u_out += 1
                        if float((q_candidate[6:] - q_target).abs().max().item()) > 1e-4:
                            num_recomputed_q += 1
                        key = base.group_key(generator, object_name, source_idx)
                        if key not in groups:
                            groups[key] = {
                                "generator": generator,
                                "object_name": object_name,
                                "source_grasp_index": source_idx,
                                "object_points": object_points_cache[object_name],
                                "w_star": q_seed[source_idx, :6].clone(),
                                "q_star": q_seed[source_idx, 6:].clone(),
                                "targets": [],
                            }
                            groups_by_generator[generator] += 1
                        groups[key]["targets"].append({"w_elite": params[:6].clone(), "u_target": u_target.clone(), "q_target": q_target.clone(), "score": float(candidate.get("score", 0.0)), "generator": generator})
                        by_generator[generator] += 1
                        num_targets += 1
    out = [groups[key] for key in sorted(groups.keys())]
    stats = {
        "num_groups": len(out),
        "num_teacher_positives": num_targets,
        "num_objects": len({g["object_name"] for g in out}),
        "teacher_positives_by_generator": dict(by_generator),
        "groups_by_generator": dict(groups_by_generator),
        "data_dirs": {name: str(path) for name, path in data_dirs},
        "num_raw_u_out_of_limit": num_raw_u_out,
        "num_target_recomputed_q": num_recomputed_q,
    }
    return out, stats


def save_checkpoint(path: Path, model: SVHPiPhiMultiHead, epoch: int, metrics: dict[str, Any], finger_names: list[str]) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "model_class": "SVHPiPhiMultiHead",
        "K": model.k,
        "finger_names": finger_names,
        "active_joints": ACTIVE_JOINTS,
        "coupling": COUPLING,
        "active_lower": model.u_lower.detach().cpu().tolist(),
        "active_upper": model.u_upper.detach().cpu().tolist(),
        "delta_scale": model.delta_scale.detach().cpu().tolist(),
        "format": "params=[pred_w6,pred_u9]; pred_q=SVH Phi(pred_u)",
    }, path)


def append_report(metrics: dict[str, Any], report_title: str) -> None:
    text = f"""

## {report_title}

- 脚本: `{__file__}`
- 训练数据: `{metrics['data_dirs']}`
- 输出目录: `{Path(metrics['model_path']).parent}`
- teacher 正样本: {metrics['num_teacher_positives']}
- teacher by generator: {metrics.get('teacher_positives_by_generator', {})}
- grouped source grasps: {metrics['num_groups']}，train/val = {metrics['num_train_groups']}/{metrics['num_val_groups']}
- heads: {metrics['K']}
- best epoch: {metrics['best_epoch']}
- final train loss(last): {metrics['last_train_loss']:.6f}
- final val loss(best): {metrics['best_val_loss']:.6f}
- val best-head MAE: w={metrics['val_best_head_w_mae']:.6f}, u={metrics['val_best_head_u_mae']:.6f}, q={metrics['val_best_head_q_mae']:.6f}
- 8-head params pairwise L2 over all grouped sources: min={metrics['head_pairwise_l2']['min']:.6f}, mean={metrics['head_pairwise_l2']['mean']:.6f}, median={metrics['head_pairwise_l2']['median']:.6f}, max={metrics['head_pairwise_l2']['max']:.6f}
- head usage(val teacher assignments): {metrics['head_usage']}
- checkpoint: `{metrics['model_path']}`
- 备注: 只训练 SVH pi_phi；未运行新 B-search，未训练/修改 Inspire 或 Ability。
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", default=None, help="Teacher dir as NAME=PATH. May be repeated.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--report-title", default="Step 6: SVH pi_phi 8-head 多 basin refiner")
    args = parser.parse_args()

    data_dirs = []
    for item in (args.data_dir or [f"DRO={DATA_DIR}"]):
        if "=" not in item:
            raise ValueError(f"--data-dir must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        data_dirs.append((name, Path(path)))

    out_dir = Path(args.out_dir)
    best_model_path = out_dir / "best_svh_pi_phi_multihead.pt"
    last_model_path = out_dir / "last_svh_pi_phi_multihead.pt"
    train_log_path = out_dir / "train_log.json"
    metrics_path = out_dir / "metrics.json"

    base.set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    finger_names, q_lower, q_upper, u_lower, u_upper = svh_limits()
    groups, data_stats = build_groups(data_dirs, SVHCoupling(finger_names, q_lower, q_upper).cpu().eval(), u_lower, u_upper)
    dataset = base.AbilityGroupDataset(groups)
    train_indices, val_indices = base.random_split(len(dataset))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=base.collate_batch)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=base.collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVHPiPhiMultiHead(SVHCoupling(finger_names, q_lower, q_upper).to(device), u_lower, u_upper, DELTA_SCALE, K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_log = []
    best: dict[str, Any] | None = None
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_metrics = base.run_epoch(model, train_loader, device, optimizer)
        val_metrics = base.run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        train_log.append(row)
        print(f"epoch {epoch:03d} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} val_w={val_metrics['best_head_w_mae']:.6f} val_u={val_metrics['best_head_u_mae']:.6f} val_q={val_metrics['best_head_q_mae']:.6f}", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best = row
            save_checkpoint(best_model_path, model, epoch, row, finger_names)
    save_checkpoint(last_model_path, model, EPOCHS, train_log[-1], finger_names)
    assert best is not None
    model.load_state_dict(torch.load(best_model_path, map_location=device)["model_state_dict"])
    best_val = base.run_epoch(model, val_loader, device)
    diversity = base.head_pairwise_l2(model, dataset, device)
    metrics = {
        **data_stats,
        "num_train_groups": len(train_indices),
        "num_val_groups": len(val_indices),
        "K": K,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": int(best["epoch"]),
        "last_train_loss": float(train_log[-1]["train"]["loss"]),
        "last_val_loss": float(train_log[-1]["val"]["loss"]),
        "best_val_loss": float(best["val"]["loss"]),
        "best_val_set_loss": float(best["val"]["loss_set"]),
        "best_val_diversity_loss": float(best["val"]["diversity_loss"]),
        "val_best_head_w_mae": float(best_val["best_head_w_mae"]),
        "val_best_head_u_mae": float(best_val["best_head_u_mae"]),
        "val_best_head_q_mae": float(best_val["best_head_q_mae"]),
        "head_usage": best_val["head_usage"],
        "head_pairwise_l2": diversity,
        "model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "active_lower": u_lower.tolist(),
        "active_upper": u_upper.tolist(),
        "note": "SVH-only pi_phi 8-head refiner; grouped CPU-final trusted B-search positives; no IsaacGym/Inspire/Ability training run.",
    }
    train_log_path.write_text(json.dumps(train_log, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    append_report(metrics, args.report_title)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
