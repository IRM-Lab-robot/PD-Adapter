#!/usr/bin/env python3
"""Train Ability pi_phi 8-head refiner from B-search teacher positives."""

from __future__ import annotations

import json
import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


NEW_ROOT = Path("/home/user/yangbo_program/new")
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(DRO_ROOT))

from utils.hand_model import create_hand_model  # noqa: E402


DATA_DIR = NEW_ROOT / "results/pd_refiner_ability/bsearch_train40_p64_g4_r1_m20"
OUT_DIR = NEW_ROOT / "results/pd_refiner_ability/pi_phi"
REPORT_PATH = NEW_ROOT / "报告3.txt"

SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
K = 8
TAU = 0.15
DELTA_SCALE = torch.tensor([0.20, 0.20, 0.20, 1.0, 1.0, 1.0], dtype=torch.float32)
ACTIVE_JOINTS = ["index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q1", "thumb_q2"]
COUPLING = {
    "index_q2": {"driver": "index_q1", "multiplier": 1.05851325, "offset": 0.0},
    "middle_q2": {"driver": "middle_q1", "multiplier": 1.05851325, "offset": 0.0},
    "ring_q2": {"driver": "ring_q1", "multiplier": 1.05851325, "offset": 0.0},
    "pinky_q2": {"driver": "pinky_q1", "multiplier": 1.05851325, "offset": 0.0},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(object_name: str) -> str:
    return object_name.replace("+", "_")


def object_pc_path(object_name: str) -> Path:
    dataset, name = object_name.split("+", 1)
    return DRO_ROOT / "data/PointCloud/object" / dataset / f"{name}.pt"


def load_object_points(object_name: str) -> torch.Tensor:
    value = torch.load(object_pc_path(object_name), map_location="cpu")
    if isinstance(value, dict):
        for key in ("points", "object_points", "pc"):
            if key in value:
                value = value[key]
                break
    pts = torch.as_tensor(value, dtype=torch.float32).clone()
    if pts.ndim != 2 or pts.shape[0] < 512 or pts.shape[1] < 3:
        raise ValueError(f"Bad object pc for {object_name}: {tuple(pts.shape)}")
    return pts[:512, :3].contiguous()


def ability_limits() -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hand = create_hand_model("ability", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    q_lower = torch.as_tensor(lower_np[6:], dtype=torch.float32)
    q_upper = torch.as_tensor(upper_np[6:], dtype=torch.float32)
    name2finger = {name: i for i, name in enumerate(finger_names)}
    u_lower = torch.tensor([float(q_lower[name2finger[name]]) for name in ACTIVE_JOINTS], dtype=torch.float32)
    u_upper = torch.tensor([float(q_upper[name2finger[name]]) for name in ACTIVE_JOINTS], dtype=torch.float32)
    return finger_names, q_lower, q_upper, u_lower, u_upper


class AbilityCoupling(nn.Module):
    def __init__(self, finger_names: list[str], q_lower: torch.Tensor, q_upper: torch.Tensor) -> None:
        super().__init__()
        self.finger_names = finger_names
        self.name2finger = {name: i for i, name in enumerate(finger_names)}
        self.register_buffer("q_lower", q_lower.float())
        self.register_buffer("q_upper", q_upper.float())

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(u.shape[:-1] + (10,), dtype=u.dtype, device=u.device)
        for active_i, active_name in enumerate(ACTIVE_JOINTS):
            q[..., self.name2finger[active_name]] = u[..., active_i]
        for passive_name, spec in COUPLING.items():
            q[..., self.name2finger[passive_name]] = float(spec["multiplier"]) * q[..., self.name2finger[spec["driver"]]] + float(spec["offset"])
        return torch.minimum(torch.maximum(q, self.q_lower), self.q_upper)


def group_key(generator: str, object_name: str, source_idx: int) -> tuple[str, str, int]:
    return generator, object_name, int(source_idx)


def build_groups(data_dirs: list[tuple[str, Path]], coupling: AbilityCoupling, u_lower: torch.Tensor, u_upper: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                        object_points_cache[object_name] = load_object_points(object_name)
                    for candidate in row["passing_candidates"]:
                        if not bool(candidate.get("cpu_success", False)) or candidate.get("label_source") != "cpu_final_trusted":
                            continue
                        source_idx = int(candidate["source_grasp_index"])
                        params = torch.as_tensor(candidate["params"], dtype=torch.float32).view(12)
                        raw_u = params[6:].clone()
                        u_target = torch.minimum(torch.maximum(raw_u, u_lower), u_upper)
                        q_target = coupling(u_target.unsqueeze(0)).squeeze(0).cpu()
                        q_candidate = torch.as_tensor(candidate["q"], dtype=torch.float32).view(16)
                        if bool(torch.any(raw_u < u_lower - 1e-6) or torch.any(raw_u > u_upper + 1e-6)):
                            num_raw_u_out += 1
                        if float((q_candidate[6:] - q_target).abs().max().item()) > 1e-4:
                            num_recomputed_q += 1
                        key = group_key(generator, object_name, source_idx)
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
                        groups[key]["targets"].append({
                            "w_elite": params[:6].clone(),
                            "u_target": u_target.clone(),
                            "q_target": q_target.clone(),
                            "score": float(candidate.get("score", 0.0)),
                            "generator": generator,
                        })
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


class AbilityGroupDataset(Dataset):
    def __init__(self, groups: list[dict[str, Any]]) -> None:
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.groups[idx]
        return {
            "object_points": group["object_points"].float(),
            "target_grasp": torch.cat([group["w_star"].float(), group["q_star"].float()], dim=0),
            "w_star": group["w_star"].float(),
            "q_star": group["q_star"].float(),
            "targets": group["targets"],
            "meta": {k: group[k] for k in ["generator", "object_name", "source_grasp_index"]},
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object_points": torch.stack([item["object_points"] for item in batch], dim=0),
        "target_grasp": torch.stack([item["target_grasp"] for item in batch], dim=0),
        "w_star": torch.stack([item["w_star"] for item in batch], dim=0),
        "q_star": torch.stack([item["q_star"] for item in batch], dim=0),
        "targets": [item["targets"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


class PointNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(inplace=True), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, 256), nn.ReLU(inplace=True))

    def forward(self, object_points: torch.Tensor) -> torch.Tensor:
        return self.net(object_points).max(dim=1).values


class AbilityPiPhiMultiHead(nn.Module):
    def __init__(self, coupling: AbilityCoupling, u_lower: torch.Tensor, u_upper: torch.Tensor, delta_scale: torch.Tensor, k: int = K) -> None:
        super().__init__()
        self.k = k
        self.object_encoder = PointNetEncoder()
        self.target_branch = nn.Sequential(nn.Linear(16, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(256 + 128, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 12)) for _ in range(k)])
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
        pred_u = self.u_lower.view(1, 1, 6) + (self.u_upper - self.u_lower).view(1, 1, 6) * torch.sigmoid(raw[..., 6:])
        pred_q = self.coupling(pred_u)
        return {"pred_w": pred_w, "pred_u": pred_u, "pred_q": pred_q, "delta_w": delta_w}


def pair_losses(pred_w: torch.Tensor, pred_u: torch.Tensor, pred_q: torch.Tensor, targets: list[dict[str, torch.Tensor]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_w = torch.stack([t["w_elite"] for t in targets], dim=0).to(device)
    target_u = torch.stack([t["u_target"] for t in targets], dim=0).to(device)
    target_q = torch.stack([t["q_target"] for t in targets], dim=0).to(device)
    pred_w_exp = pred_w[:, None, :].expand(-1, target_w.shape[0], -1)
    pred_u_exp = pred_u[:, None, :].expand(-1, target_u.shape[0], -1)
    pred_q_exp = pred_q[:, None, :].expand(-1, target_q.shape[0], -1)
    target_w_exp = target_w[None, :, :].expand(pred_w.shape[0], -1, -1)
    target_u_exp = target_u[None, :, :].expand(pred_u.shape[0], -1, -1)
    target_q_exp = target_q[None, :, :].expand(pred_q.shape[0], -1, -1)
    loss_w = torch.nn.functional.smooth_l1_loss(pred_w_exp, target_w_exp, reduction="none").mean(dim=-1)
    loss_u = torch.nn.functional.smooth_l1_loss(pred_u_exp, target_u_exp, reduction="none").mean(dim=-1)
    loss_q = torch.nn.functional.smooth_l1_loss(pred_q_exp, target_q_exp, reduction="none").mean(dim=-1)
    return 1.5 * loss_w + 2.0 * loss_u + 2.0 * loss_q, loss_w, loss_u, loss_q


def diversity_loss(pred_u: torch.Tensor) -> torch.Tensor:
    pieces = []
    for i in range(pred_u.shape[1]):
        for j in range(i + 1, pred_u.shape[1]):
            pieces.append(torch.exp(-torch.linalg.vector_norm(pred_u[:, i, :] - pred_u[:, j, :], dim=-1) / TAU))
    return torch.stack(pieces, dim=0).mean() if pieces else pred_u.new_tensor(0.0)


def compute_batch_loss(out: dict[str, torch.Tensor], batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    set_losses = []
    target_losses = []
    head_losses = []
    for i, targets in enumerate(batch["targets"]):
        pair, _, _, _ = pair_losses(out["pred_w"][i], out["pred_u"][i], out["pred_q"][i], targets, device)
        target_to_head = pair.min(dim=0).values.mean()
        head_to_target = pair.min(dim=1).values.mean()
        target_losses.append(target_to_head)
        head_losses.append(head_to_target)
        set_losses.append(target_to_head + 0.5 * head_to_target)
    loss_set = torch.stack(set_losses).mean()
    loss_div = diversity_loss(out["pred_u"])
    delta_reg = out["delta_w"].abs().mean()
    loss = loss_set + 0.05 * loss_div + 0.005 * delta_reg
    return {"loss": loss, "loss_set": loss_set, "target_to_head_loss": torch.stack(target_losses).mean(), "head_to_target_loss": torch.stack(head_losses).mean(), "diversity_loss": loss_div, "delta_w_reg": delta_reg}


def run_epoch(model: AbilityPiPhiMultiHead, loader: DataLoader, device: torch.device, optimizer=None) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = Counter()
    count_groups = 0
    count_targets = 0
    head_usage = Counter({f"head_{i}": 0 for i in range(model.k)})
    min_w_mae: list[float] = []
    min_u_mae: list[float] = []
    min_q_mae: list[float] = []
    for batch in loader:
        object_points = batch["object_points"].to(device)
        target_grasp = batch["target_grasp"].to(device)
        w_star = batch["w_star"].to(device)
        with torch.set_grad_enabled(is_train):
            out = model(object_points, target_grasp, w_star)
            losses = compute_batch_loss(out, batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
        batch_size = object_points.shape[0]
        count_groups += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu().item()) * batch_size
        with torch.no_grad():
            for i, targets in enumerate(batch["targets"]):
                pair, _, _, _ = pair_losses(out["pred_w"][i], out["pred_u"][i], out["pred_q"][i], targets, device)
                for head in pair.argmin(dim=0).detach().cpu().tolist():
                    head_usage[f"head_{int(head)}"] += 1
                target_w = torch.stack([t["w_elite"] for t in targets], dim=0).to(device)
                target_u = torch.stack([t["u_target"] for t in targets], dim=0).to(device)
                target_q = torch.stack([t["q_target"] for t in targets], dim=0).to(device)
                min_w_mae.extend([float(x) for x in (out["pred_w"][i][:, None, :] - target_w[None, :, :]).abs().mean(dim=-1).min(dim=0).values.detach().cpu().tolist()])
                min_u_mae.extend([float(x) for x in (out["pred_u"][i][:, None, :] - target_u[None, :, :]).abs().mean(dim=-1).min(dim=0).values.detach().cpu().tolist()])
                min_q_mae.extend([float(x) for x in (out["pred_q"][i][:, None, :] - target_q[None, :, :]).abs().mean(dim=-1).min(dim=0).values.detach().cpu().tolist()])
                count_targets += len(targets)
    metrics = {key: totals[key] / max(count_groups, 1) for key in ["loss", "loss_set", "target_to_head_loss", "head_to_target_loss", "diversity_loss", "delta_w_reg"]}
    metrics.update({
        "best_head_w_mae": float(np.mean(min_w_mae)) if min_w_mae else 0.0,
        "best_head_u_mae": float(np.mean(min_u_mae)) if min_u_mae else 0.0,
        "best_head_q_mae": float(np.mean(min_q_mae)) if min_q_mae else 0.0,
        "num_groups": count_groups,
        "num_targets": count_targets,
        "head_usage": dict(head_usage),
    })
    return metrics


def random_split(num_samples: int, val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    indices = list(range(num_samples))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    num_val = max(1, int(round(num_samples * val_fraction)))
    return indices[num_val:], indices[:num_val]


def save_checkpoint(path: Path, model: AbilityPiPhiMultiHead, epoch: int, metrics: dict[str, Any], finger_names: list[str]) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "model_class": "AbilityPiPhiMultiHead",
        "K": model.k,
        "finger_names": finger_names,
        "active_joints": ACTIVE_JOINTS,
        "coupling": COUPLING,
        "active_lower": model.u_lower.detach().cpu().tolist(),
        "active_upper": model.u_upper.detach().cpu().tolist(),
        "delta_scale": model.delta_scale.detach().cpu().tolist(),
        "format": "params=[pred_w6,pred_u6]; pred_q=Ability Phi(pred_u)",
    }, path)


def head_pairwise_l2(model: AbilityPiPhiMultiHead, dataset: Dataset, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_batch)
    vals = []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            out = model(batch["object_points"].to(device), batch["target_grasp"].to(device), batch["w_star"].to(device))
            params = torch.cat([out["pred_w"], out["pred_u"]], dim=-1).detach().cpu()
            for i in range(params.shape[1]):
                for j in range(i + 1, params.shape[1]):
                    vals.extend(torch.linalg.vector_norm(params[:, i, :] - params[:, j, :], dim=-1).tolist())
    t = torch.tensor(vals, dtype=torch.float32)
    return {"min": float(t.min()), "mean": float(t.mean()), "median": float(t.median()), "max": float(t.max()), "num_pairs": int(t.numel())}


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
- 备注: 只训练 Ability pi_phi；未运行 B-search，未训练/修改 Inspire。
"""
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", default=None, help="Teacher dir as NAME=PATH. May be repeated.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--report-title", default="Step 7: Ability pi_phi 8-head 多 basin refiner")
    args = parser.parse_args()

    data_dirs = []
    for item in (args.data_dir or [f"DRO={DATA_DIR}"]):
        if "=" not in item:
            raise ValueError(f"--data-dir must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        data_dirs.append((name, Path(path)))
    out_dir = Path(args.out_dir)
    best_model_path = out_dir / "best_ability_pi_phi_multihead.pt"
    last_model_path = out_dir / "last_ability_pi_phi_multihead.pt"
    train_log_path = out_dir / "train_log.json"
    metrics_path = out_dir / "metrics.json"

    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    finger_names, q_lower, q_upper, u_lower, u_upper = ability_limits()
    groups, data_stats = build_groups(data_dirs, AbilityCoupling(finger_names, q_lower, q_upper).cpu().eval(), u_lower, u_upper)
    dataset = AbilityGroupDataset(groups)
    train_indices, val_indices = random_split(len(dataset))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AbilityPiPhiMultiHead(AbilityCoupling(finger_names, q_lower, q_upper).to(device), u_lower, u_upper, DELTA_SCALE, K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_log = []
    best: dict[str, Any] | None = None
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
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
    best_val = run_epoch(model, val_loader, device)
    diversity = head_pairwise_l2(model, dataset, device)
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
        "note": "Ability-only pi_phi 8-head refiner; grouped CPU-final trusted B-search positives; no IsaacGym or Inspire training run.",
    }
    train_log_path.write_text(json.dumps(train_log, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    append_report(metrics, args.report_title)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
