#!/usr/bin/env python3
"""Generate pi_phi_v5 head-0 warm-start seeds for step 12."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
from torch import nn


NEW_ROOT = Path("/home/user/yangbo_program/new")
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(NEW_ROOT / "eval"))
sys.path.insert(0, str(DRO_ROOT))

import inspire_coupling  # noqa: E402
from utils.hand_model import create_hand_model  # noqa: E402


OBJECTS = ["ycb+baseball", "contactdb+mug", "contactdb+knife"]
RUN_SOURCES = {
    "DRO": NEW_ROOT / "eval/bsearch_train_rejected/test10_gensweep/dro_g4",
    "GenDexGrasp": NEW_ROOT / "eval/bsearch_train_rejected/test10_gensweep/gendex_g4",
}
MODEL_PATH = NEW_ROOT / "results/pd_refiner/pi_phi_v5_multihead/best_pi_phi_v5_multihead.pt"
OUT_DIR = NEW_ROOT / "results/pd_refiner/scorer_v9_bsearch_rejected/amortization_baseline"
OUT_PATH = OUT_DIR / "pi_phi_v5_head0_warmstart_step12.pt"
DRO_OUT_PATH = OUT_DIR / "pi_phi_v5_head0_warmstart_step12_dro.pt"
GENDEX_OUT_PATH = OUT_DIR / "pi_phi_v5_head0_warmstart_step12_gendex.pt"
SUMMARY_PATH = OUT_DIR / "pi_phi_v5_head0_warmstart_step12_summary.json"
URDF_PATH = DRO_ROOT / "data/data_urdf/robot/inspire/inspire_hand_right_extended.urdf"

FINGER_JOINT_ORDER = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
]


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


def load_q_limits() -> tuple[torch.Tensor, torch.Tensor]:
    root = ET.parse(URDF_PATH).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name in FINGER_JOINT_ORDER:
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Missing limit for {name}")
            limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    return (
        torch.tensor([limits[name][0] for name in FINGER_JOINT_ORDER], dtype=torch.float32),
        torch.tensor([limits[name][1] for name in FINGER_JOINT_ORDER], dtype=torch.float32),
    )


class InspireCoupling(nn.Module):
    def __init__(self, q_lower: torch.Tensor, q_upper: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("q_lower", q_lower.float())
        self.register_buffer("q_upper", q_upper.float())

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(u.shape[:-1] + (12,), dtype=u.dtype, device=u.device)
        q[..., 0] = u[..., 0]
        q[..., 1] = u[..., 1]
        q[..., 4] = u[..., 2]
        q[..., 6] = u[..., 3]
        q[..., 8] = u[..., 4]
        q[..., 10] = u[..., 5]
        q[..., 2] = 1.334 * q[..., 1]
        q[..., 3] = 0.667 * q[..., 1]
        q[..., 5] = 1.06399 * q[..., 4] - 0.04545
        q[..., 7] = 1.06399 * q[..., 6] - 0.04545
        q[..., 9] = 1.06399 * q[..., 8] - 0.04545
        q[..., 11] = 1.06399 * q[..., 10] - 0.04545
        return torch.minimum(torch.maximum(q, self.q_lower), self.q_upper)


class PointNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
        )

    def forward(self, object_points: torch.Tensor) -> torch.Tensor:
        return self.net(object_points).max(dim=1).values


class PiPhiV5MultiHead(nn.Module):
    def __init__(self, coupling: nn.Module, u_lower: torch.Tensor, u_upper: torch.Tensor, delta_scale: torch.Tensor, k: int) -> None:
        super().__init__()
        self.k = k
        self.object_encoder = PointNetEncoder()
        self.target_branch = nn.Sequential(nn.Linear(18, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True))
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
        raw_delta_w = raw[..., :6]
        raw_u = raw[..., 6:]
        delta_w = self.delta_scale.view(1, 1, 6) * torch.tanh(raw_delta_w)
        pred_w = w_star.unsqueeze(1) + delta_w
        pred_u = self.u_lower.view(1, 1, 6) + (self.u_upper - self.u_lower).view(1, 1, 6) * torch.sigmoid(raw_u)
        pred_q = self.coupling(pred_u)
        return {"pred_w": pred_w, "pred_u": pred_u, "pred_q": pred_q, "delta_w": delta_w}


def load_model() -> tuple[PiPhiV5MultiHead, dict[str, Any]]:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    q_lower, q_upper = load_q_limits()
    u_lower = torch.tensor(ckpt["active_lower"], dtype=torch.float32)
    u_upper = torch.tensor(ckpt["active_upper"], dtype=torch.float32)
    model = PiPhiV5MultiHead(InspireCoupling(q_lower, q_upper), u_lower, u_upper, torch.tensor(ckpt["delta_scale"], dtype=torch.float32), int(ckpt["K"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def lsq_project_q_local(q_batch: torch.Tensor, finger_names: list[str], lower, upper) -> tuple[torch.Tensor, torch.Tensor]:
    name2finger = {name: i for i, name in enumerate(finger_names)}
    q_rows = []
    u_rows = []
    for q in q_batch.detach().cpu().numpy().astype("float32"):
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
            u6.append(float(max(min(numerator / denominator, upper[active_idx]), lower[active_idx])))
        q12 = inspire_coupling.forward_coupling(torch.tensor(u6).numpy().astype("float32"), name2finger, lower, upper)
        q_rows.append(torch.tensor([*q[:6], *q12], dtype=torch.float32))
        u_rows.append(torch.tensor(u6, dtype=torch.float32))
    return torch.stack(q_rows), torch.stack(u_rows)


def safe_name(object_name: str) -> str:
    return object_name.replace("+", "_")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model()
    hand = create_hand_model("inspire", device=torch.device("cpu"))
    finger_names = hand.get_joint_orders()[6:]
    lower_np, upper_np = hand.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower_np[6:], dtype=torch.float32).numpy()
    upper = torch.as_tensor(upper_np[6:], dtype=torch.float32).numpy()

    seeds = []
    diagnostics = []
    with torch.no_grad():
        for generator, run_dir in RUN_SOURCES.items():
            for object_name in OBJECTS:
                data = torch.load(run_dir / f"{safe_name(object_name)}.pt", map_location="cpu")
                row = data["rows"][0]
                q_seed = torch.as_tensor(row["q_seed"], dtype=torch.float32)
                _, u_seed = lsq_project_q_local(q_seed, finger_names, lower, upper)
                object_points = load_object_points(object_name).unsqueeze(0)
                for source_idx in range(q_seed.shape[0]):
                    w_star = q_seed[source_idx, :6].view(1, 6)
                    q_star = q_seed[source_idx, 6:].view(1, 12)
                    target_grasp = torch.cat([w_star, q_star], dim=1)
                    out = model(object_points, target_grasp, w_star)
                    head_id = 0
                    pred_w = out["pred_w"][0, head_id].cpu().float()
                    pred_u = out["pred_u"][0, head_id].cpu().float()
                    pred_q = out["pred_q"][0, head_id].cpu().float()
                    params = torch.cat([pred_w, pred_u]).float()
                    source_params = torch.cat([q_seed[source_idx, :6], u_seed[source_idx].float()]).float()
                    delta = params - source_params
                    seeds.append({
                        "object_name": object_name,
                        "generator": generator,
                        "source_grasp_index": int(source_idx),
                        "head_id": int(head_id),
                        "params": params,
                        "pred_w": pred_w,
                        "pred_u": pred_u,
                        "pred_q": pred_q,
                        "q_isaac": torch.cat([pred_w, pred_q]).float(),
                        "source_params": source_params,
                        "source_q_seed": q_seed[source_idx].float(),
                    })
                    diagnostics.append({
                        "object_name": object_name,
                        "generator": generator,
                        "source_grasp_index": int(source_idx),
                        "head_id": int(head_id),
                        "l2_params_to_source": float(torch.linalg.vector_norm(delta).item()),
                        "l2_w_to_source": float(torch.linalg.vector_norm(delta[:6]).item()),
                        "l2_u_to_source": float(torch.linalg.vector_norm(delta[6:]).item()),
                        "max_abs_params_to_source": float(delta.abs().max().item()),
                        "u_min": float(pred_u.min().item()),
                        "u_max": float(pred_u.max().item()),
                        "u_in_limits": bool(torch.all(pred_u >= torch.tensor(ckpt["active_lower"]) - 1e-6) and torch.all(pred_u <= torch.tensor(ckpt["active_upper"]) + 1e-6)),
                    })

    payload = {
        "info": {
            "model_path": str(MODEL_PATH),
            "model_class": ckpt.get("model_class"),
            "epoch": ckpt.get("epoch"),
            "head_selection": "fixed head_id=0 for every source",
            "format": "params = [pred_w6, pred_u6], pred_w absolute, pred_u bounded absolute active commands",
        },
        "seeds": seeds,
    }
    torch.save(payload, OUT_PATH)
    torch.save({**payload, "seeds": [s for s in seeds if s["generator"] == "DRO"]}, DRO_OUT_PATH)
    torch.save({**payload, "seeds": [s for s in seeds if s["generator"] == "GenDexGrasp"]}, GENDEX_OUT_PATH)

    l2 = torch.tensor([d["l2_params_to_source"] for d in diagnostics], dtype=torch.float32)
    l2w = torch.tensor([d["l2_w_to_source"] for d in diagnostics], dtype=torch.float32)
    l2u = torch.tensor([d["l2_u_to_source"] for d in diagnostics], dtype=torch.float32)
    summary = {
        "out_path": str(OUT_PATH),
        "dro_out_path": str(DRO_OUT_PATH),
        "gendex_out_path": str(GENDEX_OUT_PATH),
        "model_path": str(MODEL_PATH),
        "model_class": ckpt.get("model_class"),
        "epoch": ckpt.get("epoch"),
        "num_seeds": len(seeds),
        "expected_seeds": len(OBJECTS) * len(RUN_SOURCES) * 20,
        "head_selection": "fixed head_id=0",
        "all_u_in_limits": all(d["u_in_limits"] for d in diagnostics),
        "l2_params_to_source": {"min": float(l2.min()), "mean": float(l2.mean()), "median": float(l2.median()), "max": float(l2.max())},
        "l2_w_to_source": {"min": float(l2w.min()), "mean": float(l2w.mean()), "median": float(l2w.median()), "max": float(l2w.max())},
        "l2_u_to_source": {"min": float(l2u.min()), "mean": float(l2u.mean()), "median": float(l2u.median()), "max": float(l2u.max())},
        "first5": diagnostics[:5],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
