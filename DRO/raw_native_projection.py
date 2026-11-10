"""Source-only active-to-passive and LSQ-to-passive grasp projections.

Raw here means the source's *active* finger targets drive the dependent joints
through the URDF coupling. Native fits the active targets to all original
finger joints before using the very same coupling. Both keep the source wrist.
Neither method is the historical full-actuated, projection_mode=none Raw.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPECS = {"inspire": ROOT / "assets/inspire/coupling.json",
         "ability": ROOT / "assets/ability_hand/coupling.json",
         "svh": ROOT / "assets/svh_hand/coupling.json"}


def project(q_source, finger_names, lower, upper, spec):
    """Return (Raw-active/Φ, Native-LSQ/Φ) in the source hand joint order."""
    q = torch.as_tensor(q_source, dtype=torch.float32).detach().cpu().numpy()
    names = list(finger_names)
    lo = np.asarray(lower, dtype=np.float32)
    hi = np.asarray(upper, dtype=np.float32)
    active = list(spec["active"])
    coupling = spec["coupling"]
    if (q.ndim != 2 or q.shape[1] != len(names) + 6 or len(lo) != len(names)
            or len(hi) != len(names) or len(set(names)) != len(names)
            or set(active) | set(coupling) != set(names)
            or set(active) & set(coupling) or not np.isfinite(q).all()):
        raise ValueError("Incomplete or non-finite source/coupling/joint-order data")
    name_to_index = {name: i for i, name in enumerate(names)}
    for dep in coupling.values():
        if dep["driver"] not in active:
            raise ValueError(f"Passive joint with unknown driver: {dep}")
    fingers = q[:, 6:]
    u_raw, u_native = [], []
    for name in active:
        idx = name_to_index[name]
        value = fingers[:, idx]
        terms = [(1.0, 0.0, idx)]
        for passive, dep in coupling.items():
            if dep["driver"] == name:
                terms.append((float(dep["multiplier"]), float(dep["offset"]), name_to_index[passive]))
        numerator = sum(a * (fingers[:, j] - b) for a, b, j in terms)
        denominator = sum(a * a for a, _, _ in terms)
        u_raw.append(np.clip(value, lo[idx], hi[idx]))
        u_native.append(np.clip(numerator / denominator, lo[idx], hi[idx]))

    def expand(commands):
        out = np.zeros_like(fingers)
        driven = {name: commands[:, j] for j, name in enumerate(active)}
        for name, value in driven.items():
            out[:, name_to_index[name]] = value
        for passive, dep in coupling.items():
            out[:, name_to_index[passive]] = (
                float(dep["multiplier"]) * driven[dep["driver"]] + float(dep["offset"]))
        out = np.clip(out, lo, hi)
        return torch.from_numpy(np.concatenate((q[:, :6], out), axis=1).astype(np.float32))

    return expand(np.stack(u_raw, axis=1)), expand(np.stack(u_native, axis=1))


def load_spec(hand: str) -> dict:
    if hand not in SPECS:
        raise ValueError(f"Unsupported hand: {hand}")
    return json.loads(SPECS[hand].read_text())
