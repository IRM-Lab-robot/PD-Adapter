import json
from pathlib import Path

import numpy as np


COUPLING_PATH = Path(__file__).resolve().parent / "assets" / "inspire" / "coupling.json"


def load_coupling(path=COUPLING_PATH):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    active = data.get("active", data.get("active_joints"))
    if active is None:
        raise KeyError(f"{path} must contain 'active' or 'active_joints'")
    return active, data["coupling"]


active_joints, coupling = load_coupling()


def get_active_idx(name2idx):
    return np.array([name2idx[name] for name in active_joints], dtype=np.int32)


def forward_coupling(u6, name2idx, lower, upper):
    u6 = np.asarray(u6, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)

    if u6.shape != (len(active_joints),):
        raise ValueError(f"Expected u6 shape {(len(active_joints),)}, got {u6.shape}")

    q12 = np.zeros_like(lower, dtype=np.float32)

    for value, name in zip(u6, active_joints):
        q12[name2idx[name]] = value

    for passive_name, spec in coupling.items():
        driver_name = spec["driver"]
        driver_q = q12[name2idx[driver_name]]
        q12[name2idx[passive_name]] = (
            spec["multiplier"] * driver_q + spec["offset"]
        )

    return np.clip(q12, lower, upper)


active_idx = None
