"""Hand-specific bounded-LSQ projection of finger *position targets*.

This is used after the grasp controller constructs its pre-open and closing
targets. It does not constrain actual joint positions during contact.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=2)
def _spec(hand):
    if hand not in ("ability", "svh"):
        raise ValueError("Target LSQ coupling supports only ability and svh")
    directory = "ability_hand" if hand == "ability" else "svh_hand"
    path = Path(__file__).resolve().parent / "assets" / directory / "coupling.json"
    spec = json.loads(path.read_text())
    return spec["active"], spec["coupling"]


def project_finger_targets_lsq(hand, finger_q, finger_names, lower, upper):
    """Return Phi(LSQ(finger_q)), clipped to the hand's joint limits.

    `finger_q` accepts one [n] target or a [batch,n] tensor/array. Each joint
    belongs to exactly one active command, so the constrained LSQ solution is
    computed independently for each driver, as in the existing search seeds.
    The six-dimensional wrist is not part of this function.
    """
    names = tuple(finger_names)
    values = np.asarray(finger_q, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != len(names):
        raise ValueError("Finger targets must have shape [n] or [batch,n]")
    one = values.ndim == 1
    batch = values[None, :] if one else values
    lo = np.asarray(lower, dtype=np.float32)
    hi = np.asarray(upper, dtype=np.float32)
    if (lo.shape != (len(names),) or hi.shape != lo.shape or
            not np.all(np.isfinite(batch)) or not np.all(np.isfinite(lo)) or
            not np.all(np.isfinite(hi)) or np.any(lo > hi)):
        raise ValueError("Invalid finger targets or joint limits")
    name_to_index = {name: index for index, name in enumerate(names)}
    if len(name_to_index) != len(names):
        raise ValueError("Duplicate hand joint name")
    active, coupling = _spec(hand)
    if set(names) != set(active) | set(coupling) or set(active) & set(coupling):
        raise ValueError(f"Joint order does not cover the {hand} coupling map exactly")
    if any(rule["driver"] not in active for rule in coupling.values()):
        raise ValueError(f"Invalid {hand} coupling driver")

    result = np.empty_like(batch)
    for driver in active:
        if driver not in name_to_index:
            raise ValueError(f"Unknown active joint {driver}")
        active_idx = name_to_index[driver]
        numerator = batch[:, active_idx].astype(np.float64)
        denominator = 1.0
        for follower, rule in coupling.items():
            if rule["driver"] != driver:
                continue
            coeff, offset = float(rule["multiplier"]), float(rule["offset"])
            numerator = numerator + coeff * (batch[:, name_to_index[follower]] - offset)
            denominator += coeff * coeff
        command = np.clip(numerator / denominator, lo[active_idx], hi[active_idx]).astype(np.float32)
        result[:, active_idx] = command
        for follower, rule in coupling.items():
            if rule["driver"] == driver:
                idx = name_to_index[follower]
                result[:, idx] = np.clip(float(rule["multiplier"]) * command + float(rule["offset"]),
                                          lo[idx], hi[idx])
    result = np.clip(result, lo, hi)
    return result[0] if one else result
