#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

MIRROR_Y = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def mirror_points(points: np.ndarray, mirror: np.ndarray = MIRROR_Y) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ mirror.T


def mirror_wrist_pose(R: np.ndarray, t: np.ndarray, mirror: np.ndarray = MIRROR_Y) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return mirror @ R @ mirror, mirror @ t


def mirror_command(u: np.ndarray) -> np.ndarray:
    return np.asarray(u).copy()
