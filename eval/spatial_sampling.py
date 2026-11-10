"""Fixed-budget, position-stratified random grasp proposals.

The local third matches the historical Gaussian proposal. The other thirds
sample uniformly directed wrist offsets on two spherical shells around the
source wrist, instead of relying on Gaussian tails to reach new places.
Finger-command noise is unchanged across layers. NumPy-only to preserve the
Isaac Gym / torch import order of the search drivers.
"""

import numpy as np


LAYERS = ("local", "3-6cm", "6-12cm")
SHELLS_M = ((.03, .06), (.06, .12))
ROTATION_SCALES = (2.0, 3.0)


def layer_slices(population):
    """Near/mid/far allocation; keep the first layer at least as large."""
    if population < 3:
        raise ValueError("Spatial sampling needs at least three candidates")
    widths = [population // 3 + (i < population % 3) for i in range(3)]
    offsets = (0, widths[0], widths[0] + widths[1], population)
    return tuple(slice(offsets[i], offsets[i + 1]) for i in range(3))


def sample_stratified(means, stds, population, rng):
    """Return [source, population, parameter] samples with balanced layers."""
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    if (means.ndim != 2 or stds.shape != means.shape
            or means.shape[1] not in (12, 15) or population < 3):
        raise ValueError("Expected [source,12/15] means/stds and population >= 3")
    slices = layer_slices(population)
    samples = rng.normal(means[:, None, :], stds[:, None, :],
                         size=(len(means), population, means.shape[1])).astype(np.float32)
    samples[:, 0] = means  # Match the historical deterministic source candidate.
    for layer, (lower, upper) in enumerate(SHELLS_M, start=1):
        start, stop = slices[layer].start, slices[layer].stop
        count = stop - start
        directions = rng.normal(size=(len(means), count, 3)).astype(np.float32)
        directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
        # Uniform-in-volume shells cover space, rather than a single radius.
        radii = rng.uniform(lower ** 3, upper ** 3, size=(len(means), count, 1)) ** (1 / 3)
        samples[:, start:stop, :3] = means[:, None, :3] + directions * radii
        samples[:, start:stop, 3:6] = rng.normal(
            means[:, None, 3:6], stds[:, None, 3:6] * ROTATION_SCALES[layer - 1],
            size=(len(means), count, 3)).astype(np.float32)
    return samples
