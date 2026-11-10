#!/usr/bin/env python3
"""Paper-style Gaussian Random-9 control, matched to K8 diversity on val10.

Unlike the exploratory shell-initialization baseline, eight starting points
per source are iid fixed-scale Gaussian offsets from its LSQ-projected grasp.
The scale is frozen on training-disjoint validation objects and all nine
branches use the paper's local P32×G1 search. Each source has 288 GPU trials.
This entry point uses the audited nine-arm search/recheck engine, never a model
prediction as a random start. Run `prepare`, then `all --line ... --all-objects`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DRO import run_inspire_random9_spatial as base

BASE_SEARCH = base.search
BASE_CHECK_SEARCH = base.check_search
BASE_REPORT = base.report
base.__doc__ = __doc__
base.OUT = ROOT / "results/paper_random9_gaussian_b64"
base.WARM_OUT = base.OUT
base.WARM_METHOD = "Random-9-Gaussian/288"
base.SEARCH_TAG = "random9_gaussian"
base.SPATIAL_MODE = "gaussian"
base.engine.OUT = base.OUT
base.engine.METHOD = base.WARM_METHOD
base.engine.WORKER_ENTRYPOINT = Path(__file__).resolve()


def frozen_scale() -> tuple[float, str]:
    manifest = ROOT / "results/supplement_inspire_standard/seed_summary.json"
    model = ROOT / "results/supplement_inspire_standard/train/best_pi_phi_v5_multihead.pt"
    info = json.loads(manifest.read_text())
    sha = base.engine.sha(model)
    if (info.get("model_sha256") != sha
            or info.get("evaluation_objects") != list(base.OBJECTS)
            or info.get("validation_target_count") != 400
            or info.get("seed") != 20262041):
        raise ValueError(f"Incomplete validation-only random scale calibration: {manifest}")
    sigma = float(info["sigma_frozen_on_val"])
    if not 0 < sigma < .3:
        raise ValueError(f"Invalid frozen random sigma: {sigma}")
    return sigma, sha


def prepare(line: str) -> None:
    import torch

    if line not in base.LINES:
        raise ValueError(line)
    source_sha = base.source_sha(line)
    sigma, model_sha = frozen_scale()
    manifest = base.manifest_path(line)
    if manifest.exists():
        check_warm(line)
        print(f"reuse Gaussian starts for {line}", flush=True)
        return
    sys.path.insert(0, str(ROOT / "src/eval"))
    import generate_pi_phi_warmstart_step12 as projection

    hand = projection.create_hand_model("inspire", device=torch.device("cpu"))
    fingers = hand.get_joint_orders()[6:]
    lower, upper = hand.pk_chain.get_joint_limits()
    ckpt = torch.load(ROOT / "results/supplement_inspire_standard/train/best_pi_phi_v5_multihead.pt",
                      map_location="cpu", weights_only=False)
    u_lo = torch.tensor(ckpt["active_lower"], dtype=torch.float32)
    u_hi = torch.tensor(ckpt["active_upper"], dtype=torch.float32)
    generator = torch.Generator().manual_seed(20262041 + (line == "inspire-gen"))
    by_slot = {slot: [] for slot in range(8)}
    for source in base.pd.sources.source_rows(base.pd.source_file(line), "inspire", 64):
        obj = source["object_name"]
        projected, u = projection.lsq_project_q_local(
            torch.as_tensor(source["q_dro_order"], dtype=torch.float32),
            fingers, lower[6:], upper[6:])
        for sid in range(64):
            seed = torch.cat((projected[sid, :6], u[sid])).float()
            proposed = seed.unsqueeze(0) + sigma * torch.randn((8, 12), generator=generator)
            proposed[:, 6:] = torch.minimum(torch.maximum(proposed[:, 6:], u_lo), u_hi)
            for slot in range(8):
                by_slot[slot].append({"object_name": obj, "source_grasp_index": sid,
                                      "generator": "DRO" if line == "inspire-dro" else "GenDexGrasp",
                                      "slot_id": slot, "source_q_seed": projected[sid].clone(),
                                      "params": proposed[slot].clone()})
    hashes = []
    expected = {(obj, sid) for obj in base.OBJECTS for sid in range(64)}
    for slot, rows in by_slot.items():
        if len(rows) != 640 or {(r["object_name"], r["source_grasp_index"]) for r in rows} != expected:
            raise ValueError(f"Incomplete Gaussian slot {slot}")
        path = base.warm_path(line, slot)
        payload = {"info": {"method": base.WARM_METHOD, "line": line, "slot_id": slot,
                            "source_sha256": source_sha, "calibration_model_sha256": model_sha,
                            "sigma_validation_only": sigma, "seed": 20262041 + (line == "inspire-gen")},
                   "seeds": rows}
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=False)
            if (old["info"] != payload["info"] or len(old["seeds"]) != 640
                    or any(not torch.equal(a["params"], b["params"])
                           for a, b in zip(old["seeds"], rows))):
                raise ValueError(f"Existing partial Gaussian slot differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(path)
        hashes.append(base.engine.sha(path))
    base.engine.atomic_json(manifest, {"method": base.WARM_METHOD, "line": line,
                                       "source_sha256": source_sha,
                                       "calibration_model_sha256": model_sha,
                                       "sigma_validation_only": sigma,
                                       "warm_sha256": hashes, "slots": 8,
                                       "sources_per_slot": 640})
    check_warm(line)
    print(f"prepared {line}: eight validation-calibrated Gaussian starts × 640", flush=True)


def check_warm(line: str) -> str:
    import torch

    source_sha = base.source_sha(line)
    sigma, model_sha = frozen_scale()
    stored = json.loads(base.manifest_path(line).read_text())
    if (stored.get("method") != base.WARM_METHOD or stored.get("line") != line
            or stored.get("source_sha256") != source_sha
            or stored.get("calibration_model_sha256") != model_sha
            or stored.get("sigma_validation_only") != sigma
            or stored.get("slots") != 8 or stored.get("sources_per_slot") != 640):
        raise ValueError(f"Wrong Gaussian-start manifest: {line}")
    expected = {(obj, i) for obj in base.OBJECTS for i in range(64)}
    for slot in range(8):
        path = base.warm_path(line, slot)
        if not path.exists() or stored["warm_sha256"][slot] != base.engine.sha(path):
            raise ValueError(f"Missing or altered Gaussian-start file: {path}")
        rows = torch.load(path, map_location="cpu", weights_only=False)["seeds"]
        if (len(rows) != 640
                or {(r["object_name"], int(r["source_grasp_index"])) for r in rows} != expected
                or any(torch.as_tensor(r["params"]).shape != (12,) for r in rows)):
            raise ValueError(f"Incomplete Gaussian-start targets: {path}")
    return source_sha


def check_search(*args, **kwargs):
    # The inherited local-mode auditor validates search flags and candidate
    # source identities without imposing the exploratory warm-start shells.
    previous = base.SPATIAL_MODE
    base.SPATIAL_MODE = "local"
    try:
        return BASE_CHECK_SEARCH(*args, **kwargs)
    finally:
        base.SPATIAL_MODE = previous


def search(*args, **kwargs):
    previous = base.SPATIAL_MODE
    base.SPATIAL_MODE = "local"  # Each Gaussian center has standard P32 local sampling.
    try:
        return BASE_SEARCH(*args, **kwargs)
    finally:
        base.SPATIAL_MODE = previous


def report(line: str, obj: str) -> dict:
    value = BASE_REPORT(line, obj)
    value["comparison_note"] = ("All arms use local P32 sampling. Random Gaussian starts "
                                "replace learned K8 starts; data and source IDs are held fixed.")
    base.engine.atomic_json(base.OUT / line / "reports" / f"{base.safe(obj)}.json", value)
    return value


base.prepare = prepare
base.check_warm = check_warm
base.check_search = check_search
base.search = search
base.report = report


if __name__ == "__main__":
    base.main()
