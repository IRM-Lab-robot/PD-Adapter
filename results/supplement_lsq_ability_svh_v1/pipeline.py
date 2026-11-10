#!/usr/bin/env python3
"""Frozen inputs, LSQ teachers, object-disjoint K8 training, and test search."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from DRO import generate_64 as source_audit
from DRO import run_svh_cem288 as common

HANDS = ("ability", "svh")
GENERATORS = ("dro", "gendex")
METHODS = ("cem288", "sample288", "pd9")
BATTERY = "standard_six_force05_steps6"
TRAINING_SEED = 20262040
PART_SIZE = 8
ARMS = ("default",) + tuple(f"head{i}" for i in range(8))
SPLIT = json.loads((ROOT / "results/supplement_inspire_standard/teacher_summary.json").read_text())
TRAIN30 = tuple(SPLIT["train_objects"])
VAL10 = tuple(SPLIT["val_objects"])
TRAIN40 = tuple(json.loads((ROOT / "eval/inspire_train_test_split.json").read_text())["train"])
TEST10 = tuple(json.loads((ROOT / "eval/inspire_train_test_split.json").read_text())["test"][:10])


def safe(obj):
    return obj.replace("+", "_")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def source_file(hand, gen, training=False):
    if training:
        if gen == "dro":
            return ROOT / f"results/pd_refiner_{hand}/{hand}_train40_zero_shot_b60.pt"
        return ROOT / f"results/pd_refiner_{hand}/gendex_train40/{hand}_gendex_train40_grasps.pt"
    return ROOT / ("DRO" if gen == "dro" else "GEN") / f"{hand}.pt"


def model_file(hand):
    return OUT / "train" / hand / "best.pt"


def warm_file(hand, gen, head):
    return (OUT / "warmstarts" / hand / gen /
            f"{hand}_mixed_pi_phi_{gen}_b64_lsq_head{head}.pt")


def teacher_file(hand, gen, obj, smoke=False):
    stage = "smoke_teacher" if smoke else "teacher"
    return OUT / stage / hand / gen / f"{safe(obj)}.pt"


def indices(part):
    if part not in range(64 // PART_SIZE):
        raise ValueError(part)
    return list(range(part * PART_SIZE, (part + 1) * PART_SIZE))


def arms(method):
    return ARMS if method == "pd9" else (method,)


def search_file(hand, gen, method, arm, obj, part):
    ids = indices(part)
    return (OUT / "search" / hand / gen / method / arm / safe(obj) /
            f"part{part:02d}_s{ids[0]:02d}_e{ids[-1]:02d}.pt")


def search_configuration(method, arm):
    if method == "cem288":
        return 72, 4, 4, None
    if method == "sample288":
        return 288, 1, 1, None
    if method == "pd9" and arm in ARMS:
        return 32, 1, 8, None if arm == "default" else int(arm[4:])
    raise ValueError((method, arm))


def import_training(hand):
    path = ROOT / "src/training" / f"train_{hand}_pi_phi_multihead.py"
    spec = importlib.util.spec_from_file_location(f"lsq_{hand}_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_inputs(hand=None):
    import torch

    if (len(TRAIN30) != 30 or len(VAL10) != 10 or set(TRAIN30) | set(VAL10) != set(TRAIN40)
            or set(TRAIN30) & set(VAL10) or (set(TRAIN40) & set(TEST10))):
        raise ValueError("Frozen train30/val10/test10 object split is inconsistent")
    for h in HANDS if hand is None else (hand,):
        for gen in GENERATORS:
            train = source_file(h, gen, training=True)
            test = source_file(h, gen)
            train_rows = torch.load(train, map_location="cpu", weights_only=False)
            if (len(train_rows) != 40 or {r["object_name"] for r in train_rows} != set(TRAIN40)
                    or any(torch.as_tensor(r["q_dro_order"]).shape !=
                           (60 if gen == "dro" else 20, source_audit.DIMS[h]) for r in train_rows)):
                raise ValueError(f"Wrong train40 source distribution: {train}")
            full = source_audit.source_rows(test, h, 64)
            manifest = json.loads(test.with_suffix(".json").read_text())
            if (manifest["output_sha256"] != common.sha(test)
                    or [r["object_name"] for r in full] != list(TEST10)
                    or manifest["old_per_object"] != 20):
                raise ValueError(f"Wrong test10×64 source: {test}")
            print(f"{h}/{gen}: train40 × first20, disjoint test10 × 64; SHA {manifest['output_sha256'][:12]}", flush=True)


def check_teacher(path, hand, gen, obj, smoke=False):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    args = data["args"]
    expected = {"zero_file": str(source_file(hand, gen, training=True)),
                "selection_battery": BATTERY, "validator_projection": "lsq",
                "population": 64, "elites": 8, "generations": 4,
                "restarts": 1, "batch_source_grasps": 8,
                "keep_per_grasp": 3, "keep_per_object": 120,
                "cpu_final_candidates": 240, "search_pipeline": "gpu",
                "reuse_gpu_sim": True, "log_rejected": True, "seed": 10001,
                "max_grasps": 20, "source_indices": [0, 1] if smoke else None}
    if any(args.get(k) != value for k, value in expected.items()):
        raise ValueError(f"Inconsistent LSQ teacher configuration: {path}")
    count = 2 if smoke else 20
    if (data["objects"] != [obj] or data["total_source_grasps"] != count
            or len(data["rows"]) != 1):
        raise ValueError(f"Incomplete teacher source IDs: {path}")
    row = data["rows"][0]
    if (torch.as_tensor(row["q_seed"]).shape != (count, source_audit.DIMS[hand])
            or [int(g["grasp_index"]) for g in row["grasp_rows"]] != list(range(count))):
        raise ValueError(f"Wrong teacher projected grasp shape/IDs: {path}")
    for cand in row["passing_candidates"]:
        if cand.get("cpu_success") is not True or cand.get("label_source") != "cpu_final_trusted":
            raise ValueError(f"Untrusted teacher candidate: {path}")
    if len(row["passing_candidates"]) + len(row["cpu_final_rejected_candidates"]) != row["cpu_final_evaluated"]:
        raise ValueError(f"Teacher CPU-final rejected candidates missing: {path}")
    return data


def teachers(hand, gen, objects, smoke=False):
    check_inputs(hand)
    driver = ROOT / "eval" / f"generate_{hand}_train_bsearch.py"
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    for obj in objects:
        dest = teacher_file(hand, gen, obj, smoke)
        if dest.is_file():
            check_teacher(dest, hand, gen, obj, smoke)
            print(f"reuse teacher {hand}/{gen}/{obj}", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(driver), "--zero-file", str(source_file(hand, gen, training=True)),
               "--objects", obj, "--max-grasps", "20", "--selection-battery", BATTERY,
               "--validator-projection", "lsq", "--population", "64", "--elites", "8",
               "--generations", "4", "--restarts", "1", "--batch-source-grasps", "8",
               "--keep-per-grasp", "3", "--keep-per-object", "120",
               "--cpu-final-candidates", "240", "--search-pipeline", "gpu",
               "--reuse-gpu-sim", "--log-rejected", "--seed", "10001", "--out", str(dest)]
        if smoke:
            cmd.extend(("--source-indices", "0", "1"))
        with dest.with_suffix(".log").open("w") as log:
            ret = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if ret.returncode or not dest.is_file():
            raise RuntimeError(f"Teacher search failed: {dest.with_suffix('.log')}")
        check_teacher(dest, hand, gen, obj, smoke)
        print(f"teacher {hand}/{gen}/{obj} ({2 if smoke else 20} sources)", flush=True)


def check_all_teachers(hand):
    stats = {}
    for gen in GENERATORS:
        positive = 0
        per_object = {}
        for obj in TRAIN40:
            path = teacher_file(hand, gen, obj)
            if not path.is_file():
                raise FileNotFoundError(f"Complete all train40 teacher searches first: {path}")
            row = check_teacher(path, hand, gen, obj)["rows"][0]
            per_object[obj] = len(row["passing_candidates"])
            positive += per_object[obj]
        stats[gen] = {"positives": positive, "per_object": per_object,
                      "train30_positive": sum(per_object[x] for x in TRAIN30),
                      "val10_positive": sum(per_object[x] for x in VAL10)}
    return stats


def train(hand, smoke=False):
    import torch
    from torch.utils.data import DataLoader

    teacher_stats = check_all_teachers(hand)
    for gen in GENERATORS:
        if teacher_stats[gen]["train30_positive"] == 0 or teacher_stats[gen]["val10_positive"] == 0:
            raise ValueError(f"Cannot train/select a mixed {hand} model without {gen} positives")
    m = import_training(hand)
    base = m if hand == "ability" else m.base
    fingers, qlo, qhi, ulo, uhi = m.ability_limits() if hand == "ability" else m.svh_limits()
    def coupling():
        return (m.AbilityCoupling(fingers, qlo, qhi) if hand == "ability"
                else m.SVHCoupling(fingers, qlo, qhi))
    paths = [("DRO", OUT / "teacher" / hand / "dro"),
             ("GenDexGrasp", OUT / "teacher" / hand / "gendex")]
    groups, data_stats = m.build_groups(paths, coupling().cpu().eval(), ulo, uhi)
    training = [g for g in groups if g["object_name"] in TRAIN30]
    validation = [g for g in groups if g["object_name"] in VAL10]
    if (not training or not validation or len(training) + len(validation) != len(groups)
            or {g["object_name"] for g in training} & {g["object_name"] for g in validation}
            or {g["object_name"] for g in groups} & set(TEST10)):
        raise ValueError("Teacher source groups leak across object-disjoint splits")
    if smoke:
        training, validation = training[:8], validation[:8]
    stage_dir = OUT / ("smoke_train" if smoke else "train") / hand
    model_path = stage_dir / "best.pt"
    metrics_path = stage_dir / "metrics.json"
    if not smoke and metrics_path.exists():
        prior = json.loads(metrics_path.read_text())
        if (not model_path.is_file() or prior.get("checkpoint_sha256") != common.sha(model_path)
                or prior.get("seed") != TRAINING_SEED or prior.get("validator_projection") != "lsq"):
            raise ValueError(f"Existing training checkpoint/metrics disagree: {stage_dir}")
        print(f"reuse {hand} K8 checkpoint: {model_path}", flush=True)
        return
    if not smoke and model_path.exists():
        raise FileExistsError(f"Training stopped mid-run; inspect checkpoint before retry: {model_path}")
    base.set_seed(TRAINING_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = m.AbilityPiPhiMultiHead if hand == "ability" else m.SVHPiPhiMultiHead
    model = model_cls(coupling().to(device), ulo, uhi, m.DELTA_SCALE, k=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=m.LR, weight_decay=m.WEIGHT_DECAY)
    loader = lambda data, shuffle: DataLoader(base.AbilityGroupDataset(data),
                    batch_size=m.BATCH_SIZE, shuffle=shuffle,
                    collate_fn=base.collate_batch, num_workers=0)
    tr, va = loader(training, True), loader(validation, False)
    stage_dir.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    epochs = 2 if smoke else m.EPOCHS
    for epoch in range(1, epochs + 1):
        train_metrics = base.run_epoch(model, tr, device, optimizer)
        val_metrics = base.run_epoch(model, va, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            if not smoke:
                temp = stage_dir / "best.pt.tmp"
                m.save_checkpoint(temp, model, epoch, record, fingers)
                temp.replace(model_path)
        if epoch % 10 == 0 or smoke:
            print(f"{hand} epoch {epoch}/{epochs} train={train_metrics['loss']:.5f} "
                  f"val={val_metrics['loss']:.5f} best={best:.5f}", flush=True)
    atomic_json(stage_dir / "train_log.json", {"epochs": history})
    info = {"hand": hand, "K": 8, "seed": TRAINING_SEED,
            "battery": BATTERY, "validator_projection": "lsq",
            "split": "frozen object-disjoint train30/val10",
            "train_objects": list(TRAIN30), "val_objects": list(VAL10), "test_objects": list(TEST10),
            "teacher_stats": teacher_stats, "dataset_stats": data_stats,
            "train_groups": len(training), "val_groups": len(validation),
            "epochs": epochs,
            "best_epoch": min(history, key=lambda x: x["val"]["loss"])["epoch"],
            "best_val_loss": best,
            "checkpoint_sha256": common.sha(model_path) if not smoke else None,
            "checkpoint": str(model_path) if not smoke else None}
    atomic_json(metrics_path, info)
    print(f"{hand} {'smoke' if smoke else 'complete'} training: {info['best_epoch']=} {best=:.5f}")


def warm(hand, gen):
    import torch

    check_inputs(hand)
    model = model_file(hand)
    meta = OUT / "train" / hand / "metrics.json"
    if not meta.is_file() or not model.is_file():
        raise FileNotFoundError(f"Train the LSQ mixed K8 model first: {model}")
    info = json.loads(meta.read_text())
    if info["checkpoint_sha256"] != common.sha(model) or info["validator_projection"] != "lsq":
        raise ValueError(f"Wrong model for new LSQ warm starts: {model}")
    folder = OUT / "warmstarts" / hand / gen
    manifest = folder / "manifest.json"
    expected = {(obj, i) for obj in TEST10 for i in range(64)}

    def validate_files():
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing complete warm-start manifest: {manifest}")
        stored = json.loads(manifest.read_text())
        if (stored.get("model_sha256") != common.sha(model)
                or stored.get("source_sha256") != common.sha(source_file(hand, gen))):
            raise ValueError("Warm-start model or source changed")
        for head in range(8):
            path = warm_file(hand, gen, head)
            if not path.is_file() or common.sha(path) != stored["head_sha256"][head]:
                raise ValueError(f"Missing/modified learned initializations: {path}")
            rows = torch.load(path, map_location="cpu", weights_only=False)
            rows = rows["seeds"] if isinstance(rows, dict) else rows
            keys = {(r["object_name"], int(r["source_grasp_index"])) for r in rows}
            if len(rows) != 640 or keys != expected or any(torch.as_tensor(r["params"]).numel() !=
                    (12 if hand == "ability" else 15) for r in rows):
                raise ValueError(f"Predictions do not cover all 640 test targets: {path}")
        return stored

    if manifest.exists() or any(warm_file(hand, gen, h).exists() for h in range(8)):
        validate_files()
        print(f"reuse {hand}/{gen} LSQ K8 warm starts, 8×640", flush=True)
        return
    script = (ROOT / "results/pd_refiner_ability/generate_step6_mixed_warmstarts.py"
              if hand == "ability" else ROOT / "results/pd_refiner_svh/generate_svh_mixed_warmstarts.py")
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), "--model", str(model), "--zero-file",
           str(source_file(hand, gen)), "--out-dir", str(folder), "--tag", f"{gen}_b64_lsq",
           "--max-grasps", "64"]
    with (folder / "prepare.log").open("w") as log:
        run = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode or any(not warm_file(hand, gen, h).is_file() for h in range(8)):
        raise RuntimeError(f"Warm-start inference failed; inspect {folder/'prepare.log'}")
    atomic_json(manifest, {"model_sha256": common.sha(model),
                           "source_sha256": common.sha(source_file(hand, gen)),
                           "head_sha256": [common.sha(warm_file(hand, gen, h)) for h in range(8)],
                           "tag": f"{gen}_b64_lsq"})
    validate_files()
    print(f"{hand}/{gen} LSQ K8 warm starts generated: 8×640", flush=True)


def check_search(path, hand, gen, method, arm, obj, part):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    population, generations, batch, head = search_configuration(method, arm)
    ids = indices(part)
    expected = {"zero_file": str(source_file(hand, gen)), "selection_battery": BATTERY,
                "validator_projection": "lsq", "max_grasps": 64,
                "population": population, "generations": generations,
                "batch_source_grasps": batch, "elites": 8, "restarts": 1,
                "keep_per_grasp": 3, "keep_per_object": 3 * len(ids),
                "cpu_final_candidates": 3 * len(ids), "source_indices": ids,
                "warm_start_file": str(warm_file(hand, gen, head)) if head is not None else None,
                "search_pipeline": "gpu", "reuse_gpu_sim": True,
                "log_rejected": True, "seed": 10001 + 1009 * part}
    if any(data["args"].get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected search config in {path}")
    if (data.get("objects") != [obj] or data.get("total_source_grasps") != len(ids)
            or [int(g["grasp_index"]) for g in data["rows"][0]["grasp_rows"]] != ids):
        raise ValueError(f"Incomplete source IDs in {path}")
    row = data["rows"][0]
    n = 0
    for key, success in (("passing_candidates", True), ("cpu_final_rejected_candidates", False)):
        if key not in row:
            raise ValueError(f"Missing CPU-final rejected/accepted candidates: {path}")
        for cand in row[key]:
            if (int(cand["source_grasp_index"]) not in ids or cand["cpu_success"] is not success
                    or "gpu_rank_score" not in cand or
                    not math.isfinite(float(cand["gpu_rank_score"])) or
                    torch.as_tensor(cand["q"]).numel() != source_audit.DIMS[hand]):
                raise ValueError(f"Corrupt LSQ candidate: {path}")
            n += 1
    if n != row["cpu_final_evaluated"] or n > 3 * len(ids):
        raise ValueError(f"LSQ candidate pool is incomplete/truncated: {path}")
    return row


def search(hand, gen, method, objects, chosen_arm=None, chosen_part=None):
    check_inputs(hand)
    if method == "pd9":
        warm(hand, gen)
    driver = ROOT / "eval" / f"generate_{hand}_train_bsearch.py"
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    for obj in objects:
        for arm in ((chosen_arm,) if chosen_arm else arms(method)):
            population, generations, batch, head = search_configuration(method, arm)
            for part in ((chosen_part,) if chosen_part is not None else range(8)):
                dest = search_file(hand, gen, method, arm, obj, part)
                if dest.exists():
                    check_search(dest, hand, gen, method, arm, obj, part)
                    print(f"reuse {hand}/{gen}/{method}/{arm}/{obj}/part{part}", flush=True)
                    continue
                ids = indices(part)
                dest.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(driver), "--zero-file", str(source_file(hand, gen)),
                       "--objects", obj, "--max-grasps", "64", "--selection-battery", BATTERY,
                       "--validator-projection", "lsq", "--population", str(population),
                       "--generations", str(generations), "--elites", "8", "--restarts", "1",
                       "--batch-source-grasps", str(batch), "--keep-per-grasp", "3",
                       "--keep-per-object", str(3 * len(ids)),
                       "--cpu-final-candidates", str(3 * len(ids)),
                       "--search-pipeline", "gpu", "--reuse-gpu-sim", "--log-rejected",
                       "--seed", str(10001 + 1009 * part),
                       "--source-indices", *(str(i) for i in ids), "--out", str(dest)]
                if head is not None:
                    cmd.extend(("--warm-start-file", str(warm_file(hand, gen, head))))
                with dest.with_suffix(".log").open("w") as log:
                    status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode or not dest.is_file():
                    raise RuntimeError(f"Search failed; inspect {dest.with_suffix('.log')}")
                check_search(dest, hand, gen, method, arm, obj, part)
                print(f"searched LSQ {hand}/{gen}/{method}/{arm}/{obj}/part{part}", flush=True)
