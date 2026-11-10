#!/usr/bin/env python3
"""Inspire DRO/GenDex equal-288 ablations on test10×64 under the six-force battery.

no-wrist: keep the original trained model's eight predicted commands but use
the projected source wrist as each learned branch's search mean. Search may
still move the wrist. no-set: use the already-trained single-seed K8 model
whose all-head/all-teacher mean loss replaces bidirectional set matching.
k1-equal9: train K=1 on the main model's teacher/split/seed, then use its one
learned prediction in eight independently randomized search slots.

All variants search eight P32×G1 slots (K1 repeats the same K1 prediction);
the identical historical PD-9 default-branch search is reused and audited.
All 640 targets receive a new single, tail-padded batch32 CPU-state recheck.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DRO import run_pd288_five as pd

ENGINE_PATH = ROOT / "DRO/run_sample288_six.py"
spec = importlib.util.spec_from_file_location("inspire_ablation_recheck_engine", ENGINE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

OUT = ROOT / "results/supplement_inspire_pd288_ablations_b64"
NO_SET = ROOT / "supplement_inspire_dro_1/standard_ablation_no_set/train"
MAIN_MODEL = ROOT / "results/supplement_inspire_standard/train/best_pi_phi_v5_multihead.pt"
BASELINE = ROOT / "results/supplement_pd288_five_b64"
KINDS = ("no-wrist", "no-set", "k1-equal9")
TRAIN_SEED = 20262040
K1_TRAIN = OUT / "k1-equal9/train"
LINES = ("inspire-dro", "inspire-gen")
OBJECTS = engine.OBJECTS
ARMS = tuple(f"head{h}" for h in range(8))
PART_SIZE = 8
POPULATION = 32
BATTERY = engine.BATTERY


def configure(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(kind)
    engine.OUT = OUT / kind
    engine.METHOD = f"PD-9/288-{kind}"
    engine.METHOD_SLUG = f"pd288_{kind.replace('-', '_')}"
    engine.WORKER_ENTRYPOINT = Path(__file__).resolve()
    engine.WORKER_ARGS = ("--variant", kind)
    engine.collect = collect


def model_path(kind: str) -> Path:
    if kind == "no-wrist":
        return MAIN_MODEL
    if kind == "no-set":
        return NO_SET / "best_pi_phi_v5_multihead.pt"
    if kind == "k1-equal9":
        return K1_TRAIN / "best.pt"
    raise ValueError(kind)


def model_sha(kind: str) -> str:
    path = model_path(kind)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = engine.sha(path)
    if kind == "no-set":
        meta = json.loads((NO_SET / "metrics.json").read_text())
        if (meta.get("checkpoint_sha256") != digest
                or meta.get("objective") != "mean_over_all_heads_and_all_teacher_candidates"
                or meta.get("seed") != 20262040
                or meta.get("teacher_sha256") != engine.sha(
                    ROOT / "results/supplement_inspire_standard/teacher_object_split.pt")
                or meta.get("K") != 8 or meta.get("battery") != BATTERY):
            raise ValueError("The single-seed no-set training checkpoint changed")
    elif kind == "k1-equal9":
        meta = json.loads((K1_TRAIN / "metrics.json").read_text())
        if (meta.get("checkpoint_sha256") != digest or meta.get("K") != 1
                or meta.get("seed") != TRAIN_SEED or meta.get("train_groups") != 564
                or meta.get("val_groups") != 244 or meta.get("battery") != BATTERY
                or meta.get("baseline_model_sha256") != engine.sha(MAIN_MODEL)
                or meta.get("teacher_sha256") != engine.sha(
                    ROOT / "results/supplement_inspire_standard/teacher_object_split.pt")):
            raise ValueError("The matched-seed K1 training checkpoint changed")
    else:
        info = json.loads((ROOT / "results/supplement_inspire_standard/seed_summary.json").read_text())
        if info.get("model_sha256") != digest:
            raise ValueError("The main K8 Inspire checkpoint changed")
    return digest


def train_k1() -> None:
    """Retrain K=1 on precisely the main K8 teacher and object-disjoint split."""
    import torch
    from torch.utils.data import DataLoader

    checkpoint = model_path("k1-equal9")
    meta_path = K1_TRAIN / "metrics.json"
    if meta_path.is_file():
        model_sha("k1-equal9")
        print(f"reuse matched-seed K1 checkpoint: {checkpoint}", flush=True)
        return
    if checkpoint.exists():
        raise FileExistsError(f"Incomplete K1 training: inspect {checkpoint} before retrying")
    teacher = ROOT / "results/supplement_inspire_standard/teacher_object_split.pt"
    data = torch.load(teacher, map_location="cpu", weights_only=False)
    if (set(data["summary"]["train_objects"]) & set(data["summary"]["val_objects"])
            or set(OBJECTS) & set(data["summary"]["train_objects"] + data["summary"]["val_objects"])):
        raise ValueError("Training/validation/test objects overlap")
    from results.supplement_inspire_standard import train_standard_refiner as trainer

    base = trainer.load_training_module()
    base.set_seed(TRAIN_SEED)
    qlo, qhi, ulo, uhi = base.v4_mod.load_joint_limits()
    coupling = base.v4_mod.InspireCoupling(qlo, qhi).cpu().eval()
    tr, _ = base.build_groups(data["train"], coupling, ulo, uhi)
    va, _ = base.build_groups(data["val"], coupling, ulo, uhi)
    if (len(tr) != 564 or len(va) != 244
            or {g["object_name"] for g in tr} & {g["object_name"] for g in va}):
        raise ValueError("K1 and main K8 teacher groups differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.PiPhiV5MultiHead(base.v4_mod.InspireCoupling(qlo, qhi).to(device),
                                   ulo, uhi, base.DELTA_SCALE, k=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    train_loader = DataLoader(base.MultiEliteGroupDataset(tr), batch_size=32, shuffle=True,
                              collate_fn=base.collate_batch, num_workers=0)
    val_loader = DataLoader(base.MultiEliteGroupDataset(va), batch_size=32, shuffle=False,
                            collate_fn=base.collate_batch, num_workers=0)
    K1_TRAIN.mkdir(parents=True, exist_ok=True)
    best, best_epoch, log = float("inf"), None, []
    for epoch in range(1, 201):
        train_metrics = base.run_epoch(model, train_loader, device, optimizer)
        val_metrics = base.run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        log.append(row)
        if val_metrics["loss"] < best:
            best, best_epoch = val_metrics["loss"], epoch
            base.save_checkpoint(checkpoint, model, epoch, row)
        if epoch % 10 == 0:
            print(f"K1 epoch {epoch}: val={val_metrics['loss']:.6f} "
                  f"best={best:.6f} (epoch {best_epoch})", flush=True)
    engine.atomic_json(K1_TRAIN / "train_log.json", log)
    engine.atomic_json(meta_path, {
        "K": 1, "seed": TRAIN_SEED, "train_groups": len(tr), "val_groups": len(va),
        "best_epoch": best_epoch, "best_val_loss": best,
        "checkpoint_sha256": engine.sha(checkpoint), "teacher_sha256": engine.sha(teacher),
        "baseline_model_sha256": model_sha("no-wrist"), "battery": BATTERY,
        "selection": "best validation loss in 200 epochs; same teacher, split and seed as main K8",
        "search": "one K1 prediction repeated in eight independently randomized P32 search slots"})
    print(f"trained K1 epoch {best_epoch}: {checkpoint}", flush=True)


def generator(line: str) -> str:
    return "dro" if line == "inspire-dro" else "gendex"


def folder(kind: str, line: str) -> Path:
    return OUT / kind / line


def warm_path(kind: str, line: str, head: int) -> Path:
    return folder(kind, line) / "warmstarts" / f"pd_{generator(line)}_head{head}.pt"


def manifest_path(kind: str, line: str) -> Path:
    return folder(kind, line) / "warmstarts/manifest.json"


def search_path(kind: str, line: str, obj: str, head: int, part: int) -> Path:
    return (folder(kind, line) / "search" / f"head{head}" / engine.shared.metrics.safe(obj)
            / f"part{part:02d}_s{part*PART_SIZE:02d}_e{part*PART_SIZE+PART_SIZE-1:02d}.pt")


def expected_keys():
    return {(obj, sid) for obj in OBJECTS for sid in range(64)}


def search_seed(kind: str, slot: int, part: int) -> int:
    # K1 uses one frozen network prediction in eight *independent* stochastic
    # searches; the other two variants retain the historical search seeds.
    return (11001 + 1009 * slot + 100003 * part
            if kind == "k1-equal9" else 10001 + 1009 * part)


def baseline_sources(line: str) -> str:
    sha = engine.shared.check_source(line)
    if pd.check_source(line) != sha:
        raise ValueError(f"Merged source differs from frozen PD baseline for {line}")
    return sha


def prepare(kind: str, line: str) -> None:
    import torch

    source_sha, checkpoint_sha = baseline_sources(line), model_sha(kind)
    manifest = manifest_path(kind, line)
    if manifest.exists():
        check_warm(kind, line)
        print(f"reuse {kind}/{line} eight complete warm-starts", flush=True)
        return
    seeds_by_head = {head: [] for head in range(8)}
    if kind == "no-wrist":
        # Reuse exactly the baseline K8 command prediction for each of 640 IDs.
        baseline_manifest = BASELINE / line / "warmstarts/manifest.json"
        info = json.loads(baseline_manifest.read_text())
        if info.get("source_sha256") != source_sha or info.get("model_sha256") != checkpoint_sha:
            raise ValueError(f"Wrong baseline warm-start lineage: {baseline_manifest}")
        for head in range(8):
            path = BASELINE / line / "warmstarts" / f"pd_{generator(line)}_head{head}.pt"
            if engine.sha(path) != info["warm_sha256"][head]:
                raise ValueError(f"Modified baseline warm-start: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["seeds"]:
                params = torch.as_tensor(row["params"], dtype=torch.float32).view(12).clone()
                source = torch.as_tensor(row["source_q_seed"], dtype=torch.float32).view(18)
                params[:6] = source[:6]
                seeds_by_head[head].append({**row, "params": params})
    else:
        # Exactly the same projection, point cloud and network inference path as
        # the main 640-target PD baseline; only K or the objective differs.
        sys.path.insert(0, str(ROOT / "src/eval"))
        import generate_pi_phi_warmstart_step12 as warm

        warm.MODEL_PATH = model_path(kind)
        model, checkpoint = warm.load_model()
        learned_heads = 1 if kind == "k1-equal9" else 8
        if checkpoint["K"] != learned_heads:
            raise ValueError(f"Wrong number of trained heads in {model_path(kind)}")
        hand = warm.create_hand_model("inspire", device=torch.device("cpu"))
        finger_names = hand.get_joint_orders()[6:]
        lower, upper = hand.pk_chain.get_joint_limits()
        for row in pd.sources.source_rows(pd.source_file(line), "inspire", 64):
            obj = row["object_name"]
            raw = torch.as_tensor(row["q_dro_order"], dtype=torch.float32)
            q_seed, _ = warm.lsq_project_q_local(raw, finger_names, lower[6:], upper[6:])
            points = warm.load_object_points(obj).unsqueeze(0)
            with torch.no_grad():
                for sid in range(64):
                    wrist = q_seed[sid, :6].reshape(1, 6)
                    pred = model(points, q_seed[sid].reshape(1, 18), wrist)
                    for slot in range(8):
                        head = 0 if kind == "k1-equal9" else slot
                        seeds_by_head[slot].append({
                            "object_name": obj, "source_grasp_index": sid,
                            "generator": "DRO" if generator(line) == "dro" else "GenDexGrasp",
                            "head_id": head, "slot_id": slot, "source_q_seed": q_seed[sid].clone(),
                            "params": torch.cat([pred["pred_w"][0, head],
                                                 pred["pred_u"][0, head]]).cpu().float()})

    warm_hashes = []
    for head, seeds in seeds_by_head.items():
        if len(seeds) != 640 or {(r["object_name"], int(r["source_grasp_index"])) for r in seeds} != expected_keys():
            raise ValueError(f"Incomplete {kind}/{line}/head{head} warm-start predictions")
        path = warm_path(kind, line, head)
        payload = {"info": {"kind": kind, "line": line, "head_id": 0 if kind == "k1-equal9" else head,
                            "slot_id": head, "trained_heads": 1 if kind == "k1-equal9" else 8,
                            "model_sha256": checkpoint_sha, "source_sha256": source_sha,
                            "ablation": "learned wrist reset; command unchanged" if kind == "no-wrist"
                                        else ("one K1 proposal in eight independent random search slots"
                                              if kind == "k1-equal9" else
                                              "all-head/all-teacher mean matching, seed 20262040")},
                   "seeds": seeds}
        if path.exists():
            prior = torch.load(path, map_location="cpu", weights_only=False)
            if (prior["info"] != payload["info"] or len(prior["seeds"]) != 640
                    or any(not torch.equal(torch.as_tensor(a["params"]), b["params"])
                           for a, b in zip(prior["seeds"], seeds))):
                raise ValueError(f"Conflicting partial warm-start: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(path)
        warm_hashes.append(engine.sha(path))
    engine.atomic_json(manifest, {"kind": kind, "line": line,
                                 "source_sha256": source_sha, "model_sha256": checkpoint_sha,
                                 "baseline_model_sha256": model_sha("no-wrist"),
                                 "warm_sha256": warm_hashes, "heads": 8,
                                 "trained_heads": 1 if kind == "k1-equal9" else 8,
                                 "targets_per_head": 640, "search_gpu_evaluations_per_source": 288})
    check_warm(kind, line)
    print(f"prepared {kind}/{line}: eight search slots × 640 source IDs", flush=True)


def check_warm(kind: str, line: str) -> str:
    import torch

    source_sha, checkpoint_sha = baseline_sources(line), model_sha(kind)
    manifest = json.loads(manifest_path(kind, line).read_text())
    if (manifest.get("kind") != kind or manifest.get("line") != line
            or manifest.get("source_sha256") != source_sha
            or manifest.get("model_sha256") != checkpoint_sha
            or manifest.get("baseline_model_sha256") != model_sha("no-wrist")
            or manifest.get("heads") != 8 or manifest.get("targets_per_head") != 640
            or manifest.get("trained_heads", 8) != (1 if kind == "k1-equal9" else 8)
            or manifest.get("search_gpu_evaluations_per_source") != 288):
        raise ValueError(f"Wrong ablation warm-start manifest: {kind}/{line}")
    keys = expected_keys()
    for head in range(8):
        path = warm_path(kind, line, head)
        if not path.is_file() or engine.sha(path) != manifest["warm_sha256"][head]:
            raise ValueError(f"Missing/changed ablation head: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if (data["info"].get("model_sha256") != checkpoint_sha
                or data["info"].get("source_sha256") != source_sha
                or data["info"].get("kind") != kind
                or data["info"].get("head_id") != (0 if kind == "k1-equal9" else head)
                or data["info"].get("slot_id", head) != head
                or data["info"].get("trained_heads", 8) != (1 if kind == "k1-equal9" else 8)):
            raise ValueError(f"Wrong head provenance: {path}")
        seeds = data["seeds"]
        if (len(seeds) != 640 or {(r["object_name"], int(r["source_grasp_index"])) for r in seeds} != keys
                or any(torch.as_tensor(r["params"]).numel() != 12 for r in seeds)):
            raise ValueError(f"Incomplete prediction coverage: {path}")
        if kind == "no-wrist":
            original = torch.load(BASELINE / line / "warmstarts" / f"pd_{generator(line)}_head{head}.pt",
                                  map_location="cpu", weights_only=False)["seeds"]
            if any(not torch.equal(torch.as_tensor(a["params"])[6:], torch.as_tensor(b["params"])[6:])
                   or not torch.equal(torch.as_tensor(a["params"])[:6], torch.as_tensor(a["source_q_seed"])[:6])
                   for a, b in zip(seeds, original)):
                raise ValueError(f"No-wrist variant changed a baseline command: {path}")
        elif kind == "no-set" and line == "inspire-dro":
            # Independently reconstruct the prior 10×20 no-set experiment.
            # BLAS kernels may differ at the last float32 bit across batches.
            previous = torch.load(
                ROOT / "supplement_inspire_dro_1/standard_ablation_no_set/warmstarts"
                / f"pd_dro_head{head}.pt", map_location="cpu", weights_only=False)["seeds"]
            by_key = {(r["object_name"], int(r["source_grasp_index"])): r for r in seeds}
            if (len(previous) != 200 or any(
                    not torch.allclose(torch.as_tensor(old["params"]),
                                       torch.as_tensor(by_key[(old["object_name"],
                                                              int(old["source_grasp_index"]))]["params"]),
                                       rtol=0, atol=1e-6)
                    for old in previous)):
                raise ValueError(f"No-set old20 predictions differ from the prior ablation: {path}")
        elif kind == "k1-equal9" and head:
            original = torch.load(warm_path(kind, line, 0), map_location="cpu", weights_only=False)["seeds"]
            if any(not torch.equal(torch.as_tensor(a["params"]), torch.as_tensor(b["params"]))
                   for a, b in zip(seeds, original)):
                raise ValueError(f"K1 search slots contain distinct learned proposals: {path}")
    return source_sha


def expected_wrist(line: str, obj: str, part: int):
    import torch

    old = torch.as_tensor(pd.check_old(line, obj, "default")["q_seed"], dtype=torch.float32)
    added = {}
    rows = []
    for sid in range(part * PART_SIZE, (part + 1) * PART_SIZE):
        if sid < 20:
            rows.append(old[sid])
        else:
            block = (sid - 20) // PART_SIZE
            if block not in added:
                added[block] = torch.as_tensor(pd.check_new(line, obj, "default", block)["q_seed"],
                                               dtype=torch.float32)
            rows.append(added[block][(sid - 20) % PART_SIZE])
    return torch.stack(rows)


def check_search(kind: str, line: str, obj: str, head: int, part: int):
    import torch

    path = search_path(kind, line, obj, head, part)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    ids = list(range(part * PART_SIZE, (part + 1) * PART_SIZE))
    args = data["args"]
    expected = {"zero_file": str(pd.source_file(line)), "selection_battery": BATTERY,
                "max_grasps": 64, "source_indices": ids, "population": POPULATION,
                "generations": 1, "elites": 8, "restarts": 1,
                "batch_source_grasps": PART_SIZE, "keep_per_grasp": 3,
                "keep_per_object": 3 * PART_SIZE, "cpu_final_candidates": 3 * PART_SIZE,
                "search_pipeline": "gpu", "reuse_gpu_sim": True, "log_rejected": True,
                "seed": search_seed(kind, head, part), "warm_start_file": str(warm_path(kind, line, head)),
                "arm_id": f"{kind}_{generator(line)}_head{head}", "allow_non_train": True,
                "init_std_scale": 1.0, "wrist_xyz_std": .01, "wrist_rot_std": .10, "u_std": .25,
                "spatial_sampling": "local"}
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Wrong ablation search configuration: {path}")
    if data.get("objects") != [obj] or data.get("total_source_grasps") != PART_SIZE:
        raise ValueError(f"Incomplete ablation search: {path}")
    row = data["rows"][0]
    if ([int(g["grasp_index"]) for g in row["grasp_rows"]] != ids
            or not torch.equal(torch.as_tensor(row["q_seed"]), expected_wrist(line, obj, part))):
        raise ValueError(f"Ablation/source target mismatch: {path}")
    pd.check_candidate_row(path, row, set(ids), 3 * PART_SIZE, "inspire")
    return row


def search(kind: str, line: str, objects, arms, selected_part=None):
    check_warm(kind, line)
    driver = ROOT / "eval/generate_inspire_train_bsearch.py"
    env = dict(os.environ, CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9", MAX_JOBS="4")
    for obj in objects:
        for head in arms:
            for part in ((selected_part,) if selected_part is not None else range(8)):
                path = search_path(kind, line, obj, head, part)
                if path.exists():
                    check_search(kind, line, obj, head, part)
                    print(f"reuse {kind}/{line}/{obj}/head{head}/part{part}", flush=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                ids = list(range(part * PART_SIZE, (part + 1) * PART_SIZE))
                cmd = [sys.executable, str(driver), "--zero-file", str(pd.source_file(line)),
                       "--objects", obj, "--allow-non-train", "--max-grasps", "64",
                       "--selection-battery", BATTERY, "--population", str(POPULATION),
                       "--generations", "1", "--elites", "8", "--restarts", "1",
                       "--batch-source-grasps", str(PART_SIZE), "--keep-per-grasp", "3",
                       "--keep-per-object", str(3 * PART_SIZE), "--cpu-final-candidates", str(3 * PART_SIZE),
                       "--search-pipeline", "gpu", "--reuse-gpu-sim", "--log-rejected",
                       "--spatial-sampling", "local", "--seed", str(search_seed(kind, head, part)),
                       "--source-indices", *(str(i) for i in ids),
                       "--warm-start-file", str(warm_path(kind, line, head)),
                       "--arm-id", f"{kind}_{generator(line)}_head{head}", "--out", str(path)]
                with path.with_suffix(".log").open("w") as stream:
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if proc.returncode or not path.is_file():
                    raise RuntimeError(f"Search failed; inspect {path.with_suffix('.log')}")
                check_search(kind, line, obj, head, part)
                print(f"searched {kind}/{line}/{obj}/head{head}/part{part}", flush=True)


def collect(line: str, obj: str, source_sha: str):
    import torch

    prefix = "PD-9/288-"
    kind = engine.METHOD[len(prefix):] if engine.METHOD.startswith(prefix) else ""
    if kind not in KINDS or check_warm(kind, line) != source_sha:
        raise ValueError(f"Invalid ablation configuration: {kind}/{line}")
    candidates, pool_hashes = [], []
    for arm in ("default", *ARMS):
        parts = range(7) if arm == "default" else range(8)
        for part in parts:
            if arm == "default":
                if part == 0:
                    path = pd.old_path(line, obj, "default")
                    row = pd.check_old(line, obj, "default")
                    tag = "old20"
                else:
                    path = pd.new_path(line, obj, "default", part - 1)
                    row = pd.check_new(line, obj, "default", part - 1)
                    tag = f"new{part-1:02d}"
            else:
                head = int(arm[4:])
                path = search_path(kind, line, obj, head, part)
                row = check_search(kind, line, obj, head, part)
                tag = f"part{part:02d}"
            pool_hashes.append(engine.sha(path))
            for group in ("passing_candidates", "cpu_final_rejected_candidates"):
                for idx, cand in enumerate(row[group]):
                    params = torch.as_tensor(cand["params"], dtype=torch.float32).contiguous()
                    candidates.append({"id": f"{arm}/{tag}/{group}/{idx}",
                                       "source_index": int(cand["source_grasp_index"]),
                                       "q": torch.as_tensor(cand["q"], dtype=torch.float32).view(18),
                                       "gpu_rank_score": float(cand["gpu_rank_score"]),
                                       "search_params_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
                                       "search_cpu_success": bool(cand["cpu_success"])})
    if len(pool_hashes) != 7 + 8 * 8:
        raise ValueError("Incomplete nine-branch search pool")
    return candidates, pool_hashes


def paired_report(kind: str, line: str, obj: str) -> dict:
    path = engine.result_path(line, obj)
    result = json.loads(path.read_text())
    source_sha = check_warm(kind, line)
    candidates, hashes = collect(line, obj, source_sha)
    selected = engine.shared.metrics.freeze_top1(candidates)
    if (result.get("method") != engine.METHOD or result.get("line") != line
            or result.get("source_sha256") != source_sha
            or result.get("search_part_sha256") != hashes
            or result.get("battery") != BATTERY
            or result.get("validator_projection") != "lsq"
            or result.get("gpu_evaluations_per_source") != 288
            or result.get("revalidation_count") != 1 or result.get("verify_batch_size") != 32
            or result.get("tail_padded") is not True
            or result.get("candidate_ids") != [c["id"] for c in candidates]
            or result.get("top1_candidate_indices") != selected
            or result.get("top1_candidate_ids")
            != [candidates[i]["id"] if i is not None else None for i in selected]
            or len(result.get("candidate_success", [])) != len(candidates)):
        raise ValueError(f"Incomplete/incompatible ablation recheck: {path}")
    top1, multi = engine.shared.metrics.outcomes(candidates, result["candidate_success"], selected)
    if result.get("top1_success") != top1 or result.get("multi_success") != multi:
        raise ValueError(f"Ablation candidate labels and target outcomes differ: {path}")
    baseline_path = BASELINE / line / "revalidation_once" / f"{engine.shared.metrics.safe(obj)}.json"
    base = json.loads(baseline_path.read_text())
    if (base.get("line") != line or base.get("object_name") != obj
            or base.get("source_sha256") != source_sha or base.get("battery") != BATTERY
            or base.get("validator_projection", "lsq") != "lsq"
            or base.get("model_sha256") != model_sha("no-wrist")
            or base.get("verify_batch_size") != 32 or base.get("tail_padded") is not True
            or base.get("revalidation_count") != 1):
        raise ValueError(f"Wrong paired main-model baseline: {baseline_path}")
    n = base["candidate_count"]
    if any(len(base.get(key, [])) != n for key in (
            "candidate_ids", "source_indices", "candidate_gpu_rank_scores",
            "candidate_search_params_sha256", "candidate_success")):
        raise ValueError(f"Incomplete main-model candidate labels: {baseline_path}")
    base_candidates = [{"id": cid, "source_index": sid, "gpu_rank_score": float(score),
                        "search_params_sha256": digest}
                       for cid, sid, score, digest in zip(
                           base["candidate_ids"], base["source_indices"],
                           base["candidate_gpu_rank_scores"], base["candidate_search_params_sha256"])]
    base_selected = engine.shared.metrics.freeze_top1(base_candidates)
    base_top1, base_multi = engine.shared.metrics.outcomes(
        base_candidates, base["candidate_success"], base_selected)
    if (base.get("top1_candidate_indices") != base_selected
            or base.get("top1_success") != base_top1
            or base.get("multi_success") != base_multi):
        raise ValueError(f"Main-model top-1 or labels are not frozen: {baseline_path}")
    metrics = {}
    for name, ablation_labels in (("top1", top1), ("multi", multi)):
        baseline_labels = base[f"{name}_success"]
        if len(baseline_labels) != 64 or len(ablation_labels) != 64:
            raise ValueError("Expected 64 matched source IDs")
        metrics[name] = {"baseline": sum(baseline_labels), "ablation": sum(ablation_labels),
                         "old20_baseline": sum(baseline_labels[:20]),
                         "old20_ablation": sum(ablation_labels[:20]),
                         "new44_baseline": sum(baseline_labels[20:]),
                         "new44_ablation": sum(ablation_labels[20:]),
                         "gained": sum(not a and b for a, b in zip(baseline_labels, ablation_labels)),
                         "lost": sum(a and not b for a, b in zip(baseline_labels, ablation_labels)),
                         "total": 64, "delta_pp": (sum(ablation_labels) - sum(baseline_labels)) * 100 / 64,
                         "baseline_source_success": baseline_labels,
                         "ablation_source_success": ablation_labels}
    return {"kind": kind, "line": line, "object_name": obj,
            "source_sha256": source_sha, "model_sha256": model_sha(kind),
            "baseline_model_sha256": model_sha("no-wrist"),
            "trained_heads": 1 if kind == "k1-equal9" else 8,
            "independent_search_slots": 8,
            "search_seed_rule": "11001+1009*slot+100003*part" if kind == "k1-equal9"
                                else "10001+1009*part",
            "baseline_recheck_sha256": engine.sha(baseline_path),
            "ablation_recheck_sha256": engine.sha(path),
            "search_pool_sha256": hashes, "candidate_count": len(candidates),
            "battery": BATTERY, "verify_batch_size": 32,
            "gpu_evaluations_per_source": 288,
            "validator_projection": "lsq", "metrics": metrics}


def summarize(kind: str, line: str, obj: str = None) -> None:
    if obj is not None:
        row = paired_report(kind, line, obj)
        dest = folder(kind, line) / "reports" / f"{engine.shared.metrics.safe(obj)}.json"
        engine.atomic_json(dest, row)
        print(f"{kind}/{line}/{obj}: top-1 {row['metrics']['top1']['ablation']}/64, "
              f"multi {row['metrics']['multi']['ablation']}/64 | {dest}", flush=True)
        return
    reports = [paired_report(kind, line, obj) for obj in OBJECTS]
    rng = random.Random(20261221)
    draws = [[rng.randrange(10) for _ in range(10)] for _ in range(5000)]
    metrics = {}
    for name in ("top1", "multi"):
        items = [r["metrics"][name] for r in reports]
        differences = [r["delta_pp"] for r in items]
        boots = [sum(differences[i] for i in draw) / 10 for draw in draws]
        metrics[name] = {"baseline": sum(r["baseline"] for r in items),
                         "ablation": sum(r["ablation"] for r in items),
                         "old20_baseline": sum(r["old20_baseline"] for r in items),
                         "old20_ablation": sum(r["old20_ablation"] for r in items),
                         "new44_baseline": sum(r["new44_baseline"] for r in items),
                         "new44_ablation": sum(r["new44_ablation"] for r in items),
                         "gained": sum(r["gained"] for r in items),
                         "lost": sum(r["lost"] for r in items), "total": 640,
                         "delta_pp": sum(differences) / 10,
                         "object_cluster_95ci_pp": [engine.shared.metrics.quantile(boots, .025),
                                                    engine.shared.metrics.quantile(boots, .975)]}
    report = {"kind": kind, "line": line, "scope": "single-seed 10 objects × 64 source IDs",
              "trained_heads": 1 if kind == "k1-equal9" else 8,
              "independent_search_slots": 8,
              "search_seed_rule": "11001+1009*slot+100003*part" if kind == "k1-equal9"
                                  else "10001+1009*part",
              "source_sha256": reports[0]["source_sha256"],
              "model_sha256": reports[0]["model_sha256"],
              "baseline_model_sha256": reports[0]["baseline_model_sha256"],
              "battery": BATTERY, "validator_projection": "lsq",
              "gpu_evaluations_per_source": 288,
              "revalidation_count": 1, "verify_batch_size": 32, "tail_padded": True,
              "metrics": metrics, "per_object": {r["object_name"]: r for r in reports}}
    dest = folder(kind, line) / "summary.json"
    engine.atomic_json(dest, report)
    for name, row in metrics.items():
        print(f"{kind}/{line} {name}: {row['baseline']}/640 -> "
              f"{row['ablation']}/640 ({row['delta_pp']:+.2f} pp)", flush=True)
    print(f"saved {dest}", flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        if len(sys.argv) < 4 or sys.argv[-2] != "--variant":
            raise ValueError("Ablation worker requires --variant as its last argument")
        configure(sys.argv[-1])
        sys.argv = sys.argv[:-2]
        engine.main()  # Fresh validator/Isaac Gym import occurs before torch.
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "check", "prepare", "search", "recheck", "summarize", "all"))
    parser.add_argument("--variant", required=True, choices=KINDS)
    parser.add_argument("--line", required=True, choices=(*LINES, "all"))
    parser.add_argument("--object", choices=OBJECTS)
    parser.add_argument("--head", type=int, choices=range(8),
                        help="Search one slot (one learned head for K8, repeated K1 prediction for K1)")
    parser.add_argument("--part", type=int, choices=range(8),
                        help="Search one eight-source block; requires one line and object")
    parser.add_argument("--max-worker-groups", type=int)
    args = parser.parse_args()
    if args.part is not None and (args.stage != "search" or args.object is None or args.line == "all"):
        parser.error("--part requires search --line <one> --object <one>")
    if args.head is not None and args.stage != "search":
        parser.error("--head is only valid for search")
    if args.max_worker_groups is not None and (
            args.max_worker_groups < 1 or args.stage not in ("recheck", "all")):
        parser.error("--max-worker-groups requires recheck/all and a positive value")
    if args.object is not None and args.stage in ("train", "check", "prepare"):
        parser.error("--object does not apply to train/check/prepare")
    if args.stage == "train" and (args.variant != "k1-equal9" or args.line != "all"):
        parser.error("K1 training is shared by both generators: use train --variant k1-equal9 --line all")
    configure(args.variant)
    if args.stage == "train":
        train_k1()
        return
    if args.stage == "all" and args.variant == "k1-equal9":
        train_k1()
    for line in (LINES if args.line == "all" else (args.line,)):
        if args.stage == "check":
            check_warm(args.variant, line)
        elif args.stage == "prepare":
            prepare(args.variant, line)
        elif args.stage == "search":
            search(args.variant, line, (args.object,) if args.object else OBJECTS,
                   (args.head,) if args.head is not None else range(8), args.part)
        elif args.stage == "recheck":
            check_warm(args.variant, line)
            engine.recheck(line, (args.object,) if args.object else OBJECTS, args.max_worker_groups)
        elif args.stage == "summarize":
            summarize(args.variant, line, args.object)
        else:
            prepare(args.variant, line)
            search(args.variant, line, (args.object,) if args.object else OBJECTS, range(8))
            engine.recheck(line, (args.object,) if args.object else OBJECTS, args.max_worker_groups)
            if args.max_worker_groups is None:
                summarize(args.variant, line, args.object)


if __name__ == "__main__":
    main()
