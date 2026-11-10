import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


NEW_ROOT = Path("/home/user/yangbo_program/new")
DRO_ROOT = Path("/home/user/yangbo_program/DRO-Grasp")
sys.path.insert(0, str(DRO_ROOT))

from model.network import create_network  # noqa: E402
from utils.hand_model import create_hand_model  # noqa: E402
from utils.multilateration import multilateration  # noqa: E402
from utils.optimization import create_problem, optimization, process_transform  # noqa: E402
from utils.se3_transform import compute_link_pose  # noqa: E402


TEST10_OBJECTS = [
    "contactdb+banana",
    "contactdb+bowl",
    "contactdb+knife",
    "contactdb+mug",
    "contactdb+scissors",
    "contactdb+toothbrush",
    "contactdb+torus_medium",
    "contactdb+water_bottle",
    "ycb+baseball",
    "ycb+mustard_bottle",
]


def load_object_pc(object_name, batch_size, device):
    dataset, name = object_name.split("+")
    path = DRO_ROOT / "data" / "PointCloud" / "object" / dataset / f"{name}.pt"
    pc = torch.load(path, map_location="cpu")[:, :3]
    return pc.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", default=TEST10_OBJECTS)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--out", type=str, default=str(NEW_ROOT / "results" / "pd_refiner_ability" / "ability_test10_zero_shot_b20.pt"))
    parser.add_argument("--summary", type=str, default=str(NEW_ROOT / "results" / "pd_refiner_ability" / "ability_test10_zero_shot_b20_summary.json"))
    parser.add_argument("--model", type=str, default=str(DRO_ROOT / "ckpt" / "model" / "model_3robots.pth"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None, center_pc=True, block_computing=True)
    network = create_network(cfg, mode="validate").to(device)
    network.load_state_dict(torch.load(args.model, map_location=device))
    network.eval()

    hand = create_hand_model("ability", device)
    layer = create_problem(hand.pk_chain, hand.links_pc.keys())

    all_results = []
    rows = []
    for object_name in args.objects:
        initial_q = torch.stack([hand.get_initial_q() for _ in range(args.batch_size)], dim=0).to(device)
        robot_pc = torch.stack([hand.get_transformed_links_pc(q)[:, :3] for q in initial_q], dim=0).to(device)
        object_pc = load_object_pc(object_name, args.batch_size, device)
        with torch.no_grad():
            dro = network(robot_pc, object_pc)["dro"].detach()
        mlat_pc = multilateration(dro, object_pc)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform, device=device)
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform).detach().cpu()
        result = {
            "robot_name": "ability",
            "object_name": object_name,
            "q_dro_order": predict_q,
            "joint_order": hand.get_joint_orders(),
        }
        all_results.append(result)
        row = {
            "object_name": object_name,
            "q_shape": list(predict_q.shape),
            "q_min": float(predict_q.min()),
            "q_max": float(predict_q.max()),
            "q_finite": bool(torch.isfinite(predict_q).all()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, out)
    summary = {
        "out": str(out),
        "robot_name": "ability",
        "batch_size": args.batch_size,
        "num_objects": len(all_results),
        "objects": args.objects,
        "joint_order": hand.get_joint_orders(),
        "rows": rows,
        "sample": {
            "object_name": all_results[0]["object_name"],
            "q0": [float(x) for x in all_results[0]["q_dro_order"][0].tolist()],
        },
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(out), "summary": args.summary}, indent=2))


if __name__ == "__main__":
    main()
