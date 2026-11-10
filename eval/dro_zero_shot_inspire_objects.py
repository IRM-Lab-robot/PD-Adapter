import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

DRO_ROOT = Path(__file__).resolve().parents[2] / "DRO-Grasp"
sys.path.insert(0, str(DRO_ROOT))

from model.network import create_network
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.optimization import create_problem, optimization, process_transform
from utils.se3_transform import compute_link_pose


DEFAULT_OBJECTS = [
    "contactdb+banana",
    "contactdb+toothbrush",
    "contactdb+cell_phone",
    "contactdb+mug",
    "contactdb+pan",
    "contactdb+scissors",
    "ycb+cracker_box",
    "ycb+baseball",
]


def load_object_pc(object_name, batch_size, device):
    dataset, name = object_name.split("+")
    path = DRO_ROOT / "data" / "PointCloud" / "object" / dataset / f"{name}.pt"
    pc = torch.load(path, map_location="cpu")[:, :3]
    return pc.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--out", type=str, default="inspire_shape_sweep_zero_shot_grasps.pt")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", type=str, default=str(DRO_ROOT / "ckpt" / "model" / "model_3robots.pth"))
    parser.add_argument("--robot-name", type=str, default="inspire")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = SimpleNamespace(emb_dim=512, latent_dim=64, pretrain=None, center_pc=True, block_computing=True)
    network = create_network(cfg, mode="validate").to(device)
    ckpt_path = Path(args.model)
    network.load_state_dict(torch.load(ckpt_path, map_location=device))
    network.eval()

    hand = create_hand_model(args.robot_name, device)
    layer = create_problem(hand.pk_chain, hand.links_pc.keys())

    all_results = []
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
        all_results.append(
            {
                "robot_name": args.robot_name,
                "object_name": object_name,
                "q_dro_order": predict_q,
                "joint_order": hand.get_joint_orders(),
            }
        )
        print(f"generated {predict_q.shape[0]} grasps for {object_name}")

    out = Path(__file__).resolve().parent / args.out
    torch.save(all_results, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
