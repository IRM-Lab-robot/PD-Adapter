import sys
from pathlib import Path
from types import SimpleNamespace
import random

import numpy as np
import torch

DRO_ROOT = Path(__file__).resolve().parents[2] / "DRO-Grasp"
sys.path.insert(0, str(DRO_ROOT))

from model.network import create_network
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import create_problem, optimization, process_transform


def load_object_pc(object_name, batch_size, device):
    dataset, name = object_name.split("+")
    path = DRO_ROOT / "data" / "PointCloud" / "object" / dataset / f"{name}.pt"
    pc = torch.load(path, map_location="cpu")[:, :3]
    return pc.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    object_names = ["contactdb+apple", "contactdb+cylinder_medium", "ycb+baseball"]

    cfg = SimpleNamespace(
        emb_dim=512,
        latent_dim=64,
        pretrain=None,
        center_pc=True,
        block_computing=True,
    )
    network = create_network(cfg, mode="validate").to(device)
    ckpt_path = DRO_ROOT / "ckpt" / "model" / "model_3robots.pth"
    network.load_state_dict(torch.load(ckpt_path, map_location=device))
    network.eval()

    hand = create_hand_model("inspire", device)
    layer = create_problem(hand.pk_chain, hand.links_pc.keys())

    all_results = []
    for object_name in object_names:
        initial_q = torch.stack([hand.get_initial_q() for _ in range(batch_size)], dim=0).to(device)
        robot_pc = torch.stack(
            [hand.get_transformed_links_pc(q)[:, :3] for q in initial_q],
            dim=0,
        ).to(device)
        object_pc = load_object_pc(object_name, batch_size, device)

        with torch.no_grad():
            dro = network(robot_pc, object_pc)["dro"].detach()

        mlat_pc = multilateration(dro, object_pc)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform, device=device)
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform).detach().cpu()

        all_results.append(
            {
                "robot_name": "inspire",
                "object_name": object_name,
                "q_dro_order": predict_q,
                "joint_order": hand.get_joint_orders(),
            }
        )
        print(f"generated {predict_q.shape[0]} grasps for {object_name}")

    out_path = Path(__file__).resolve().parent / "inspire_zero_shot_grasps.pt"
    torch.save(all_results, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
