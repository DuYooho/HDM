import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PVT.lib.pvt_cross import PolypPVT
from FCBFormer.Models.models_cross_attn import FCBFormer
import numpy as np
from PIL import Image
import os
from pathlib import Path
from global_config import workspace, parser_config, EXP
from data.polyp_all import PolypBase


def get_data(args):
    test_dataset = PolypBase(
        name="SUN-SEG/TestUnseenDataset",
        size=args.trainsize,
        use_processed_features=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )

    return test_dataloader


if __name__ == "__main__":
    args = parser_config()
    test_loader = get_data(args)

    if args.model_name == "PVT":
        model = PolypPVT().cuda()
        ckpt = torch.load(Path(workspace).joinpath("logs", "PVT", EXP, f"PVT-best.pth").as_posix())
        msg = model.load_state_dict(ckpt)
        print(f"Loaded PVT ckpt: {msg}.")
        model.eval()

        save_path = Path(workspace).joinpath("results", "PVT", EXP, "TestUnseenDataset").as_posix()
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            for example in test_loader:
                image, gt_shape, name, gt, features, parent_name = (
                    example["image"],
                    example["ori_shape"],
                    example["image_name"],
                    example["mask"],
                    example["features"],
                    example["image_parent_name"],
                )

                image = image.cuda()
                features = features.cuda()
                features = F.interpolate(
                    features,
                    size=(32, 32),
                    mode="bilinear",
                    align_corners=True,
                )

                P1, P2 = model(image, features)
                res = F.interpolate(
                    P1 + P2, size=(gt_shape[0][0], gt_shape[0][1]), mode="bilinear", align_corners=False
                )
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                res = (res * 255).astype(np.uint8)
                save_dir = Path(save_path).joinpath(parent_name[0]).as_posix()
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(res).save(os.path.join(save_dir, name[0]))
                print(f"Finish {parent_name[0]}/{name[0]}")

    elif args.model_name == "FCBFormer":
        model = FCBFormer().cuda()
        ckpt = torch.load(Path(workspace).joinpath("logs", "FCBFormer", EXP, f"FCBFormer-best.pth").as_posix())
        msg = model.load_state_dict(ckpt)
        print(f"Loaded FCBFormer ckpt: {msg}.")
        model.eval()

        save_path = Path(workspace).joinpath("results", "FCBFormer", EXP, "TestUnseenDataset").as_posix()
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            for example in test_loader:
                image, gt_shape, name, gt, features, parent_name = (
                    example["image"],
                    example["ori_shape"],
                    example["image_name"],
                    example["mask"],
                    example["features"],
                    example["image_parent_name"],
                )

                image = image.cuda()
                features = features.cuda()
                features = F.interpolate(
                    features,
                    size=(32, 32),
                    mode="bilinear",
                    align_corners=True,
                )

                res = model(image, features)
                res = F.interpolate(res, size=(gt_shape[0][0], gt_shape[0][1]), mode="bilinear", align_corners=True)

                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                res = ((res > 0.5) * 255).astype(np.uint8)
                save_dir = Path(save_path).joinpath(parent_name[0]).as_posix()
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(res).save(os.path.join(save_dir, name[0]))
                print(f"Finish {parent_name[0]}/{name[0]}")
