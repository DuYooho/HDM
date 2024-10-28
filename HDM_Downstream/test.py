import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PVT.lib.pvt_cross import PolypPVT
from PraNet.lib.PraNet_Res2Net_cross import PraNet
from FCBFormer.Models.models_cross_attn import FCBFormer
import numpy as np
from PIL import Image
import os
from pathlib import Path
from global_config import workspace, parser_config, EXP
from data.polyp_all import PolypBase


def get_data(args):
    CVC_300_dataset = PolypBase(
        name="2D_polyp_test_seperate/CVC-300",
        size=args.trainsize,
        use_processed_features=True,
    )
    CVC_ClinicDB_dataset = PolypBase(
        name="2D_polyp_test_seperate/CVC-ClinicDB",
        size=args.trainsize,
        use_processed_features=True,
    )
    Kvasir_dataset = PolypBase(
        name="2D_polyp_test_seperate/Kvasir",
        size=args.trainsize,
        use_processed_features=True,
    )
    CVC_ColonDB_dataset = PolypBase(
        name="2D_polyp_test_seperate/CVC-ColonDB",
        size=args.trainsize,
        use_processed_features=True,
    )
    ETIS_LaribPolypDB_dataset = PolypBase(
        name="2D_polyp_test_seperate/ETIS-LaribPolypDB",
        size=args.trainsize,
        use_processed_features=True,
    )

    CVC_300_dataloader = DataLoader(
        dataset=CVC_300_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )
    CVC_ClinicDB_dataloader = DataLoader(
        dataset=CVC_ClinicDB_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )
    Kvasir_dataloader = DataLoader(
        dataset=Kvasir_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )
    CVC_ColonDB_dataloader = DataLoader(
        dataset=CVC_ColonDB_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )
    ETIS_LaribPolypDB_dataloader = DataLoader(
        dataset=ETIS_LaribPolypDB_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )

    dataloaders = {
        "CVC_300": CVC_300_dataloader,
        "CVC_ClinicDB": CVC_ClinicDB_dataloader,
        "Kvasir": Kvasir_dataloader,
        "CVC_ColonDB": CVC_ColonDB_dataloader,
        "ETIS_LaribPolypDB": ETIS_LaribPolypDB_dataloader,
    }

    return dataloaders


if __name__ == "__main__":
    args = parser_config()
    dataloaders = get_data(args)

    if args.model_name == "PVT":
        model = PolypPVT().cuda()
        ckpt = torch.load(Path(workspace).joinpath("logs", "PVT", EXP, f"PVT-best.pth").as_posix())
        msg = model.load_state_dict(ckpt)
        print(f"Loaded PVT ckpt: {msg}.")
        model.eval()

        for dataset in ["CVC_300", "CVC_ClinicDB", "Kvasir", "CVC_ColonDB", "ETIS_LaribPolypDB"]:
            save_path = Path(workspace).joinpath("results", "PVT", EXP, dataset.replace("_", "-")).as_posix()
            os.makedirs(save_path, exist_ok=True)
            test_loader = dataloaders[dataset]

            with torch.no_grad():
                for example in test_loader:
                    image, gt_shape, name, gt, features = (
                        example["image"],
                        example["ori_shape"],
                        example["image_name"],
                        example["mask"],
                        example["features"],
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
                    Image.fromarray(res).save(os.path.join(save_path, name[0]))
                    print(f"Finish {dataset}/{name}")

    elif args.model_name == "PraNet":
        model = PraNet().cuda()
        ckpt = torch.load(Path(workspace).joinpath("logs", "PraNet", EXP, f"PraNet-best.pth").as_posix())
        msg = model.load_state_dict(ckpt)
        print(f"Loaded PraNet ckpt: {msg}.")
        model.eval()

        for dataset in ["CVC_300", "CVC_ClinicDB", "Kvasir", "CVC_ColonDB", "ETIS_LaribPolypDB"]:
            save_path = Path(workspace).joinpath("results", "PraNet", EXP, dataset.replace("_", "-")).as_posix()
            os.makedirs(save_path, exist_ok=True)
            test_loader = dataloaders[dataset]

            with torch.no_grad():
                for example in test_loader:
                    image, gt_shape, name, gt, features = (
                        example["image"],
                        example["ori_shape"],
                        example["image_name"],
                        example["mask"],
                        example["features"],
                    )

                    image = image.cuda()

                    features = features.cuda()
                    features = F.interpolate(
                        features,
                        size=(32, 32),
                        mode="bilinear",
                        align_corners=True,
                    )

                    res5, res4, res3, res2 = model(image, features)
                    res = res2
                    res = F.interpolate(
                        res, size=(gt_shape[0][0], gt_shape[0][1]), mode="bilinear", align_corners=False
                    )
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                    res = ((res > 0.5) * 255).astype(np.uint8)
                    Image.fromarray(res).save(os.path.join(save_path, name[0]))
                    print(f"Finish {dataset}/{name}")

    elif args.model_name == "FCBFormer":
        model = FCBFormer().cuda()
        ckpt = torch.load(Path(workspace).joinpath("logs", "FCBFormer", EXP, f"FCBFormer-best.pth").as_posix())
        msg = model.load_state_dict(ckpt)
        print(f"Loaded FCBFormer ckpt: {msg}.")
        model.eval()

        for dataset in ["CVC_300", "CVC_ClinicDB", "Kvasir", "CVC_ColonDB", "ETIS_LaribPolypDB"]:
            save_path = Path(workspace).joinpath("results", "FCBFormer", EXP, dataset.replace("_", "-")).as_posix()
            os.makedirs(save_path, exist_ok=True)
            test_loader = dataloaders[dataset]

            with torch.no_grad():
                for example in test_loader:
                    image, gt_shape, name, gt, features = (
                        example["image"],
                        example["ori_shape"],
                        example["image_name"],
                        example["mask"],
                        example["features"],
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
                    Image.fromarray(res).save(os.path.join(save_path, name[0]))
                    print(f"Finish {dataset}/{name}")
