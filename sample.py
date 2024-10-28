import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from utils.utils import instantiate_from_config
import os
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import torchvision

workspace = Path("~/.workspace").expanduser().as_posix()

""" global params """
BATCH_SIZE = 1
CONFIG_FILE_PATH = "configs/HDM_2D_Polyp_sample_features.yaml"
# CONFIG_FILE_PATH = "configs/HDM_SUN-SEG_sample.yaml"
RESUME_PATH = f"xxx" # TODO: change this
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size",
    type=int,
    help="batch size",
    default=BATCH_SIZE,
)
parser.add_argument(
    "--config_file",
    type=str,
    help="the path of the config file",
    default=CONFIG_FILE_PATH,
)
parser.add_argument(
    "--resume_path",
    type=str,
    help="the path of the ckpt",
    default=RESUME_PATH,
)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print(m, u)
    model.to("cuda:0")
    model.eval()
    return model


def get_model(args):
    config = OmegaConf.load(args.config_file)
    model = load_model_from_config(config, Path(workspace).joinpath(args.resume_path).as_posix())
    return model


def get_data(args):
    config = OmegaConf.load(args.config_file)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataloader = DataLoader(
        data.datasets[split],
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False,
    )

    return train_dataloader


def log_local(save_dir, images, batch_idx, prefix="inputs"):
    root = os.path.join(save_dir)

    grid = torchvision.utils.make_grid(images, nrow=4)
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    filename = "{}_b-{:06}.png".format(prefix, batch_idx)
    path = os.path.join(root, filename)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)


if __name__ == "__main__":
    # save_type = "png" # for visualization only
    save_type = "pt"

    data_type = "2D_Polyp"
    # data_type = "SUN-SEG"

    split = "train"
    # split = "validation"

    feature_type = "model_output"

    args = parser.parse_args()
    model = get_model(args)
    train_dataloader = get_data(args)

    result_dir = Path(workspace).joinpath(
        Path(args.resume_path).parent.parent.as_posix(), f"saved_features_{save_type}", split, feature_type
    )

    ### save features
    os.makedirs(result_dir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for idx, batch in enumerate(train_dataloader):
                name = batch["image_name"][0]
                if data_type == "SUN-SEG":
                    parent_name = batch["image_parent_name"][0]
                out = model.get_input(batch, "image")
                x, c, masks = out

                images = model.log_features(x, c, feature_type=feature_type)
                if isinstance(images, torch.Tensor):
                    images = images.detach().cpu()
                    images = torch.clamp(images, -1.0, 1.0)

                images = (images + 1.0) / 2.0  # (-1,1) --> (0,1)

                if save_type == "png":
                    images = images[0].transpose(0, 1).transpose(1, 2).numpy()
                    images = (images * 255).astype(np.uint8)

                if data_type == "SUN-SEG":
                    _result_dir = Path(result_dir).joinpath(parent_name).as_posix()
                    os.makedirs(_result_dir, exist_ok=True)
                    result_path = Path(_result_dir).joinpath(name).as_posix()
                else:
                    result_path = Path(result_dir).joinpath(name).as_posix()

                if save_type == "png":
                    Image.fromarray(images).save(f"{result_path}")
                else:
                    torch.save(images[0], f"{result_path[:-4]}.pt")

                # print(idx)
    ###
