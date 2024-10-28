import sys

sys.path.append("../")
from torch.utils.data import Dataset
import albumentations
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from natsort import natsorted

workspace = Path("~/.workspace").expanduser().as_posix()


class PolypBase(Dataset):
    def __init__(self, config=None, **kwargs):
        self.kwargs = kwargs
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        self.NAME = kwargs["name"]
        self.size = kwargs.pop("size", 352)

        self.preprocessor = albumentations.Compose(albumentations.Resize(self.size, self.size))

        self._prepare()

    def __len__(self):
        return len(self.images_list_absolute)

    def __getitem__(self, i):
        data = {}
        image = Image.open(self.images_list_absolute[i]).convert("RGB")
        image = np.array(image).astype(np.uint8)
        data["image_name"] = Path(self.images_list_absolute[i]).name
        if str(self.NAME).startswith("SUN-SEG"):
            data["image_parent_name"] = Path(self.images_list_absolute[i]).parent.name

        hc = Image.open(self.hcs_list_absolute[i]).convert("RGB")
        hc = np.array(hc).astype(np.uint8)
        _preprocessor = self.preprocessor(image=image, mask=hc)
        image, hc = _preprocessor["image"], _preprocessor["mask"]
        data["image"], data["hc"] = image, hc

        return data

    def getitem(self, i):
        return self.__getitem__(i)

    def _prepare(self):
        if str(self.NAME).startswith("SUN-SEG"):
            self.root = Path(workspace).joinpath("datasets", self.NAME).as_posix()
            self.images_dir = Path(self.root).joinpath("Frame").as_posix()
            self.masks_dir = Path(self.root).joinpath("GT").as_posix()
            self.hcs_dir = Path(self.root).joinpath("Highlighted_GT").as_posix()
        else:
            self.root = Path(workspace).joinpath("datasets/diffusion_datasets", self.NAME).as_posix()
            self.images_dir = Path(self.root).joinpath("images").as_posix()
            self.masks_dir = Path(self.root).joinpath("masks").as_posix()
            self.hcs_dir = Path(self.root).joinpath("highlighted_GT").as_posix()

        print(f"Preparing dataset {self.NAME}")

        if str(self.NAME).startswith("SUN-SEG"):
            self.images_list_absolute = Path(self.images_dir).rglob("*.jpg")
        else:
            self.images_list_absolute = Path(self.images_dir).rglob("*.png")
        self.images_list_absolute = [file_path.as_posix() for file_path in self.images_list_absolute]
        self.images_list_absolute = natsorted(self.images_list_absolute)

        self.masks_list_absolute = [
            Path(self.masks_dir).joinpath(f"{Path(p).stem}.png").as_posix() for p in self.images_list_absolute
        ]

        if str(self.NAME).startswith("SUN-SEG"):
            self.hcs_list_absolute = [
                Path(self.hcs_dir).joinpath(Path(p).parent.name, Path(p).name).as_posix()
                for p in self.images_list_absolute
            ]
        else:
            self.hcs_list_absolute = [
                Path(self.hcs_dir).joinpath(Path(p).name).as_posix() for p in self.images_list_absolute
            ]
