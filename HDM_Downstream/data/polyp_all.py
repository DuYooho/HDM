import torch
from torch.utils.data import Dataset, DataLoader
import albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from pathlib import Path
from global_config import workspace
from natsort import natsorted


class PolypBase(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.NAME = kwargs["name"]
        self.size = kwargs.pop("size", 352)
        self.use_processed_features = kwargs.pop("use_processed_features", False)

        self.preprocessor = albumentations.Compose(
            [albumentations.Resize(self.size, self.size), albumentations.Normalize(), ToTensorV2()]
        )
        self.preprocessor_resize = albumentations.Compose([albumentations.Resize(self.size, self.size), ToTensorV2()])

        self._prepare()

    def __len__(self):
        return len(self.images_list_absolute)

    def __getitem__(self, i):
        data = {}
        image_path = self.images_list_absolute[i]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)

        data["image_name"] = Path(self.images_list_absolute[i]).name
        if str(self.NAME).startswith("SUN-SEG"):
            data["image_parent_name"] = Path(self.images_list_absolute[i]).parent.name

        features = torch.load(self.features_list_absolute[i])
        data["features"] = features

        mask = Image.open(self.masks_list_absolute[i]).convert("L")
        mask = np.array(mask).astype(np.uint8)
        data["ori_shape"] = np.array(mask.shape)
        _preprocessor = self.preprocessor(image=image, mask=mask)
        image, mask = _preprocessor["image"], _preprocessor["mask"]
        mask = mask.unsqueeze(0)
        mask = mask / 255
        data["image"], data["mask"] = image, mask

        if self.use_processed_features:
            data["features"] = features * image

        return data

    def getitem(self, i):
        return self.__getitem__(i)

    def _prepare(self):
        if str(self.NAME).startswith("SUN-SEG"):
            self.root = Path(workspace).joinpath("datasets", self.NAME).as_posix()
            self.images_dir = Path(self.root).joinpath("Frame").as_posix()
            self.masks_dir = Path(self.root).joinpath("GT").as_posix()
            self.features_dir = Path(self.root).joinpath("Features").as_posix()
        else:
            self.root = Path(workspace).joinpath("datasets/diffusion_datasets", self.NAME).as_posix()
            self.images_dir = Path(self.root).joinpath("images").as_posix()
            self.masks_dir = Path(self.root).joinpath("masks").as_posix()
            self.features_dir = Path(self.root).joinpath("features").as_posix()

        print(f"Preparing dataset {self.NAME}")

        if str(self.NAME).startswith("SUN-SEG"):
            self.images_list_absolute = Path(self.images_dir).rglob("*.jpg")
        else:
            self.images_list_absolute = Path(self.images_dir).rglob("*.png")
        self.images_list_absolute = [file_path.as_posix() for file_path in self.images_list_absolute]
        self.images_list_absolute = natsorted(self.images_list_absolute)

        if str(self.NAME).startswith("SUN-SEG"):
            self.masks_list_absolute = [
                Path(self.masks_dir).joinpath(Path(p).parent.name, f"{Path(p).stem}.png").as_posix()
                for p in self.images_list_absolute
            ]
            self.features_list_absolute = [
                Path(self.features_dir).joinpath(Path(p).parent.name, f"{Path(p).stem}.pt").as_posix()
                for p in self.images_list_absolute
            ]
        else:
            self.masks_list_absolute = [
                Path(self.masks_dir).joinpath(f"{Path(p).stem}.png").as_posix() for p in self.images_list_absolute
            ]
            self.features_list_absolute = [
                Path(self.features_dir).joinpath(f"{Path(p).stem}.pt").as_posix() for p in self.images_list_absolute
            ]



