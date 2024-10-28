# --------------------------------------------------------
# References:
# https://github.com/CompVis/taming-transformers/blob/master/taming/data/utils.py
# https://github.com/CompVis/taming-transformers/blob/master/taming/util.py
# https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py
# --------------------------------------------------------
import importlib

import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

import collections
import os
import tarfile
import urllib
import zipfile
from pathlib import Path

from utils.helper_types import Annotation

# from torch._six import string_classes
int_classes = int
string_classes = str
from torch.utils.data._utils.collate import (
    np_str_obj_array_pattern,
    default_collate_err_msg_format,
)
from tqdm import tqdm

import hashlib
import requests

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg_lpips": "vgg.pth"}

MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
    func: callable,
    data,
    n_proc,
    target_data_type="ndarray",
    cpu_intensive=True,
    use_worker_id=False,
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [[func, Q, part, i, use_worker_id] for i, part in enumerate(np.array_split(data, n_proc))]
    else:
        step = int(len(data) / n_proc + 1) if len(data) % n_proc != 0 else int(len(data) / n_proc)
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate([data[i : i + step] for i in range(0, len(data), step)])
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == "ndarray":
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == "list":
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


### tamimg.data.utils.py
def unpack(path):
    if path.endswith("tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=os.path.split(path)[0])
    elif path.endswith("tar"):
        with tarfile.open(path, "r:") as tar:
            tar.extractall(path=os.path.split(path)[0])
    elif path.endswith("zip"):
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(path=os.path.split(path)[0])
    else:
        raise NotImplementedError("Unknown file extension: {}".format(os.path.splitext(path)[1]))


def reporthook(bar):
    """tqdm progress bar for downloads."""

    def hook(b=1, bsize=1, tsize=None):
        if tsize is not None:
            bar.total = tsize
        bar.update(b * bsize - bar.n)

    return hook


def get_root(name):
    base = "data/"
    root = os.path.join(base, name)
    os.makedirs(root, exist_ok=True)
    return root


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()


def mark_prepared(root):
    Path(root).joinpath(".ready").touch()


def prompt_download(file_, source, target_dir, content_dir=None):
    targetpath = os.path.join(target_dir, file_)
    while not os.path.exists(targetpath):
        if content_dir is not None and os.path.exists(os.path.join(target_dir, content_dir)):
            break
        print("Please download '{}' from '{}' to '{}'.".format(file_, source, targetpath))
        if content_dir is not None:
            print("Or place its content into '{}'.".format(os.path.join(target_dir, content_dir)))
        input("Press Enter when done...")
    return targetpath


def download_url(file_, url, target_dir):
    targetpath = os.path.join(target_dir, file_)
    os.makedirs(target_dir, exist_ok=True)
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=file_) as bar:
        urllib.request.urlretrieve(url, targetpath, reporthook=reporthook(bar))
    return targetpath


def download_urls(urls, target_dir):
    paths = dict()
    for fname, url in urls.items():
        outpath = download_url(fname, url, target_dir)
        paths[fname] = outpath
    return paths


def quadratic_crop(x, bbox, alpha=1.0):
    """bbox is xmin, ymin, xmax, ymax"""
    im_h, im_w = x.shape[:2]
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.clip(bbox, 0, max(im_h, im_w))
    center = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    l = int(alpha * max(w, h))
    l = max(l, 2)

    required_padding = -1 * min(center[0] - l, center[1] - l, im_w - (center[0] + l), im_h - (center[1] + l))
    required_padding = int(np.ceil(required_padding))
    if required_padding > 0:
        padding = [
            [required_padding, required_padding],
            [required_padding, required_padding],
        ]
        padding += [[0, 0]] * (len(x.shape) - 2)
        x = np.pad(x, padding, "reflect")
        center = center[0] + required_padding, center[1] + required_padding
    xmin = int(center[0] - l / 2)
    ymin = int(center[1] - l / 2)
    return np.array(x[ymin : ymin + l, xmin : xmin + l, ...])


def custom_collate(batch):
    r"""source: pytorch 1.9.0, only one modification to original code"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):  # added
        return batch  # added
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


### taming.util.py
def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError("Trying to get past callable node with expand=False."),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


if __name__ == "__main__":
    config = {
        "keya": "a",
        "keyb": "b",
        "keyc": {
            "cc1": 1,
            "cc2": 2,
        },
    }
    from omegaconf import OmegaConf

    config = OmegaConf.create(config)
    print(config)
    retrieve(config, "keya")
