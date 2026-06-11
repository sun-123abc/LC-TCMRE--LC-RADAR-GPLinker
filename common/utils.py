# common/utils.py
import json
import os
import random
import logging
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, indent=2):
    path = Path(path)
    if path.parent:
        ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 结果更稳定，但速度可能略慢
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg="auto"):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def move_to_device(obj, device):
    """
    递归地把 tensor 放到 device。
    meta 这种非 tensor 字段会原样保留。
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)

    return obj


def get_model_inputs(batch):
    """
    训练时传给 model 的内容。
    meta 不传入模型，只给 decode 和 evaluation 使用。
    """
    return {k: v for k, v in batch.items() if k != "meta"}


def get_logger(name="RE", log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def save_checkpoint(state, path):
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)