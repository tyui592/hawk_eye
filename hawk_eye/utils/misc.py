"""Miscellaneous utilities."""

import json
import pickle
import logging
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T

NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def save_pickle(path, obj):
    """Save an object as a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_json_file(path):
    """Read a json file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_logger(save_root: Path, name: str):
    """Get a logger that writes DEBUG to a file and INFO to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(Path(save_root) / 'log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        """Init."""
        self.reset()

    def reset(self):
        """Reset."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update with a value averaged over n samples."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_inputs(img, imsize):
    """Make model inputs from a RGB numpy image.

    The output has the same keys as `datasets.court.collate_fn` with batch size 1.
    """
    height, width = imsize
    resized = cv2.resize(img, (width, height))
    return {
        'image': NORMALIZE(resized).unsqueeze(0),
        'size': [(height, width)],
        'raw_size': [img.shape[:2]],
        'raw_img': [img],
    }


def get_single_image(args):
    """Load `args.image_path` as model inputs."""
    img = cv2.cvtColor(cv2.imread(str(args.image_path)), cv2.COLOR_BGR2RGB)
    return to_inputs(img, args.imsize)
