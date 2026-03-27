import cv2
import numpy as np


def _read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return img[..., None]
    if img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def read_image_any(path):
    return _read_image(path)


def read_image_jpegturbo(path):
    return _read_image(path)
