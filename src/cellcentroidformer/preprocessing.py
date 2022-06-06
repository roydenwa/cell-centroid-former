import numpy as np
import matplotlib.pyplot as plt


def min_max_scaling(img: np.ndarray) -> np.ndarray:
    if img.max() - img.min() != 0:
        img_scaled = (img - img.min()) / (img.max() - img.min())
        return img_scaled
    else:
        return img


def preprocess_img(img: np.ndarray) -> np.ndarray:
    img = min_max_scaling(img)
    img = plt.cm.gist_ncar(img)[..., 0:3]

    return img.astype(np.float32)