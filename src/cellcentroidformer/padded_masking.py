import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from typing import Tuple
from tensorflow.keras import layers


class PaddedMasking(layers.Layer):
    def __init__(
        self, patch_size: Tuple[int, int], img_size: Tuple[int, ...], **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask = tf.Variable(
            initial_value=tnp.ones(img_size, dtype=tf.float32), trainable=False
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mask = update_padded_mask(self.mask, self.patch_size, self.img_size)
        x = x * mask

        return x


@tf.function
def update_padded_mask(
    mask: tf.Variable,
    patch_size: Tuple[int, int] = (12, 12),
    img_size: Tuple[int, ...] = (384, 384),
) -> tf.Variable:
    pad_size = patch_size[0] // 2
    n_rows = img_size[0] // (patch_size[0] + pad_size)
    n_cols = img_size[1] // (patch_size[1] + pad_size)

    row_start, col_start = pad_size, pad_size

    for row in range(n_rows):
        for col in range(n_cols):
            row_end = row_start + patch_size[0]
            col_end = col_start + patch_size[1]
            patch_val = tnp.random.randint(low=0, high=2, dtype=tf.int32)
            patch_val = tf.cast(patch_val, tf.float32)
            mask[row_start:row_end, col_start:col_end].assign(patch_val)
            col_start += pad_size + patch_size[1]

        row_start += pad_size + patch_size[0]
        col_start = pad_size

    return mask


def get_padded_patch_mask(
    patch_size: Tuple[int, int] = (12, 12), img_size: Tuple[int, ...] = (384, 384)
) -> np.ndarray:
    mask = np.zeros(img_size)
    pad_size = patch_size[0] // 2
    n_rows = img_size[0] // (patch_size[0] + pad_size)
    n_cols = img_size[1] // (patch_size[1] + pad_size)

    row_start, col_start = pad_size, pad_size

    for row in range(n_rows):
        for col in range(n_cols):
            row_end = row_start + patch_size[0]
            col_end = col_start + patch_size[1]

            mask[row_start:row_end, col_start:col_end] += np.random.randint(
                low=0, high=2
            )
            col_start += pad_size + patch_size[1]

        row_start += pad_size + patch_size[0]
        col_start = pad_size

    return mask
