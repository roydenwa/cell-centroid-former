import numpy as np

from typing import Tuple


def get_padded_patch_mask(
    patch_size: Tuple[int, int] = (12, 12), img_size: Tuple[int, ...] = (384, 384)
) -> np.ndarray:
    pad_size = patch_size[0] // 2
    mask = np.zeros(img_size)

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
