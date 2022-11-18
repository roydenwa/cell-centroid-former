import numpy as np

from typing import Tuple


def get_padded_patch_mask(
    patch_size: Tuple[int, int] = (12, 12), img_size: Tuple[int, int] = (384, 384)
) -> np.ndarray:
    pad_size = patch_size[0] // 4

    if len(img_size) == 2:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    elif len(img_size) == 3:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))

    n_rows = img_size[0] // (patch_size[0] + 2 * pad_size)
    n_cols = img_size[1] // (patch_size[1] + 2 * pad_size)

    for row in range(n_rows):
        for col in range(n_cols):
            if len(img_size) == 3:
                patch = np.zeros((*patch_size, 3))
            else:
                patch = np.zeros(patch_size)
            patch += np.random.randint(low=0, high=2)

            patch = np.pad(
                patch, mode="constant", constant_values=1, pad_width=pad_width
            )

            if col == 0:
                patch_row = patch
            else:
                patch_row = np.hstack((patch_row, patch))
        if row == 0:
            patch_mask = patch_row
        else:
            patch_mask = np.vstack((patch_mask, patch_row))

    last_pad = (img_size[0] - patch_mask.shape[0], img_size[1] - patch_mask.shape[1])

    last_pad_x = last_pad[0] // 2
    last_pad_x_r = last_pad[0] % 2
    last_pad_y = last_pad[0] // 2
    last_pad_y_r = last_pad[0] % 2

    patch_mask = np.pad(
        patch_mask,
        mode="constant",
        constant_values=1,
        pad_width=(
            (last_pad_x, last_pad_x + last_pad_x_r),
            (last_pad_y, last_pad_y + last_pad_y_r),
        ),
    )

    return patch_mask
