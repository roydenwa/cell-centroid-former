import cv2
import numpy as np
import tensorflow as tf

from scipy import ndimage as ndi


def basic_labeling(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8)
    return ndi.label(img)[0].astype(np.uint16)


def decode_centroid_representation(y: dict) -> list:
    # TODO: Batch-wise version
    centroids = basic_labeling(
        tf.where(y["centroid_heatmap"] > 0.75, 1, 0).numpy()[0, ..., 0]
    )
    labels = np.unique(centroids)
    img_height = centroids.shape[0]
    img_width = centroids.shape[1]
    bboxes = []

    for label in labels[1:]:
        label_mask = (centroids == label).astype(np.uint8).copy()
        moments = cv2.moments(label_mask)

        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            b_height = y["cell_dimensions"][0, centroid_y, centroid_x, 0] * img_height
            b_width = y["cell_dimensions"][0, centroid_y, centroid_x, 1] * img_width

            bboxes.append(
                [
                    int(centroid_x - b_width // 2),
                    int(centroid_y - b_height // 2),
                    int(b_width),
                    int(b_height),
                ]
            )

    return bboxes
