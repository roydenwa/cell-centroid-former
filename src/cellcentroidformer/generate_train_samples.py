import cv2
import numpy as np


def mask2bboxes(mask: np.ndarray) -> list:
    labels = np.unique(mask)
    bboxes = []

    # Skip label 0 (background):
    for label in labels[1:]:
        label_mask = (mask == label)
        bbox = cv2.boundingRect(label_mask.astype(np.uint8))
        bboxes.append(bbox)

    return bboxes


def bboxes2heatmap(bboxes: list, img: np.ndarray, blur=True) -> np.ndarray:
    heatmap = np.zeros(img.shape)
    img_h, img_w = img.shape[0], img.shape[1]

    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        tmp_heatmap = np.zeros(img.shape)

        cv2.ellipse(
            img=tmp_heatmap,
            center=(x + w//2, y + h//2),
            axes=(int(w//2.5), int(h//2.5)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1
        )

        if blur and int(w // 1.5) and int(h // 1.5):
            blur_y1 = y - h if y - h > 0 else 0
            blur_y2 = y + 2 * h if y + 2 * h < img_h else img_h
            blur_x1 = x - w if x - w > 0 else 0
            blur_x2 = x + 2 * w if x + 2 * w < img_w else img_w

            tmp_heatmap[blur_y1:blur_y2, blur_x1:blur_x2] = \
                cv2.blur(
                    src=tmp_heatmap[blur_y1:blur_y2, blur_x1:blur_x2],
                    ksize=(int(w // 1.5), int(h // 1.5))
                )
        heatmap += tmp_heatmap

    return heatmap


def bboxes2center_dim_blocks(bboxes: list, img: np.ndarray) -> np.ndarray:
    img_h, img_w = img.shape[0], img.shape[1]
    dim_y_blocks = np.zeros(img.shape)
    dim_x_blocks = np.zeros(img.shape)

    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(
            img=dim_y_blocks,
            pt1=(x + int(0.5 * w//2), y + int(0.5 * h//2)),
            pt2=(x + int(1.5 * w//2), y + int(1.5 * h//2)),
            color=(h / img_h),
            thickness=-1
        )
        cv2.rectangle(
            img=dim_x_blocks,
            pt1=(x + int(0.5 * w//2), y + int(0.5 * h//2)),
            pt2=(x + int(1.5 * w//2), y + int(1.5 * h//2)),
            color=(w / img_w),
            thickness=-1
        )
    center_dim_blocks = np.dstack((dim_y_blocks, dim_x_blocks))

    return center_dim_blocks