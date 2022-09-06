import tensorflow as tf


class MeanIoUThresh(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_tresh = tf.where(y_true > 0.75, 1, 0)
        y_pred_tresh = tf.where(y_pred > 0.75, 1, 0)

        return super().update_state(y_true_tresh, y_pred_tresh, sample_weight)


def ssim_metric(y_true, y_pred):
    max_true = tf.reduce_max(y_true)
    max_pred = tf.reduce_max(y_pred)
    max_val = max_true if max_true > max_pred else max_pred
    ssim_score = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))

    return ssim_score


def circle2ellipse_ssim(radius_map, height_map, width_map, img_width, img_height):
    """For comparing the outputs of CellCentroidFormer and CircleNet."""
    height_map = height_map * img_height
    width_map = width_map * img_width
    max_height = height_map.max() if height_map.max() > radius_map.max() else radius_map.max()
    max_width = width_map.max() if width_map.max() > radius_map.max() else radius_map.max()

    height_ssim = tf.image.ssim(height_map, radius_map, max_val=max_height)
    width_ssim = tf.image.ssim(width_map, radius_map, max_val=max_width)

    return (height_ssim + width_ssim) / 2