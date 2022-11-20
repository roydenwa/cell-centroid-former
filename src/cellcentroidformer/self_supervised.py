import tensorflow as tf

from tensorflow.keras import layers, models
from cellcentroidformer.padded_masking import PaddedMasking
from cellcentroidformer.model_architecture import CellCentroidFormer


class PseudocolorizeMaskedCells(CellCentroidFormer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masking = PaddedMasking(patch_size=(12, 12))
        self.augs = models.Sequential(
            name="augs",
            layers=[
                layers.RandomFlip(),
                layers.RandomCrop(height=320, width=320),
                layers.RandomRotation(factor=(-0.2, 0.2)),
                layers.Resizing(height=384, width=384),
                layers.Rescaling(scale=1.0 / 255),
            ],
        )

    def train_step(self, data):
        x, y = data
        concat_xy = tf.concat([x, y], axis=0)
        concat_xy = self.augs(concat_xy)

        batch_size = tf.shape(x)[0]  # Last batch might be smaller
        x = concat_xy[0:batch_size, ...]
        x = self.masking(x)

        # To match the output format of CellCentroidFormer
        y = {
            "cell_dimensions": concat_xy[batch_size:, ...],
            "centroid_heatmap": concat_xy[batch_size:, ...],
        }

        return super().train_step((x, y))


class SimCLR(models.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = models.Model(
            inputs=base_model.input,
            outputs=base_model.output,
            name="encoder",
        )
        self.projection_head = models.Sequential(
            layers=[
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
            ],
            name="projection_head",
        )
        self.loss_temperature = 0.1
        self.augs = models.Sequential(
            layers=[
                layers.RandomCrop(width=202, height=202),
                layers.RandomRotation(factor=(-0.1, 0.1)),
                layers.Resizing(height=224, width=224),
                layers.RandomContrast(factor=0.3),
                layers.RandomBrightness(factor=0.3),
                layers.Rescaling(scale=1.0 / 255),
            ],
            name="augs",
        )

    def nt_xent_loss(self, projections_1, projections_2, temperature=0.1):
        """Normalized temperature-scaled crossentropy loss"""
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        loss = (loss_1_2 + loss_2_1) / 2

        return loss

    def train_step(self, data):
        x = data

        with tf.GradientTape() as tape:
            x_1 = self.augs(x)
            x_2 = self.augs(x)

            x_1 = self.encoder(x_1, training=True)
            projections_1 = self.projection_head(x_1, training=True)
            x_2 = self.encoder(x_2, training=True)
            projections_2 = self.projection_head(x_2, training=True)

            loss = self.nt_xent_loss(projections_1, projections_2)

        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        return {"loss": loss}

    def call(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return x
