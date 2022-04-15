import tensorflow as tf

from tensorflow.keras import models, layers
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from typing import Tuple, Any, Callable


def compact_get_layers(module: tf.Module) -> tf.Module:
    """
    Add a _layers dict and a get method to make tf modules more compact
    by initializing layers on the fly.
    """
    base_init = module.__init__

    def __init__(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self._layers = {}

    def get(self, name: str, constructor: Callable[..., Any], *args, **kwargs) -> layers.Layer:
        if name not in self._layers:
            self._layers[name] = constructor(*args, **kwargs, name=name)
        return self._layers[name]

    setattr(module, "__init__", __init__)
    setattr(module, "get", get)

    return module


@compact_get_layers
class MultiLayerPerceptron(layers.Layer):
    def __init__(self, hidden_units: list, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for idx, units in enumerate(self.hidden_units):
            x = self.get(f"dense{idx + 1}", layers.Dense, units, activation=tf.nn.swish)(x)
            x = self.get(f"drop{idx + 1}", layers.Dropout, 0.1)(x)
        return x


@compact_get_layers
class TransformerEncoder(layers.Layer):
    def __init__(self, num_blocks: int, projection_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.projection_dim = projection_dim
        self.num_heads = num_heads

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for idx in range(self.num_blocks):
            x1 = self.get(f"layer_norm{idx + 1}.1", layers.LayerNormalization, epsilon=1e-6)(x)
            attn_maps = self.get(f"mhs_attn{idx + 1}", layers.MultiHeadAttention, num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1, x1)
            x2 = self.get(f"add{idx + 1}.1", layers.Add)([attn_maps, x])
            x3 = self.get(f"layer_norm{idx + 1}.2", layers.LayerNormalization, epsilon=1e-6)(x2)
            x3 = self.get(f"mlp{idx + 1}", MultiLayerPerceptron, hidden_units=[x.shape[-1] * 2, x.shape[-1]])(x3)
            x = self.get(f"add{idx + 1}.2", layers.Add)([x3, x2])

        return x


@compact_get_layers
class MobileViTBlock(layers.Layer):
    def __init__(self, num_transformer_blocks: int, projection_dim: int, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.num_transformer_blocks = num_transformer_blocks
        self.projection_dim = projection_dim
        self.patch_size = patch_size

    def call(self, x: tf.Tensor) -> tf.Tensor:
        local_features = self.get("conv1", layers.Conv2D, filters=self.projection_dim, kernel_size=3, activation=tf.nn.swish, padding="same")(x)
        local_features = self.get("conv2", layers.Conv2D, filters=self.projection_dim, kernel_size=1, activation=tf.nn.swish, padding="same")(local_features)

        # Unfold local features into a sequence of patches for the transformer encoder:
        num_patches = int((local_features.shape[1] * local_features.shape[2]) / self.patch_size)
        patches = self.get("unfold", layers.Reshape, target_shape=(self.patch_size, num_patches, self.projection_dim))(local_features)
        global_features = self.get("transformer_encoder", TransformerEncoder, num_blocks=self.num_transformer_blocks, projection_dim=self.projection_dim, num_heads=2)(patches)

        # Fold global features again into a 3D representation to concat with the input tensor:
        global_features = self.get("fold", layers.Reshape, target_shape=(*local_features.shape[1:-1], self.projection_dim))(global_features)
        global_features = self.get("conv3", layers.Conv2D, filters=x.shape[-1], kernel_size=1, activation=tf.nn.swish, padding="same")(global_features)
        combined_features = self.get("concat", layers.Concatenate, axis=-1)([x, global_features])
        combined_features = self.get("conv4", layers.Conv2D, filters=self.projection_dim, kernel_size=3, activation=tf.nn.swish, padding="same")(combined_features)

        return combined_features


@compact_get_layers
class CellCentroidFormer(models.Model):
    def __init__(self, input_shape: Tuple[int, int, int], projection_dims_neck: Tuple[int, int], conv_filters_heads: Tuple[int, int, int]):
        super().__init__()
        input_layer = layers.Input(shape=input_shape)
        backbone = EfficientNetV2S(input_tensor=input_layer, include_top=False)
        self.backbone = models.Model(
            name="backbone",
            inputs=backbone.input,
            outputs=backbone.get_layer("block6a_expand_activation").output
        )
        self.projection_dims_neck = projection_dims_neck
        self.conv_filters_heads = conv_filters_heads

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.backbone(x)

        # Neck:
        x = self.get("mobilevit_block1", MobileViTBlock, num_transformer_blocks=2, projection_dim=self.projection_dims_neck[0], patch_size=4)(x)
        x = self.get("conv1", layers.Conv2D, filters=self.projection_dims_neck[0], kernel_size=3, activation="relu", padding="same")(x)
        x = self.get("norm1", layers.LayerNormalization)(x)
        x = self.get("conv2", layers.Conv2D, filters=self.projection_dims_neck[0], kernel_size=3, activation="relu", padding="same")(x)
        x = self.get("norm2", layers.LayerNormalization)(x)
        x = self.get("mobilevit_block2", MobileViTBlock, num_transformer_blocks=2, projection_dim=self.projection_dims_neck[1], patch_size=4)(x)
        x = self.get("conv3", layers.Conv2D, filters=self.projection_dims_neck[1] * 2, kernel_size=3, activation="relu", padding="same")(x)
        x = self.get("norm3", layers.LayerNormalization)(x)
        x = self.get("conv4", layers.Conv2D, filters=self.projection_dims_neck[1] * 2, kernel_size=3, activation="relu", padding="same")(x)
        output_neck = self.get("norm4", layers.LayerNormalization)(x)

        # Centroid heatmap head:
        x = output_neck
        for idx, num_filters in enumerate(self.conv_filters_heads):
            x = self.get(f"hm_up{idx}", layers.UpSampling2D, interpolation="bilinear")(x)
            x = self.get(f"hm_conv{idx}.1", layers.Conv2D, filters=num_filters, kernel_size=3, activation="relu", padding="same")(x)
            x = self.get(f"hm_norm{idx}.1", layers.BatchNormalization)(x)
            x = self.get(f"hm_conv{idx}.2", layers.Conv2D, filters=num_filters, kernel_size=3, activation="relu", padding="same")(x)
            x = self.get(f"hm_norm{idx}.2", layers.BatchNormalization)(x)

        x = self.get("hm_conv3", layers.Conv2D, filters=32, kernel_size=3, activation="relu", padding="same")(x)
        centroid_heatmap = self.get("centroid_heatmap", layers.Conv2D, filters=1, kernel_size=3, activation="relu", padding="same")(x)

        # Cell dimensions head:
        x = output_neck
        for idx, num_filters in enumerate(self.conv_filters_heads):
            x = self.get(f"dim_up{idx}", layers.UpSampling2D, interpolation="bilinear")(x)
            x = self.get(f"dim_conv{idx}.1", layers.Conv2D, filters=num_filters, kernel_size=3, activation="relu", padding="same")(x)
            x = self.get(f"dim_norm{idx}.1", layers.BatchNormalization)(x)
            x = self.get(f"dim_conv{idx}.2", layers.Conv2D, filters=num_filters, kernel_size=3, activation="relu", padding="same")(x)
            x = self.get(f"dim_norm{idx}.2", layers.BatchNormalization)(x)

        x = self.get("dim_conv3", layers.Conv2D, filters=32, kernel_size=3, activation="relu", padding="same")(x)
        height_map = self.get("height_map", layers.Conv2D, filters=1, kernel_size=3, activation="sigmoid", padding="same")(x)
        width_map = self.get("width_map", layers.Conv2D, filters=1, kernel_size=3, activation="sigmoid", padding="same")(x)

        return centroid_heatmap, height_map, width_map