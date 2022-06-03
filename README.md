# CellCentroidFormer
Hybrid CNN-ViT model for cell detection in biomedical microscopy images.

## Model architecture
![Model architecture](assets/cell-centroid-former.png?raw=true "Model architecture")

## Conference Paper
> [**CellCentroidFormer: Combining Self-attention and Convolution for Cell Detection**](https://arxiv.org/abs/2206.00338),
> Wagner, Royden and Rohr, Karl,
> *MIUA 2022*; *arXiv ([arXiv:2206.00338](https://arxiv.org/abs/2206.00338))*

## Citation
```bibtex
@inproceedings{wagner2022cellcentroidformer,
  title={CellCentroidFormer: Combining Self-attention and Convolution for Cell Detection},
  author={Royden Wagner and Karl Rohr},
  booktitle={Medical Image Understanding and Analysis},
  year={2022}
}
```

## Acknowledgements
The subclass implementation of the MobileViT block ([Mehta and Rastegari, 2022](https://arxiv.org/abs/2110.02178)) in this repo is based on the [functional implementation](https://keras.io/examples/vision/mobilevit) by Sayak Paul.

The `@compact_get_layers` class decorator is inspired by the [get method](https://danijar.com/structuring-models) by Danijar Hafner and the nn.compact decorator in flax.
