# CellCentroidFormer
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GHqK61yOdhAele41pHf_ueklgCNqRcNf?usp=sharing)

Hybrid CNN-ViT model for cell detection in biomedical microscopy images.

## Model architecture
![Model architecture](assets/cell-centroid-former.png?raw=true "Model architecture")

The comparison of ViTs and CNNs in computer vision applications reveals that their receptive fields are fundamentally different ([Raghu et al., 2021](https://arxiv.org/abs/2108.08810)).
The receptive fields of ViTs capture local and global information in both earlier and later layers.
The receptive fields of CNNs, on the other hand, initially capture local information and gradually grow to capture global information in later layers.
Therefore, we use MobileViT blocks ([Mehta and Rastegari, 2022](https://arxiv.org/abs/2110.02178)) in the neck part of our proposed model to enhance global information compared to a fully convolutional neck part.
We represent cells by their centroid, their width, and their height.
Our model contains two fully convolutional heads to predict these cell properties.
The first head predicts a heatmap for cell centroids, and the second head predicts the cell dimensions (width and height) at the position of the corresponding cell centroid.

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

The normalized termperature-scaled cross-entropy loss for SimCLR ([Chen et al., 2020](https://arxiv.org/abs/2002.05709))is based on the [implementation](https://keras.io/examples/vision/semisupervised_simclr) by András Béres.