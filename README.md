# SCALE: Unsupervised Multi-Scale Domain Identification in Spatial Omics Data

SCALE is a Python package for identifying multi-scale spatial domains in spatial omics data. It leverages graph neural representation learning and an entropy-based search algorithm to detect stable spatial domains at different scales, enabling comprehensive analysis of tissue organization. The preprint can be found [here](https://www.biorxiv.org/content/10.1101/2025.05.21.653987v1).

## Installation
Clone the repository:
```bash
git clone https://github.com/imsb-uke/scale.git
cd scale
```

We recommend to use poetry to install the package.
```bash
poetry install
```

Otherwise you can install the package via pip:
```bash
pip install -e .
```

## Quick Start

You can find a short vignette here `notebooks/vignette.ipynb`, applying SCALE to MERFISH mouse brain data. For convenience the data is provided in the `data` folder.

## Citation

If you use SCALE in your research, please cite:
```
@article{yousefi2025scale,
  title={SCALE: Unsupervised Multi-Scale Domain Identification in Spatial Omics Data},
  author={Yousefi, Behnam and Schaub, Darius P and Khatri, Robin and Kaiser, Nico and Kuehl, Malte and Ly, Cedric and Puelles, Victor G and Huber, Tobias B and Prinz, Immo and Krebs, Christian F and others},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```