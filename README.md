  <img align="middle" src="./docs/logo.png" alt="OpenChem" width="500px">

# OpenChem

OpenChem is a deep learning toolkit for Computational Chemistry with [PyTorch](https://pytorch.org) backend. Main goal of OpenChem is to make experimentation with deep learning models in Computational Chemistry easy and friendly for non-computer scientists.

## Main features

* Modular design with unified API, modules can be easily combined with each other.
* OpenChem is easy-to-use: new models are built with only configuration file.
* Fast training with multi-gpu support.
* Utilities for data preprocessing.
* Tensorboard support.

## Supported models

Currently OpenChem offers predictive models for variuos modalities:

* Smiles2Label
* Graph2Label
* MoleculeProtein2Label

## Installation

### General installation
In order to get started you need:
* Modern NVIDIA GPU, [compute capability 3.5](https://developer.nvidia.com/cuda-gpus) or newer.
* Python 3.6 (we recommend [Anaconda](https://www.continuum.io/downloads) distribution)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [Pytorch 0.4.1](https://pytorch.org) (compatible with your CUDA version)
* [Tensorflow 1.8.0](https://www.tensorflow.org/install/) with GPU support (compatible with your CUDA version)
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](http://scikit-learn.org/)


2. Clone the repository to your local folder and install the requirements with:

```bash
git clone https://github.com/Mariewelt/OpenChem.git
cd Openchem
conda install --yes --file requirements.txt
conda install -c rdkit rdkit nox cairo
```
3. Install PyTorch 0.4.1 from 

### Installation with Docker

## Acknowledgements

OpenChem is sponsored by [the University of North Carolina at Chapel Hill](https://www.unc.edu/) and [NVIDIA Corp.](https://www.nvidia.com/en-us/) 
<div align="middle">
  <img src="./docs/UNC_logo_RGB.png" alt="UNC" width="400px">
  <img src="./docs/NVLogo_2D_H.png" alt="NVIDIA" width="400px">
  <br>
</div>
