# SDS-2020
This repository provides all the code used to run the experiments for the SDS 2020 short paper "Effects of Graph Pooling Layers on Classificationwith Graph Neural Networks". The paper can be accessed [here](linktbd)

All scripts are using the Gale framework for reproducibly research, which can be found [here](https://github.com/v7labs/Gale). Follow the installation process provided on their GitHub to install the python environment.

The Graph Neural Netwok implementations all use the PyTorch Geometric library (more details can be founde [here](https://github.com/rusty1s/pytorch_geometric)

The datasets can be downloaded as follows:
- pT1-Gland-Graph dataset: https://github.com/LindaSt/pT1-Gland-Graph-Dataset
- AIDS and Letter dataset: http://www.fki.inf.unibe.ch/databases/iam-graph-database

The file `models.py` contains all the models used in the experiments.
