# SDS-2020
This repository provides all the code used to run the experiments for the SDS 2020 short paper "Effects of Graph Pooling Layers on Classificationwith Graph Neural Networks". The paper can be accessed [here](linktbd))

The experiments for the AIDS and Letter datasest were done using the **Gale framework** for reproducible research, which can be found [here](https://github.com/v7labs/Gale). Follow the installation process provided on their GitHub to install the python environment.
To run one of the graph neural network classifictaion task, simply use this 

`python template/RunMe.py --runner-class GraphClassificationIamdb --input-folder {path to the dataset} --model {model, e.g. gcnconv1TPK} --output-folder {path} --ignoregit --criterion CrossEntropyLoss --optimizer Adam --experiment-name sds-multirun --epochs 150 --batch-size 64 --lr 0.001 --step-size 30 --multi-run 10`

The dataset folder needs to contain a folder called `data` which contains all the gxl files with the graphs and the train.cxl, val.cxl and test.cxl files (which contain the filesnames and the class labels).


The experiments for on the pT1-GG dataset were performed using [this framework](https://github.com/waljan/GNNpT1), but they can also easily be run using the Gale framework.


The Graph Neural Netwok implementations all use the PyTorch Geometric library (more details can be found [here](https://github.com/rusty1s/pytorch_geometric))


The datasets can be downloaded as follows:
- pT1-Gland-Graph dataset: https://github.com/LindaSt/pT1-Gland-Graph-Dataset
- AIDS and Letter dataset: http://www.fki.inf.unibe.ch/databases/iam-graph-database


The file `models.py` contains all the models used in the experiments.


The following hyper-parameters were used for the experiments:
| Batch Size | # Epochs | Optimizer | Learning Rate | Learning Rate Decay Step Size | Criterion     | Multi-Run     |
|------------|----------|-----------|---------------|-------------------------------|---------------|---------------|
| 64         | 150      | Adam      | 0.001         | 30                            | Cross-Entropy | 10            |
