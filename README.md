## Anti-money Laundering on Elliptic Dataset with GNN

The goal of this work is to tackle anti-money laundering problem trying to classifying efficiently illicit transactions and create a comparison between different architectures, in particular comparing different types of Graph Neural Networks (GNNs), to identify what are the key features and approaches that enable good performances in this given context.

### Setup

First of all you have to clone the repository with the standard command:

`git clone https://github.com/simonemarasi/aml-elliptic-gnn`

If you want only run the code note that it is also available a ready-to-run Google Colab version of the project at the following [link](https://colab.research.google.com/drive/145zhW2mehWVOJi3-wlEF4Y0evJOy_uDb?usp=sharing). The full code is inspectable cloning this repository.

#### Download the data

You can download the dataset zipped from the following [link](https://www.4sync.com/web/directDownload/fQErng3L/5YfHxh7W.cc4f36f14c07d75ced4bf1fcfa1a0772). After done that make sure to extract the zip file into the `data` folder located at the root of the repository. However you can put the data also in other places, making sure to change the folder in the configuration file (`config.yaml`) accordingly.

In the configuration file it is possible also to modify some hyperparameters such as the number of epochs, the number of hidden units to use, the learning rate, etc.

### Run

It is possible to install all the packages required for the execution launching the command
`pip install -r requirements.txt`
After that, to run the script execute the command
`python main.py`

### Results

| Model      | Precision          | Recall          |  F1            | F1 Micro AVG          |
|------------|--------------------|-----------------|----------------|-----------------------|
| GCN        | 0.832              | 0.457           | 0.59           | 0.94                  |
| GAT        | 0.787              | 0.683           | 0.731          | 0.952                 |
| SAGE       | 0.931              | 0.788           | 0.853          | 0.974                 |
| Cheb       | **0.942**          | 0.795           | **0.862**      | **0.976**             |
| GATv2      | 0.891              | **0.804**       | 0.845          | 0.972                 |
| Custom GAT | 0.861              | 0.762           | 0.808          | 0.966                 | 

You can find also the complete report of the project in this repository.
