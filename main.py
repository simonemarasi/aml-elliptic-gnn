import warnings
import torch
import pandas as pd
import utils as u
import os
from loader import load_data, data_to_pyg
from train import train, test
from models import models
from argparse import ArgumentParser
from models.custom_gat.model import GAT

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = ArgumentParser()
parser.add_argument("-d", "--data", dest="data_path", help="Path of data folder")
command_line_args = parser.parse_args()
data_path = command_line_args.data_path

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("="*50)
print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path
features, edges = load_data(data_path)
data = data_to_pyg(features, edges)
print("Graph data loaded successfully")
print("="*50)
args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.device = 'cpu'
if args.use_cuda:
    args.device = 'cuda'
print ("Using CUDA: ", args.use_cuda, "- args.device: ", args.device)

args.num_features = data.num_features

models_to_train = {
    'GCN': models.GCNConvolution(args).to(args.device),
    'GAT': models.GATConvolution(args).to(args.device),
    'SAGE': models.SAGEConvolution(args).to(args.device),
    'Cheb': models.ChebyshevConvolution(args, kernel=[1,2]).to(args.device),
    'GATv2': models.GATv2Convolution(args).to(args.device),
    'Custom GAT': GAT(num_of_layers=3, num_heads_per_layer=[1, 4, 1],
                      num_features_per_layer=[args.num_features, args['hidden_units'],
                      args['hidden_units']//2, args['num_classes']], device=args.device).to(args.device)
}

compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG'])
print("Starting training models")
print("="*50)
for name, model in models_to_train.items():

    data = data.to(args.device)
    print('-'*50)
    print(f"Training model: {name}")
    print('-'*50)
    train(args, model, data)
    print('-'*50)
    print(f"Testing model: {name}")
    print('-'*50)
    test(model, data)
    print('-'*50)
    print(f"Computing metrics for model: {name}")
    print('-'*50)
    compare_illicit = compare_illicit.append(u.compute_metrics(model, name, data, compare_illicit), ignore_index=True)

compare_illicit.to_csv(os.path.join(data_path, 'metrics.csv'), index=False)
print('Results saved to metrics.csv')

u.plot_results(compare_illicit)

