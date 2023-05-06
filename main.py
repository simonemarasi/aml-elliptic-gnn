import torch
import pandas as pd
import utils as u
from loader import load_data, data_to_pyg
from train import train, test
from models import models
from models.custom_gat.model import GAT

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = u.create_parser()
args = u.parse_args(parser)

features, edges = load_data(args.data_path)
data = data_to_pyg(features, edges)

args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.args.device='cpu'
if args.use_cuda:
    args.args.device='cuda'
print ("use CUDA:", args.use_cuda, "- args.device:", args.args.device)

models_to_train = {
    'GCN': models.GCNConvolution().to(args.args.device),
    'GAT': models.GATConvolution().to(args.device),
    'SAGE': models.SAGEConvolution().to(args.device),
    'Cheb': models.ChebyshevConvolution(kernel=[1,2]).to(args.device),
    'GATv2': models.GATv2Convolution().to(args.device),
    'Custom GAT': GAT(num_of_layers=3, num_heads_per_layer=[1, 4, 1], 
                      num_features_per_layer=[data.num_features, args['hidden_units'], 
                      args['hidden_units']//2, args['num_classes']], device=args.device).to(args.device)
}

compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG'])

for name, model in models_to_train.items():

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
    u.compute_metrics(model, name, data, compare_illicit)

    u.plot_results(compare_illicit)