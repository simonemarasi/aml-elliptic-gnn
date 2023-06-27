import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric import seed_everything

def load_data(data_path):

    # Read edges, features and classes from csv files
    df_edges = pd.read_csv(osp.join(data_path, "elliptic_txs_edgelist.csv"))
    df_features = pd.read_csv(osp.join(data_path, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(osp.join(data_path, "elliptic_txs_classes.csv"))

    # Name colums basing on index
    colNames1 = {'0': 'txId', 1: "Time step"}
    colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(93)}
    colNames3 = {str(ii+95): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

    colNames = dict(colNames1, **colNames2, **colNames3 )
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}

    # Rename feature columns
    df_features = df_features.rename(columns=colNames)

    # Map unknown class to '3'
    df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    # Merge classes and features in one Dataframe
    df_class_feature = pd.merge(df_classes, df_features)

    # Exclude records with unknown class transaction
    df_class_feature = df_class_feature[df_class_feature["class"] != '3']

    # Build Dataframe with head and tail of transactions (edges)
    known_txs = df_class_feature["txId"].values
    df_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]

    # Build indices for features and edge types
    features_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["txId"].unique()))}
    class_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["class"].unique()))}

    # Apply index encoding to features
    df_class_feature["txId"] = df_class_feature["txId"].apply(lambda name: features_idx[name])
    df_class_feature["class"] = df_class_feature["class"].apply(lambda name: class_idx[name])

    # Apply index encoding to edges
    df_edges["txId1"] = df_edges["txId1"].apply(lambda name: features_idx[name])
    df_edges["txId2"] = df_edges["txId2"].apply(lambda name: features_idx[name])
    
    return df_class_feature, df_edges


def data_to_pyg(df_class_feature, df_edges):

    seed_everything(42)
    # Define PyTorch Geometric data structure with Pandas dataframe values
    edge_index = torch.tensor([df_edges["txId1"].values,
                            df_edges["txId2"].values], dtype=torch.long)
    x = torch.tensor(df_class_feature.iloc[:, 3:].values, dtype=torch.float)
    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data = RandomNodeSplit(num_val=0.15, num_test=0.2)(data)

    return data