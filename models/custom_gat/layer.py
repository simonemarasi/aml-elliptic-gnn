import torch
from torch import nn
from torch.nn.functional import leaky_relu, dropout
from torch.nn.init import xavier_uniform_

class GATLayer(nn.Module):
    """
    Implementation of a Graph Attention Network (GAT) layer using PyTorch.
    This implementation is inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(), device='cpu'):
        """
        Initializes the GAT layer.

        Args:
            num_in_features (int): Number of input features per node.
            num_out_features (int): Number of output features per node.
            num_of_heads (int): Number of attention heads.
            concat (bool, optional): If True, output features from all attention heads will be concatenated.
                                     If False, the mean of output features will be computed.
                                     Default is True.
            activation (torch.nn.Module, optional): Activation function to apply to the output features.
                                                    Default is nn.ELU().
            device (str, optional): Device to use for computation. Default is 'cpu'.
        """
        super(GATLayer, self).__init__()
        self.device = device
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=0.6)
        
        xavier_uniform_(self.linear_proj.weight)
        xavier_uniform_(self.scoring_fn_target)
        xavier_uniform_(self.scoring_fn_source)
        
    def forward(self, data):
        """
        Performs the forward pass through the GAT layer.

        Args:
            data (tuple): Tuple containing node input features (shape: [N, FIN])
                          and edge index (shape: [2, E]).

        Returns:
            tuple: Tuple containing node output features (shape: [N, FOUT])
                   and edge index (shape: [2, E]).
        """
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[0]
        
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)
        
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leaky_relu(scores_source_lifted + scores_target_lifted)
        
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)
        
        out_nodes_features = self.skip_concat(in_nodes_features, out_nodes_features)
        
        return (out_nodes_features, edge_index)
    
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        Computes the softmax considering only the neighborhoods.

        Args:
            scores_per_edge (torch.Tensor): Tensor of shape [E, NH] containing the scores for each edge.
            trg_index (torch.Tensor): Tensor of shape [E] containing the target node indices for each edge.
            num_of_nodes (int): Total number of nodes in the graph.

        Returns:
            torch.Tensor: Tensor of shape [E, NH, 1] containing the attentions for each edge.
        """
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge).to(self.device)
        size = list(exp_scores_per_edge.shape)
        size[0] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=self.device)
        
        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)
        
        return exp_scores_per_edge / (neighborhood_sums.index_select(0, trg_index) + 1e-16).unsqueeze(-1)
    
    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        """
        Aggregates the weighted and projected neighborhood feature vectors for each target node.

        Args:
            nodes_features_proj_lifted_weighted (torch.Tensor): Tensor of shape [E, NH, FOUT] containing the weighted and projected
                                                               neighborhood feature vectors.
            edge_index (torch.Tensor): Tensor of shape [2, E] containing the edge index.
            in_nodes_features (torch.Tensor): Tensor of shape [N, FIN] containing the input node features.
            num_of_nodes (int): Total number of nodes in the graph.

        Returns:
            torch.Tensor: Tensor of shape [N, NH, FOUT] containing the aggregated neighborhood feature vectors.
        """
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[0] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=self.device)
        
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_features_proj_lifted_weighted).to(self.device)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        
        return out_nodes_features
    
    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts and duplicates certain vectors based on the edge index.

        Args:
            scores_source (torch.Tensor): Tensor of shape [E, NH] containing the source scores for each edge.
            scores_target (torch.Tensor): Tensor of shape [E, NH] containing the target scores for each edge.
            nodes_features_matrix_proj (torch.Tensor): Tensor of shape [E, NH, FOUT] containing the projected node features.
            edge_index (torch.Tensor): Tensor of shape [2, E] containing the edge index.

        Returns:
            tuple: Tuple containing the lifted source scores, lifted target scores, and lifted node features.
        """
        src_nodes_index = edge_index[0].to(self.device)
        trg_nodes_index = edge_index[1].to(self.device)
        
        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(0, src_nodes_index)
        
        return scores_source, scores_target, nodes_features_matrix_proj_lifted
    
    def explicit_broadcast(self, this, other):
        """
        Explicitly broadcasts the tensor `this` to match the shape of the tensor `other`.

        Args:
            this (torch.Tensor): Tensor to be broadcasted.
            other (torch.Tensor): Tensor whose shape is used for broadcasting.

        Returns:
            torch.Tensor: Broadcasted tensor.
        """
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        
        return this.expand(other.size())
    
    def skip_concat(self, in_nodes_features, out_nodes_features):
        """
        Performs skip-connection and concatenation or averaging of input and output features.

        Args:
            in_nodes_features (torch.Tensor): Tensor of shape [N, FIN] containing the input node features.
            out_nodes_features (torch.Tensor): Tensor of shape [N, NH, FOUT] containing the output node features.

        Returns:
            torch.Tensor: Tensor of shape [N, NH*FOUT] or [N, FOUT] containing the concatenated or averaged features.
        """
        if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
            out_nodes_features += in_nodes_features.unsqueeze(1)
        else:
            out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)
        
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
