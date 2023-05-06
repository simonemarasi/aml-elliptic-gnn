
import torch
from torch.nn import Module, ELU, Linear, Parameter, LeakyReLU, Dropout, init

class GATLayer(Module):
    """
    This implementation is inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    """
    
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension (axis)
    head_dim = 1       # attention head dimension

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=ELU(), device='cpu'):

        super().__init__()

        self.device = device
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.linear_proj = Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.skip_proj = Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.leakyReLU = LeakyReLU(0.2)
        self.activation = activation
        self.dropout = Dropout(p=0.6)

        # Initialize parameters
        init.xavier_uniform_(self.linear_proj.weight)
        init.xavier_uniform_(self.scoring_fn_target)
        init.xavier_uniform_(self.scoring_fn_source)
        
    def forward(self, data):

        ### Step 1: Linear Projection + regularization
        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        
        # shape = (N, FIN) where
        # - N: number of nodes in the graph
        # - FIN: number of input features per node
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) 
        # - NH: number of heads
        # - FOUT: num of output features
        # Project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        ### Step 2: Edge attention calculation

        # Apply the scoring function (element-wise product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        ### Step 3: Neighborhood aggregation

        # Element-wise product.
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # Sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        ### Step 4: Residual/skip connections, concat

        out_nodes_features = self.skip_concat(attentions_per_edge, in_nodes_features, out_nodes_features)

        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        It does softmax only over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3)
        The score doesn't care about other edge scores that include nodes that are not in the neighborhood.

        """
        # Numerator
        scores_per_edge = scores_per_edge - scores_per_edge.max() # Make logits <= 0 (this improves the numerical stability)
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Denominator: shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is added to avoid divsion by 0
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so to be able to compute element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge).to(self.device)

        # shape = (N, NH)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=self.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):

        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=self.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted).to(self.device)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E
        """
        src_nodes_index = edge_index[self.src_nodes_dim].to(self.device)
        trg_nodes_index = edge_index[self.trg_nodes_dim].to(self.device)

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand(other.size())

    def skip_concat(self, attention_coefficients, in_nodes_features, out_nodes_features):

        if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  
            # if FIN == FOUT
            # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
            # thus we're basically copying input vectors NH times and adding to processed vectors
            out_nodes_features += in_nodes_features.unsqueeze(1)
        else:
            # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
            # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
            out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)