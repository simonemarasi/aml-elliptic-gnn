from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, Linear, Module
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
import torch.nn.functional as F

class GCNConvolution(Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = GCNConv(args['num_features'], args['hidden_units'])
        self.conv2 = GCNConv(args['hidden_units'], args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

class SAGEConvolution(Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = SAGEConv(args['num_features'], args['hidden_units'])
        self.conv2 = SAGEConv(args['hidden_units'], args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

class GATConvolution(Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = GATConv(args['num_features'], args['hidden_units'])
        self.conv2 = GATConv(args['hidden_units'], args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

class ChebyshevConvolution(Module):
    def __init__(self, args, kernel=[1,1]):
        super().__init__()
        self.conv1 = ChebConv(args['num_features'], args['hidden_units'], kernel[0])
        self.conv2 = ChebConv(args['hidden_units'], args['num_classes'], kernel[1])

    def forward(self, data):
        x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

class GATv2Convolution(Module):
    def __init__(self, args):
        super(GATv2Convolution, self).__init__()
        self.conv1 = GATv2Conv(args['num_features'], args['hidden_units'])
        self.lin1 = Linear(args['num_features'], args['hidden_units'])
        self.conv2 = GATv2Conv(args['hidden_units'], args['num_classes'])
        self.lin2 = Linear(args['hidden_units'], args['num_classes'])

    def forward(self, data):
        x, edge_index = data
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = self.conv2(x, edge_index) + self.lin2(x)
        return F.log_softmax(x, dim=1), edge_index