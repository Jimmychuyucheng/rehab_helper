# import torch
# from torch import Tensor
# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid



# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#         self.dropout = torch.nn.Dropout(0.5)
#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         # x: Node feature matrix of shape [num_nodes, in_channels]
#         # edge_index: Graph connectivity matrix of shape [2, num_edges]
#         x = self.conv1(x, edge_index).relu()
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         x = self.dropout(x)
#         x = torch.mean(x, dim=1)
#         return x

