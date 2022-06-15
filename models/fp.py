import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tgn
import torch_geometric.nn as gnn
import pytorch_lightning as pl
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch.nn import Linear, ModuleList


class FPNN_v4(nn.Module):
    def __init__(self, in_channels, out_channels=100, dense_size=512):
        super().__init__()
        self.mf = gnn.conv.MFConv(in_channels, out_channels, 
                                    max_degree=1, # computed from data
                                    #root_weight=False,
                                    bias=False
                                    )
        #self.bnorm = nn.BatchNorm1d(out_channels)
        self.dense = nn.Linear(out_channels, dense_size) # root weight
        #self.bnorm_dense = nn.BatchNorm1d(dense_size)
    
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        batch = torch.Tensor([0]*len(x)).to(torch.int64) if batch is None else batch
        
        # Conv
        x = self.mf(x, edge_index)
        x = F.relu(x)
        
        # Softmax
        x = self.dense(x)
        x = F.softmax(x, dim=-1)
        
        # Global pool
        x = gnn.global_add_pool(x, batch)
       
        return x

    def fingerprint(self, graph):
        with torch.inference_mode():
            out = self(graph).numpy()
        return out  

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, degree
# from torch.nn import Linear, ModuleList


class MFConv_custom(MessagePassing):

    def __init__(self, in_channels, out_channels, max_degree=10,
                 root_weight=False, bias=True, **kwargs):
        super(MFConv_custom, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree
        self.root_weight = root_weight

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=bias)
            for _ in range(max_degree + 1)
        ])

        if root_weight:
            self.root_lins = ModuleList([
                Linear(in_channels, out_channels, bias=False)
                for _ in range(max_degree + 1)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        if self.root_weight:
            for lin in self.root_lins:
                lin.reset_parameters()


    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)

        deg = degree(edge_index[1 if self.flow == 'source_to_target' else 0],
                     x.size(0), dtype=torch.long)
        deg.clamp_(max=self.max_degree)

        if not self.root_weight:
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        h = self.propagate(edge_index, x=x)

        out = x.new_empty(list(x.size())[:-1] + [self.out_channels])

        for i in deg.unique().tolist():
            idx = (deg == i).nonzero().view(-1)

            r = self.rel_lins[i](h.index_select(self.node_dim, idx))
            if self.root_weight:
                r = r + self.root_lins[i](x.index_select(self.node_dim, idx))

            out.index_copy_(self.node_dim, idx, r)

        return out


    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class FPNN_v3(nn.Module):
    def __init__(self, in_channels, out_channels=64, dense_size=64):
        super().__init__()
        self.mf = gnn.conv.MFConv(in_channels, out_channels, 
                                    max_degree=4, # computed from data
                                    #root_weight=False,
                                    bias=False
                                    )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.dense = nn.Linear(out_channels, dense_size)
        self.bnorm_dense = nn.BatchNorm1d(dense_size)
    
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        batch = torch.Tensor([0]*len(x)).to(torch.int64) if batch is None else batch
        
        # Conv
        x = self.mf(x, edge_index)
        x = F.relu(x)
        
        # Softmax
        x = F.softmax(x, dim=1)
        
        # Global pool
        x = gnn.global_add_pool(x, batch)
       
        return x

    def fingerprint(self, graph):
        with torch.inference_mode():
            out = self(graph).numpy()
        return out  



# class MFConv_v2(MessagePassing):
#     def __init__(self, in_channels, out_channels, max_degree=10,
#                  root_weight=False, bias=True, **kwargs):
#         super(MFConv_v2, self).__init__(aggr='add', **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.max_degree = max_degree
#         self.root_weight = root_weight

#         self.rel_lins = ModuleList([
#             Linear(in_channels, out_channels, bias=bias)
#             for _ in range(max_degree + 1)
#         ])

#         if root_weight:
#             self.root_lins = ModuleList([
#                 Linear(in_channels, out_channels, bias=False)
#                 for _ in range(max_degree + 1)
#             ])

#         self.reset_parameters()

#     def reset_parameters(self):
#         for lin in self.rel_lins:
#             lin.reset_parameters()
#         if self.root_weight:
#             for lin in self.root_lins:
#                 lin.reset_parameters()


#     def forward(self, x, edge_index):
#         edge_index, _ = remove_self_loops(edge_index)

#         deg = degree(edge_index[1 if self.flow == 'source_to_target' else 0],
#                      x.size(0), dtype=torch.long)
#         deg.clamp_(max=self.max_degree)

#         if not self.root_weight:
#             edge_index, _ = add_self_loops(edge_index,
#                                            num_nodes=x.size(self.node_dim))

#         h = self.propagate(edge_index, x=x)

#         out = x.new_empty(list(x.size())[:-1] + [self.out_channels])

#         for i in deg.unique().tolist():
#             idx = (deg == i).nonzero().view(-1)

#             r = self.rel_lins[i](h.index_select(self.node_dim, idx))
#             if self.root_weight:
#                 r = r + self.root_lins[i](x.index_select(self.node_dim, idx))

#             #### NEw entry ######
#             #print("R-shape : ", r.size())
#             #r = torch.tanh(r)

#             out.index_copy_(self.node_dim, idx, r)

#         return out


#     def message(self, x_j):
#         return x_j

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)





class FPNN_v2(nn.Module):
    def __init__(self, in_channels, out_channels=64, dense_size=64):
        super().__init__()
        self.mf = gnn.conv.MFConv(in_channels, out_channels, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.dense = nn.Linear(out_channels, dense_size)
        self.bnorm_dense = nn.BatchNorm1d(dense_size)
    
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        batch = torch.Tensor([0]*len(x)).to(torch.int64) if batch is None else batch
        # Conv
        x = self.mf(x, edge_index)
        x = F.relu(x)
        # Batch norm
        x = self.bnorm(x)
        # Max pooling over neigbors
        x_gr = gnn.pool.max_pool_neighbor_x(Data(x=x, edge_index=edge_index))
        # Dense
        x = self.dense(x_gr.x)
        x = F.relu(x)
        # Dense batch norm
        x = self.bnorm_dense(x)
        # Global pool
        x1 = gnn.global_add_pool(x, batch)
        #x1 = torch.tanh(x1)  ######## new
        #x2 = gnn.global_max_pool(x, batch)
        #x2 = torch.tanh(x2)
        #x = torch.cat((x1,x2), dim=1)
        x = torch.tanh(x1)
        return x

    def fingerprint(self, graph):
        with torch.inference_mode():
            out = self(graph).numpy()
        y1 = (out>=0) * 1
        y2 = (out<0) * 1
        return np.concatenate((y1,y2), axis=1)



class FPNN(pl.LightningModule):
    def __init__(self, input_size, mlp=[128], fp_length=256):
        super().__init__()
        layers = [nn.Linear(input_size, mlp[0], bias=False), nn.ReLU()]
        for i,layer in enumerate(mlp[1:]):
            layers.append( nn.Linear(mlp[i-1], layer) )
            layers.append( nn.ReLU() )
        self.mlp = nn.Sequential(*layers)
        self.gin1 = gnn.conv.GINConv(self.mlp, eps=0, train_eps=False)
        self.fc1 = nn.Linear(mlp[-1], fp_length, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gin1(x, edge_index)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        x = gnn.global_add_pool(x, batch)
        return x

