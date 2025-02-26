import itertools
import numpy as np
from torch import nn
import torch
from torch.nn import Sigmoid
import torch.nn.functional as F
from models.egnn_clean import egnn_clean as eg
from models.egnn_clean import net_utils
from torch_geometric.utils import to_dense_batch
import torch
from sklearn.neighbors import NearestNeighbors

def compute_edge_index(all_pos, all_batch, k=5, threshold=1.0):
    # 获取所有节点的数量
    num_nodes = all_pos.shape[0]
    
    # 初始化一个空列表来存储边
    edges = []

    # 遍历每个批次，计算批次内的边
    batch_size = all_batch.max().item() + 1  # 假设all_batch是0-indexed
    for batch_id in range(batch_size):
        # 获取当前批次中属于该批次的所有节点
        batch_nodes = torch.nonzero(all_batch == batch_id).squeeze()  # 当前批次的节点索引
        batch_pos = all_pos[batch_nodes]  # 当前批次的节点坐标
        
        # 确保将batch_pos移动到CPU并转换为numpy格式
        batch_pos_cpu = batch_pos.cpu().numpy()

        # 使用k近邻算法或距离阈值计算当前批次的边
        nbrs = NearestNeighbors(n_neighbors=k, radius=threshold)
        nbrs.fit(batch_pos_cpu)  # 使用CPU上的数据
        _, indices = nbrs.kneighbors(batch_pos_cpu)  # 返回每个节点的k近邻节点的索引

        # 生成边
        for i in range(len(batch_nodes)):
            node_idx = batch_nodes[i].item()  # 当前节点的全局索引
            for neighbor_idx in indices[i]:
                if neighbor_idx != i:  # 排除自环
                    neighbor_global_idx = batch_nodes[neighbor_idx].item()
                    edges.append([node_idx, neighbor_global_idx])

    # 将所有的边转换为 PyTorch tensor，形成 edge_index
    edge_index = torch.tensor(edges).t().contiguous()

    return edge_index

class EGNNModel(torch.nn.Module):
    def __init__(self, input_features_size, hidden_channels, out_dim, edge_features, num_egnn_layers, num_layers, mode):
        super(EGNNModel, self).__init__()
        self.mode = mode
        self.num_layers = num_layers

        self.egnn_1 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=out_dim,
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_2 = eg.EGNN(in_node_nf=out_dim,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(out_dim / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)
        
        self.egnn_4 = eg.EGNN(in_node_nf=int(out_dim / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(out_dim / 4),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.fc1 = net_utils.FC((out_dim + int(out_dim / 2) + int(out_dim / 4)) * num_layers,
                                out_dim + 50, relu=False, bnorm=True)
        self.final = net_utils.FC(out_dim + 50, out_dim, relu=False, bnorm=False)

        self.bnrelu1 = net_utils.BNormRelu(out_dim)
        self.bnrelu2 = net_utils.BNormRelu(int(out_dim / 2))
        self.bnrelu3 = net_utils.BNormRelu(int(out_dim / 4))
        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, edge_index, x_batch, x_pos = data.x, data.edge_index, data.batch, data.coords

        output_res, pre_pos_res = self.egnn_1(h=x_res.float(),
                                              x=x_pos.float(),
                                              edges=edge_index,
                                              edge_attr=None)

        output_res_2, pre_pos_res_2 = self.egnn_2(h=output_res,
                                                  x=pre_pos_res.float(),
                                                  edges=edge_index,
                                                  edge_attr=None)

        output_res_4, pre_pos_seq_4 = self.egnn_4(h=output_res_2,
                                                  x=pre_pos_res_2.float(),
                                                  edges=edge_index,
                                                  edge_attr=None)

        if('simple' in self.mode):
            output_res = net_utils.get_pool(pool_type='mean')(output_res, x_batch)
            output_res = self.bnrelu1(output_res)

            output_res_2 = net_utils.get_pool(pool_type='mean')(output_res_2, x_batch)
            output_res_2 = self.bnrelu2(output_res_2)

            output_res_4 = net_utils.get_pool(pool_type='mean')(output_res_4, x_batch)
            output_res_4 = self.bnrelu3(output_res_4)
        
        
        output = torch.cat([output_res, output_res_2, output_res_4], 1)
        return output

    def forward(self, data):
        passes = []

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)

        x = self.fc1(x)
        x = self.final(x)
        # x = self.sig(x)

        return x
