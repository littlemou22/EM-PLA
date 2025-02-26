import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentiveFP
from models.pl_model import HeteroGNN
from models.pro_model import ProCNN
from models.CAPLA.CAPLA import CAPLA
from torch_geometric.utils import to_dense_batch
from models.pro_egnn import EGNNModel
from models.pro_egnn import compute_edge_index

class LBAPredictor(torch.nn.Module):
    def __init__(self, metadata, mode):
        super().__init__()
        self.mode = mode
        self.heterognn = HeteroGNN(metadata, edge_dim=10, hidden_channels=64, out_channels=8, num_layers=3, mode=self.mode)
        self.ligandgnn = AttentiveFP(in_channels=18, hidden_channels=64, out_channels=16, edge_dim=12, num_timesteps=3,
                                     num_layers=3, dropout=0.3)
        self.procnn = ProCNN(seq_input_dim=1024, hid_dim=128, seq_layers=3, map_layers=2, mode=mode)
        
        # if('egnnAll' in self.mode):
        #     self.egnn = EGNNModel(input_features_size=1024, hidden_channels=64, out_dim=128, edge_features=0, 
        #                         num_egnn_layers=12, num_layers=1, mode=mode)
        #     # self.capla = CAPLA(seq_input_size=21, smi_input_size=18, out_dim=128, mode=mode)
        # else:
        self.egnn = EGNNModel(input_features_size=21, hidden_channels=64, out_dim=128, edge_features=0, 
                            num_egnn_layers=12, num_layers=1, mode=mode)
        self.capla = CAPLA(seq_input_size=40, smi_input_size=18, out_dim=128, mode=mode)
        # if('EnvCenter' in self.mode):
        #     self.env_mlp1 = nn.Linear(5, 40)
        #     self.env_mlp2 = nn.Linear(40, 128)
        p_emb=0
        if('complex' in self.mode):
            p_emb = 128*3
            if('capla' not in self.mode):
                p_emb += 128
        elif('simple' in self.mode):
            p_emb = 128
            if('capla' in self.mode):
                p_emb += 128*3
            if('noSeqComplex' in self.mode):
                p_emb = p_emb-128*2
        # print(mode, p_emb)
        if('noEnv' in self.mode):
            p_emb = 560-(16+32*3)
        if('noHGT' in self.mode):
            p_emb = p_emb-32*3
        if('noAllSeq' in self.mode):
            p_emb-=128
        if('noAE' in self.mode):
            p_emb-=128
            p_emb-=16
        self.out = nn.Sequential(
            nn.Linear(16+32*3+p_emb, 256),  #16+32*3+p_emb
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, data):
        if(len(data)==7):
            g_l = data[0]
            g_pl = data[1]
            p_all, p_part, p_sec_all, p_sec_part, p_graph = data[2], data[3], data[4], data[5], data[6]
        elif(len(data)==5):
            g_l = data[0]
            g_pl = data[1]
            p_sec_all, p_sec_part, p_graph = data[2], data[3], data[4]
        
        # print(len(data))
        
        # if('egnnAll' in self.mode):
        #     p_graph.x = p_all.seq
        #     p_graph.coords = p_all.coord
        #     p_graph.batch = p_all.seq_batch
        #     p_graph.edge_index = compute_edge_index(p_graph.coords, p_graph.batch, k=5, threshold=6.0).to(p_graph.x.device)
        # else:
        try:
            p_graph.coords = p_graph.x
            p_graph.x = F.one_hot(p_graph.seq, num_classes=21).float()
        except:
            print()

        if('complex' in self.mode):
            print(1)
            # l = self.ligandgnn(x=g_l.x, edge_index=g_l.edge_index, edge_attr=g_l.edge_attr, batch=g_l.batch)
            # complex = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
            # p = self.procnn(p_all, p_part, p_graph)
            # emb = torch.cat((l, complex, p), dim=1)
        elif('simple' in self.mode):
            # if('noHGT' in self.mode):
            #     l = self.ligandgnn(x=g_l.x, edge_index=g_l.edge_index, edge_attr=g_l.edge_attr, batch=g_l.batch)
            #     p = self.egnn(p_graph)
            #     emb = torch.cat((l, p), dim=1)
            # else:
            
            if('noAE' in self.mode):
                complex = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
                emb = complex
            elif('noHGT' in self.mode):
                l = self.ligandgnn(x=g_l.x, edge_index=g_l.edge_index, edge_attr=g_l.edge_attr, batch=g_l.batch)
                p = self.egnn(p_graph)
                emb = torch.cat((l, p), dim=1)
            else:
                l = self.ligandgnn(x=g_l.x, edge_index=g_l.edge_index, edge_attr=g_l.edge_attr, batch=g_l.batch)
                complex = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
                p = self.egnn(p_graph)
                emb = torch.cat((l, complex, p), dim=1)
            # emb = complex
            if('capla' in self.mode):
                pro, pro_mask = to_dense_batch(p_sec_all.seq, p_sec_all.seq_batch, max_num_nodes=1024)
                pkt, pkt_mask = to_dense_batch(p_sec_part.seq, p_sec_part.seq_batch, max_num_nodes=64)
                # pro, pro_mask = to_dense_batch(p_all.seq, p_all.seq_batch, max_num_nodes=1024)
                # pkt, pkt_mask = to_dense_batch(p_part.seq, p_part.seq_batch, max_num_nodes=64)
                lig, lig_mask = to_dense_batch(g_l.x, g_l.batch, max_num_nodes=150)
                

                capla = self.capla(pro, pkt, lig, None)
                emb = torch.cat((emb, capla), dim=1)
        # print(emb.shape)
        y_hat = self.out(emb)
        return torch.squeeze(y_hat)