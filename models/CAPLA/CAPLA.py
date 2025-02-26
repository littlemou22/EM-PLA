import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.CAPLA.self_attention import EncoderLayer
# from dataset import PT_FEATURE_SIZE

SMILESCLen = 18    # SMILES Char Num


class Squeeze(nn.Module):   #Dimention Module
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class DilatedConv(nn.Module):     # Dilated Convolution
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedConvBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  # Down Dimention
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)    # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)     # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)     # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)     # Dilated scale:8(2^3)
        self.d16 = DilatedConv(n, n, 3, 1, 16)   # Dilated scale:16(2^4)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)

        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class DilatedConvBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)  # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)   # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)   # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)   # Dilated scale:8(2^3)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):

        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class CAPLA(nn.Module):
    def __init__(self, seq_input_size, smi_input_size, out_dim, mode):
        super().__init__()
        self.mode = mode
        smi_embed_size = 128
        seq_embed_size = 128

        seq_oc = out_dim
        pkt_oc = out_dim
        smi_oc = out_dim
        td_oc = 32

        # SMILES, POCKET, PROTEIN Embedding
        self.smi_embed = nn.Linear(smi_input_size, smi_embed_size)
        self.seq_embed = nn.Linear(seq_input_size, seq_embed_size)


        # Global DilatedConv Module
        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedConvBlockA(ic, oc))
            ic = oc
        if('simple' in self.mode):
            conv_seq.append(nn.AdaptiveMaxPool1d(1))
            conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        # Pocket DilatedConv Module
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3, padding=1))
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        if('simple' in self.mode):
            conv_pkt.append(nn.AdaptiveMaxPool1d(1))
            conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)

        
        td_conv = []
        ic = 1
        for oc in [16, 32, td_oc * 2]:
            td_conv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        if('simple' in self.mode):
            td_conv.append(nn.AdaptiveMaxPool1d(1))
            td_conv.append(Squeeze())
        self.td_conv = nn.Sequential(*td_conv)


        td_onlyconv = []
        ic = 1
        for oc in [16, 32, td_oc]:
            td_onlyconv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        self.td_onlyconv = nn.Sequential(*td_onlyconv)


        # Ligand DilatedConv Module
        conv_smi = []
        ic = smi_embed_size

        
        # Cross-Attention Module
        self.smi_attention_poc = EncoderLayer(128, 128, 0.1, 0.1, 2)  # 注意力机制
        self.tdpoc_attention_tdlig = EncoderLayer(32, 64, 0.1, 0.1, 1)
        if('EnvCenter' in self.mode):
            self.env_attention = EncoderLayer(128, 128, 0.1, 0.1, 2)
        
        self.adaptmaxpool = nn.AdaptiveMaxPool1d(1)
        self.squeeze = Squeeze()

        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedConvBlockB(ic, oc))
            ic = oc
        if('simple' in self.mode):
            conv_smi.append(nn.AdaptiveMaxPool1d(1))
            conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)


        # Dropout
        self.cat_dropout = nn.Dropout(0.2)

    def forward(self, seq, pkt, coords, env):
        # print(pkt.shape)
        # D(B_s,N,L)
        seq_embed = self.seq_embed(seq)
        if('EnvCenter' in self.mode):
            env_embed = env
        seq_embed = torch.transpose(seq_embed, 1, 2)
        seq_conv = self.conv_seq(seq_embed)

        # print(pkt.shape)
        pkt_embed = self.seq_embed(pkt)
        smi_embed = self.smi_embed(coords)

        if('EnvCenter' in self.mode):
            smi_embed = self.env_attention(smi_embed, env_embed)
            pkt_embed = self.env_attention(pkt_embed, env_embed)
        smi_attention = smi_embed

        smi_embed = self.smi_attention_poc(smi_embed, pkt_embed)
        pkt_embed = self.smi_attention_poc(pkt_embed, smi_attention)

        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)

        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)
        
        if('simple' in self.mode):
            if('noSeqComplex' in self.mode):
                concat = seq_conv
            elif('noAllSeq' in self.mode):
                concat = torch.cat([pkt_conv, smi_conv], dim=1)
            else:
                concat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)
        else:
            concat = torch.cat([seq_conv.transpose(1, 2), pkt_conv.transpose(1, 2), smi_conv.transpose(1, 2)], dim=1)

        concat = self.cat_dropout(concat)

        # output = self.classifier(concat)
        return concat
