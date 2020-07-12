import torch.nn as nn
import math
import torch
import torch.nn.functional as F

from models.gcn import GraphConvLayer

class StructInduction(nn.Module):
    def __init__(self, sem_dim_size, sent_hiddent_size, bidirectional):#, py_version):
        super(StructInduction, self).__init__()
        self.bidirectional = bidirectional
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size

        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tp_linear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)

        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tc_linear.weight)
        nn.init.constant_(self.tc_linear.bias, 0)

        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fi_linear.weight)

        self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.bilinear.weight)

        self.exparam = nn.Parameter(torch.Tensor(1, 1, self.sem_dim_size))
        torch.nn.init.xavier_uniform_(self.exparam)

        self.fzlinear = nn.Linear(3 * self.sem_dim_size, 2*self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.fzlinear.weight)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input):  # batch*sent * token * hidden

        batch_size, token_size, dim_size = input.size()

        """STEP1: Calculating Attention Matrix"""
        if (self.bidirectional):
            input = input.view(batch_size, token_size, 2, dim_size // 2)
            sem_v = torch.cat((input[:, :, 0, :self.sem_dim_size // 2], input[:, :, 1, :self.sem_dim_size // 2]), 2)
            str_v = torch.cat((input[:, :, 0, self.sem_dim_size // 2:], input[:, :, 1, self.sem_dim_size // 2:]), 2)
        else:
            sem_v = input[:, :, :self.sem_dim_size]
            str_v = input[:, :, self.sem_dim_size:]

        tp = torch.tanh(self.tp_linear(str_v))  # b*s, token, h1
        tc = torch.tanh(self.tc_linear(str_v))  # b*s, token, h1
        tp = tp.unsqueeze(2).expand(tp.size(0), tp.size(1), tp.size(1), tp.size(2)).contiguous()
        tc = tc.unsqueeze(2).expand(tc.size(0), tc.size(1), tc.size(1), tc.size(2)).contiguous()


        f_ij = self.bilinear(tp, tc).squeeze()  # b*s, token , token
        f_i = torch.exp(self.fi_linear(str_v)).squeeze()  # b*s, token

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
        A_ij = torch.exp(f_ij) * mask

        """STEP: Incude Latent Structure"""
        tmp = torch.sum(A_ij, dim=1)  # nan: dimension
        res = torch.zeros(batch_size, token_size, token_size).cuda()
        # tmp = torch.stack([torch.diag(t) for t in tmp])
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res  # A_ij has 0s as diagonals

        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i

        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:, :, 0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)

        mask1 = torch.cat([temp11, temp12], 2).cuda()
        mask2 = torch.cat([temp21, temp22], 1).cuda()

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)

        ssr = torch.cat([self.exparam.repeat(batch_size, 1, 1), sem_v], 1)
        pinp = torch.bmm(df, ssr)

        cinp = torch.bmm(dx, sem_v)

        finp = torch.cat([sem_v, pinp, cinp], dim=2)

        output = F.relu(self.fzlinear(finp))

        return output, df

class StructInductionNoSplit(nn.Module):
    def __init__(self, sent_hiddent_size, bidirectional):#, py_version):
        super(StructInductionNoSplit, self).__init__()
        self.bidirectional = bidirectional
        self.str_dim_size = sent_hiddent_size #- self.sem_dim_size

        self.model_dim = sent_hiddent_size

        self.linear_keys = nn.Linear(self.model_dim, self.model_dim)
        self.linear_query = nn.Linear(self.model_dim, self.model_dim)
        self.linear_root = nn.Linear(self.model_dim, 1)

    def forward(self, input):  # batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        key = self.linear_keys(input)
        query = self.linear_query(input)
        f_i = self.linear_root(input).squeeze(-1)
        query = query / math.sqrt(self.model_dim)
        f_ij = torch.matmul(query, key.transpose(1, 2))

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
        A_ij = torch.exp(f_ij) * mask

        tmp = torch.sum(A_ij, dim=1)  # nan: dimension
        res = torch.zeros(batch_size, token_size, token_size).cuda()
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res  # A_ij has 0s as diagonals

        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i

        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:, :, 0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)

        mask1 = torch.cat([temp11, temp12], 2).cuda()
        mask2 = torch.cat([temp21, temp22], 1).cuda()

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)

        output = None

        return output, df

def b_inv(b_mat):
    eye = torch.rand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).cuda()
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

class DynamicReasoner(nn.Module):
    def __init__(self, hidden_size, gcn_layer, dropout_gcn):
        super(DynamicReasoner, self).__init__()
        self.hidden_size = hidden_size
        self.gcn_layer = gcn_layer
        self.dropout_gcn = dropout_gcn
        self.struc_att = StructInduction(hidden_size // 2, hidden_size, True)
        self.gcn = GraphConvLayer(hidden_size, self.gcn_layer, self.dropout_gcn, self_loop=True)

    def forward(self, input):
        '''
        :param input:
        :return:
        '''
        '''Structure Induction'''
        _, att = self.struc_att(input)
        '''Perform reasoning'''
        output = self.gcn(att[:, :, 1:], input)
        return output
