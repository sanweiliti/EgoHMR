from __future__ import absolute_import
import torch.nn as nn
from models.egohmr.modulated_gcn.modulated_gcn_conv import ModulatedGraphConv
from models.egohmr.modulated_gcn.graph_non_local import GraphNonLocal
from models.egohmr.modulated_gcn.nets.non_local_embedded_gaussian import NONLocalBlock2D


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv =  ModulatedGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)   # [bs, 512, 24]
        x = self.bn(x).transpose(1, 2)  # [bs, 24, 512]
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class ModulatedGCN(nn.Module):
    def __init__(self, adj, in_dim=3718, out_dim=6, hid_dim=128, num_layers=4, nonlocal_layer=False, p_dropout=None):
        super(ModulatedGCN, self).__init__()
        _gconv_input = [_GraphConv(adj, in_dim, hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))

        # if nodes_group is None:
        #     for i in range(num_layers):
        #         _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        # else:
        #     group_size = len(nodes_group[0])
        #     assert group_size > 1
        #
        #     grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
        #     restored_order = [0] * len(grouped_order)
        #     for i in range(len(restored_order)):
        #         for j in range(len(grouped_order)):
        #             if grouped_order[j] == i:
        #                 restored_order[i] = j
        #                 break
        #
        #     _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
        #     for i in range(num_layers):
        #         _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        #         _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        # self.actvn = nn.ReLU()
        # self.fc_in = nn.Linear(in_dim, resblk_hdim)

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = ModulatedGraphConv(hid_dim, out_dim, adj)
        self.nonlocal_layer = nonlocal_layer
        if self.nonlocal_layer:
            self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)

    def forward(self, x):
        # x: [bs, 24, D] D=3718
        # x = x.permute(0,2,1)  # [bs, D, 24]
        out = self.gconv_input(x)  # [bs, 24, 512]
        out = self.gconv_layers(out)  # [bs, 24, 512]

        if self.nonlocal_layer:
            out = out.unsqueeze(2)  # [bs, 24, 1, 512]
            out = out.permute(0,3,2,1)  # [bs, 512, 1, 24]
            out = self.non_local(out)  # [bs, 512, 1, 24]
            out = out.permute(0,3,1,2)  # [bs, 24, 512, 1]
            out = out.squeeze()  # [bs, 24, 512]

        out = self.gconv_output(out)  # [bs, 24, out_dim]
        # out = out.permute(0,2,1)
        # out = out.unsqueeze(2)
        # out = out.unsqueeze(4)
        return out
