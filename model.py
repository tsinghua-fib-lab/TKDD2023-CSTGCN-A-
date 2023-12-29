import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from typing import Union, Callable, Optional
from torch.utils.checkpoint import checkpoint


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()

        self.kernel_set = [2, 5]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class Conv2D(nn.Module):
    r"""An implementation of the 2D-convolution block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." 
    <https://arxiv.org/pdf/1911.08415.pdf>`_
    Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.relu.
        bn_decay (float, optional): Batch normalization momentum, default is None.
    """
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 kernel_size: Union[tuple, list],
                 stride: Union[tuple, list] = (1, 1),
                 use_bias: bool = True,
                 activation: Optional[Callable[[torch.FloatTensor],
                                               torch.FloatTensor]] = F.relu,
                 bn_decay: Optional[float] = None):
        
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(input_dims,
                                 output_dims,
                                 kernel_size,
                                 stride=stride,
                                 padding=0,
                                 bias=use_bias)
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        # 方差均匀分布
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the 2D-convolution block.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, input_dims).
        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, output_dims).
        """

        X = self._conv2d(X)
        X = self._batch_norm(X)
        if self._activation is not None:
            X = self._activation(X)

        return X


class FullyConnected(nn.Module):
    r"""An implementation of the fully-connected layer.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_
    Args:
        input_dims (int or list): Dimension(s) of input.
        units (int or list): Dimension(s) of outputs in each 2D convolution block.
        activations (Callable or list): Activation function(s).
        bn_decay (float, optional): Batch normalization momentum, default is None.
        use_bias (bool, optional): Whether to use bias, default is True.
    """
    def __init__(self,
                 input_dims: Union[int, list],
                 units: Union[int, list],
                 activations: Union[Callable[[torch.FloatTensor],
                                             torch.FloatTensor], list],
                 bn_decay: float,
                 use_bias: bool = True,
                 drop: float = None):
        
        super(FullyConnected, self).__init__()
        # 判断该量是否为该类型
        
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        
        assert type(units) == list
        
        self._conv2ds = nn.ModuleList([
            Conv2D(input_dims=input_dim,
                   output_dims=num_unit,
                   kernel_size=[1, 1],
                   stride=[1, 1],
                   use_bias=use_bias,
                   activation=activation,
                   bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)
        ])

        self.drop = drop

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the fully-connected layer.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, 1).
        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        for conv in self._conv2ds:
            if self.drop is not None:
                X = F.dropout(X, self.drop, training=self.training)
            X = conv(X)
        return X


class MYEmbedding(nn.Module):
    """
    An implementation of the spatial-temporal embedding block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_
    Args:
        D (int) : Dimension of output.
        bn_decay (float): Batch normalization momentum.
        steps_per_day (int): Steps to take for a day.
        use_bias (bool, optional): Whether to use bias in Fully Connected layers, default is True.
    """
    def __init__(self,
                 D_SE: int,
                 D: int,
                 bn_decay: float,
                 steps_per_day: int,
                 use_bias: bool = True):
        super(MYEmbedding, self).__init__()
        self._fully_connected_se = FullyConnected(input_dims=[D_SE, D],
                                                  units=[D, D],
                                                  activations=[F.relu, None],
                                                  bn_decay=bn_decay,
                                                  use_bias=use_bias)

        self._fully_connected_te = FullyConnected(
            input_dims=[steps_per_day + 7, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
            use_bias=use_bias)

        self.steps_per_day = steps_per_day

    def forward(self, SE: torch.FloatTensor, TE: torch.FloatTensor,
                T: int) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.
        Arg types:
            * **SE** (PyTorch Float Tensor) - Spatial embedding, with shape (num_nodes, D).
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, 2).(dayofweek, timeofday) 
            * **T** (int) - Number of time steps in one day.
        Return types:
            * **output** (PyTorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_his + num_pred, num_nodes, D).
        """

        SE = self._fully_connected_se(SE)

        timeofday = torch.nn.functional.one_hot(TE[:, :1, :].to(torch.int64),
                                                num_classes=self.steps_per_day)

        dayofweek = torch.nn.functional.one_hot(TE[:, 1:, :].to(torch.int64),
                                                num_classes=7)

        TE = torch.cat((dayofweek, timeofday), dim=-1)

        TE = self._fully_connected_te(TE.transpose(1, 3).float())
        del dayofweek, timeofday
        
        return SE + TE


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class nconv_events(nn.Module):
    def __init__(self):
        super(nconv_events, self).__init__()

    def forward(self, x, A):

        x = torch.einsum('nvl,vw->nwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in,
                                   c_out,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   stride=(1, 1),
                                   bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):

        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class end_gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout,adj_mx):
        super(end_gcn, self).__init__()
        self.nconv = nconv()
        c_in = c_in*3
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.adj_mx = adj_mx
    def forward(self, x):
        out = [x]
        x1 = self.nconv(x, self.adj_mx)
        out.append(x1)
        x2 = self.nconv(x1,self.adj_mx)
        out.append(x2)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn_events(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_events, self).__init__()
        self.nconv = nconv_events()
        c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):

        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

import math

class position_embedding(nn.Module):
    def __init__(self,
                 input_length,
                 num_of_vertices,
                 embedding_size,
                 temporal = True,
                 spatial = True):

        super(position_embedding, self).__init__()
        self.temporal_emb = None
        self.spatial_emb = None
        if temporal:
            self.temporal_emb = nn.Parameter(torch.randn(1, embedding_size, 1, input_length), requires_grad=True)
            nn.init.xavier_uniform_(self.temporal_emb, gain=math.sqrt(0.0003 / 6))
        if spatial:
            self.spatial_emb = nn.Parameter(torch.randn(1, embedding_size, num_of_vertices, 1), requires_grad=True)
            nn.init.xavier_uniform_(self.spatial_emb, gain=math.sqrt(0.0003 / 6))

    def forward(self, data):

        if self.temporal_emb is not None:
            data = data + self.temporal_emb
        if self.spatial_emb is not None:
            data = data + self.spatial_emb
        return data


class position_embedding_att(nn.Module):
    def __init__(self,
                 input_length,
                 num_of_vertices,
                 embedding_size,
                 temporal = True,
                 spatial = True):
        super(position_embedding_att, self).__init__()
        self.temporal_emb = None
        self.spatial_emb = None
        if temporal:
            self.temporal_emb = nn.Parameter(torch.randn(1, embedding_size, 1, 1, input_length), requires_grad=True)
            nn.init.xavier_uniform_(self.temporal_emb, gain=math.sqrt(0.0003 / 6))

        if spatial:
            self.spatial_emb = nn.Parameter(torch.randn(1, embedding_size, 1, num_of_vertices, 1), requires_grad=True)
            nn.init.xavier_uniform_(self.spatial_emb, gain=math.sqrt(0.0003 / 6))

    def forward(self, data):
        if self.temporal_emb is not None:
            data = data + self.temporal_emb
        if self.spatial_emb is not None:
            data = data + self.spatial_emb

        return data


class attention(nn.Module):
    def __init__(self, 
                len_days, 
                len_temp, 
                conv_channels, 
                att_channels,
                num_nodes):

        super(attention, self).__init__()
        len_temp = int(len_temp / (len_days + 1))
        self.fc_days = nn.Conv2d(in_channels=conv_channels, out_channels=att_channels, kernel_size=(1, len_temp))
        self.fc_query = nn.Conv2d(in_channels=conv_channels, out_channels=att_channels, kernel_size=(1, len_temp))
        self.len_days = len_days
        self.len_temp = len_temp
        self.conv_channels = conv_channels
        self.att_channels = att_channels
        self.num_nodes = num_nodes
        self.v = nn.Conv2d(in_channels=att_channels, out_channels=1, kernel_size=(1, 1))
        temporal_emb, spatial_emb = True, True
        self.position_embedding = position_embedding_att(len_temp, num_nodes, conv_channels, temporal_emb, spatial_emb)

    def forward(self, days_input, query):

        days_input = torch.stack(days_input, 2)
        days_input = self.position_embedding(days_input)
        days = self.fc_days(days_input.view(-1, self.conv_channels, self.len_days * self.num_nodes, self.len_temp))
        days = days.view(-1, self.att_channels, self.len_days, self.num_nodes)
        query = self.fc_query(query)
        att_tensor = torch.softmax( self.v ( torch.tanh( days + query.view (-1, self.att_channels, 1, self.num_nodes) ) ), 2)
        output = torch.sum(att_tensor.unsqueeze(-1) * days_input, dim=2)
        return output

class gwnet(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 dropout=0.3,
                 supports=None,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 in_dim=2,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=3,
                 adj_mx= None):
        
        super(gwnet, self).__init__()

        dilation_channels = 40
        self.adj_mx = adj_mx

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.att = nn.ModuleList()
        self.STE = nn.ModuleList()

        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        receptive_field = 1

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes,10).to(device),requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.nodevec1_events = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2_events = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10]**0.5))
                initemb2 = torch.mm(torch.diag(p[:10]**0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1,requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2,requires_grad=True).to(device)
                self.supports_len += 1

        self.len_period = 3
        residual_channels = 40
        conv_channels = 40
        skip_channels = 64
        end_channels = 128
        dilation_exponential = 1
        seq_length = 3 * 12
        self.seq_length = seq_length
        kernel_size = 5

        self.gcn_true = True
        
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential**layers - 1) /(dilation_exponential - 1)) * self.len_period
        
        else:
            self.receptive_field = (layers * (kernel_size - 1) + 1) * self.len_period
        
        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential**layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            
            new_dilation = 1
            
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential**j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels,
                                      conv_channels,
                                      dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels,
                                      conv_channels,
                                      dilation_factor=new_dilation))
                self.filter_convs1.append(
                    dilated_inception(residual_channels,
                                      conv_channels,
                                      dilation_factor=new_dilation))
                self.gate_convs1.append(
                    dilated_inception(residual_channels,
                                      conv_channels,
                                      dilation_factor=new_dilation))
                self.filter_convs2.append(
                    dilated_inception(residual_channels,
                                      conv_channels,
                                      dilation_factor=new_dilation))
                self.gate_convs2.append(
                    dilated_inception(residual_channels,
                                      conv_channels,
                                      dilation_factor=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(in_channels=conv_channels,
                              out_channels=residual_channels,
                              kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(in_channels = 2 * conv_channels,
                                  out_channels = skip_channels,
                                  kernel_size = (1, self.seq_length - self.len_period * rf_size_j + self.len_period * 1)))
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels = 2 * conv_channels,
                            out_channels = skip_channels,
                            kernel_size = (1, int((self.receptive_field - self.len_period * rf_size_j + self.len_period * 1) / 3))))

                if self.gcn_true:

                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

                len_days, len_temp, att_channels = 2, self.receptive_field - self.len_period * rf_size_j + self.len_period * 1, 64
                self.att.append(attention(len_days, len_temp, conv_channels, att_channels, num_nodes))
                len_temp = len_temp + (kernel_size - 1) * self.len_period
                temporal_emb, spatial_emb = True, True
                self.STE.append(position_embedding(len_temp, num_nodes, conv_channels, temporal_emb, spatial_emb))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= dilation_exponential
        
        if self.seq_length > self.receptive_field:

            self.skip0_x = nn.Conv2d(in_channels=residual_channels,
                                     out_channels=skip_channels,
                                     kernel_size=(1, self.seq_length),
                                     bias=True)
            self.skip0_congestion = nn.Conv2d(in_channels=residual_channels,
                                              out_channels=skip_channels,
                                              kernel_size=(1, self.seq_length),
                                              bias=True)
            self.skip0_time = nn.Conv2d(in_channels=residual_channels,
                                        out_channels=skip_channels,
                                        kernel_size=(1, self.seq_length),
                                        bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels,
                                   out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1),
                                   bias=True)

        else:

            self.skip0_x = nn.Conv2d(in_channels=residual_channels,
                                     out_channels=skip_channels,
                                     kernel_size=(1, self.receptive_field),
                                     bias=True)
            self.skip0_congestion = nn.Conv2d(in_channels=residual_channels,
                                              out_channels=skip_channels,
                                              kernel_size=(1, self.receptive_field),
                                              bias=True)
            self.skip0_time = nn.Conv2d(in_channels=residual_channels,
                                        out_channels=skip_channels,
                                        kernel_size=(1, self.receptive_field),
                                        bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels,
                                   out_channels=skip_channels,
                                   kernel_size=(1, self.len_period),
                                   bias=True)

        self.outdim_rnn = 128
        self.outdim_gcn = 64
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels + self.outdim_rnn,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=6 * 4,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_linear = nn.Linear(8,1)
        # self.input_cat1 = nn.Linear(36,24)
        self.input_cat1 = MLP(36,24)
        D = residual_channels
        bn_decay = 0.1
        self._fully_connected_1 = FullyConnected(input_dims=2, units=D, activations=F.relu, bn_decay=bn_decay)
        use_bias = True

        self.start_conv_x = nn.Conv2d(in_channels=residual_channels,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))
        self.start_conv_congestion = nn.Conv2d(in_channels=residual_channels,
                                               out_channels=residual_channels,
                                               kernel_size=(1, 1))
        self.start_conv_time = nn.Conv2d(in_channels=residual_channels,
                                         out_channels=residual_channels,
                                         kernel_size=(1, 1))
    # Event

        dim_emb = 16
        self.indim_rnn = 48
        
        self.emb_timeofday = nn.Embedding(290, dim_emb, padding_idx=0)
        self.emb_dayofholiday = nn.Embedding(10, dim_emb, padding_idx=0)
        self.emb_congestion = nn.Embedding(5 + 1, dim_emb)
        self.fc_events = nn.Linear(dim_emb * 2 + 2, 64)
        self.fc_events_2 = nn.Linear(64, self.indim_rnn)
        self.lstm = nn.LSTM(input_size=self.indim_rnn,
                            hidden_size=self.outdim_rnn,
                            num_layers=1,
                            dropout=0.1,
                            batch_first=True)#48,128
        self.gconv_events = gcn_events(self.outdim_rnn,
                                       self.outdim_gcn,
                                       dropout,
                                       support_len=self.supports_len)

        self.len_events = 10
        self.device = device
        self.middle_dim = 64
        self.fc_time_series = nn.Conv2d(in_channels=dim_emb * 3 + 1,
                                        out_channels=self.middle_dim,
                                        kernel_size=(1, 1))
        self.start_conv = nn.Conv2d(in_channels=self.middle_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.skip0 = nn.Conv2d(in_channels=self.middle_dim,
                               out_channels=skip_channels,
                               kernel_size=(1, self.receptive_field),
                               bias=True)
        self.end_gcn = end_gcn(end_channels, end_channels, dropout, self.adj_mx)

    def forward(self, x, events):
        assert events.size(-1) == 4, 'Shape of events is wrong!'
        input_x = x[:, 0:1, :, :]#
        input_cat = F.relu(self.input_cat1(input_x))
        events = events + 1
        duration = (events[..., 0:1]).float()
        timeofday = self.emb_timeofday(events[..., 1].long())
        dayofholiday = self.emb_dayofholiday(events[..., 2].long())   
        end_delta_t = (events[..., 3:] - 1).float()

        emb_events = torch.cat([duration, timeofday, dayofholiday, end_delta_t], -1)
        emb_events = self.fc_events(emb_events)
        emb_events = self.fc_events_2(torch.relu(emb_events))

        duration = events[..., 0]
        lengths = torch.where(duration > 0, torch.ones_like(duration), torch.zeros_like(duration))
        lengths = torch.sum(lengths, 1).flatten().to('cpu')
        mask1 = torch.where(lengths == 0, torch.ones_like(lengths),torch.zeros_like(lengths))
        mask2 = torch.where(lengths > 0, torch.ones_like(lengths),torch.zeros_like(lengths))

        lengths = lengths + mask1
        emb_events = emb_events.transpose(1, 2).contiguous().view( -1, self.len_events, self.indim_rnn)#emb_events.shape :[31104, 10, 48]
        pack = nn.utils.rnn.pack_padded_sequence(emb_events, lengths, batch_first=True, enforce_sorted=False)#length.shape:torch.Size([31104])
        out, (h, c) = self.lstm(pack)
        out_rnn = h[0]
        out_rnn = out_rnn * (mask2.unsqueeze(-1).to(self.device))
        out_rnn = out_rnn.view(-1, self.num_nodes, self.outdim_rnn)
        out_gnn = out_rnn.transpose(1, 2).contiguous().unsqueeze(-1)

        residual, residual1, residual2 = torch.chunk(x, 3, -1)
        residual = nn.functional.pad(residual, (1, 0, 0, 0))
        residual1 = nn.functional.pad(residual1, (1, 0, 0, 0))
        residual2 = nn.functional.pad(residual2, (1, 0, 0, 0))
        
        x = torch.cat((residual, residual1, residual2), -1)

        timeofday = self.emb_timeofday(x[:, -3, 0, :].long() + 1)
        dayofholiday = self.emb_dayofholiday(x[:, -2, 0, :].long() + 1)
        congestion = self.emb_congestion(x[:, -1, :, :].long())

        emb_time_series = torch.cat([x[:, :1, :, :], 
                                    timeofday.transpose(1, 2).unsqueeze(2).expand([-1, -1, self.num_nodes, -1]), 
                                    dayofholiday.transpose(1, 2).unsqueeze(2).expand([-1, -1, self.num_nodes, -1]), 
                                    congestion.permute(0, 3, 1, 2)], 
                                    dim = 1)
        emb_time_series = torch.relu(self.fc_time_series(emb_time_series))

        x = self.start_conv(emb_time_series)
        skip = self.skip0(emb_time_series)
        new_supports = None
        
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        for i in range(self.layers):
            x = self.STE[i](x)
            residual, residual1, residual2 = torch.chunk(x, 3, -1)

            filter = checkpoint(self.filter_convs[i], residual)
            filter = torch.tanh(filter)

            gate = checkpoint(self.gate_convs[i], residual)
            gate = torch.sigmoid(gate)

            filter1 = checkpoint(self.filter_convs1[i], residual1)
            filter1 = torch.tanh(filter1)

            gate1 = checkpoint(self.gate_convs1[i], residual1)
            gate1 = torch.sigmoid(gate1)

            filter2 = checkpoint(self.filter_convs2[i], residual2)
            filter2 = torch.tanh(filter2)

            gate2 = checkpoint(self.gate_convs2[i], residual2)
            gate2 = torch.sigmoid(gate2)

            temp, temp1, temp2 = filter * gate, filter1 * gate1, filter2 * gate2

            x = torch.cat((temp, temp1, temp2), -1)

            att_out = self.att[i]((temp, temp1), temp2)

            s = torch.cat((att_out, temp2), 1)
            s = F.dropout(s, self.dropout, training=self.training)
            s = checkpoint(self.skip_convs[i], s)
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            size_x = int(x.size(3) / 3)
            x = x + torch.cat(
                (residual[..., -size_x:], residual1[..., -size_x:],
                 residual2[..., -size_x:]), -1)
            x = self.bn[i](x)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = torch.cat([x, out_gnn], 1)
        x = F.relu(self.end_conv_1(x))
        # x = x.squeeze(-1)
        x = self.end_gcn(x)
        # x = x.unsqueeze(-1)
        
        x = self.end_conv_2(x)
        seg_emb = x.view(-1, 24, self.num_nodes).transpose(-1, -2).contiguous()
        out_ = x.view(-1, 6, 4, self.num_nodes).transpose(-1, -2).contiguous()
        input_cat = input_cat.view(-1, self.num_nodes, 6, 4).transpose(1, 2)
        output = torch.cat([input_cat, out_], -1)
        output = self.end_linear(output)

        return output,seg_emb

class MLP(nn.Module):
    def __init__(self, in_features, out_features):

        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features,64)
        self.linear2 = NLinear(64,out_features)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        seg = self.linear2(x)
        return seg

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len,pred_len):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Channel, Input length]
        seq_last = x[...,-1:].detach()
        x = x - seq_last
        x = self.Linear(x)
        x = x + seq_last
        return x # [Batch, Channel, Output length]