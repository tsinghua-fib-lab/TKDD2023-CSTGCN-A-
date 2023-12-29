import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResnetModel(nn.Module):
    def __init__(self, state_dim: int,  h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()

        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm


        self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_ac1 = nn.ReLU()
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                res_ac2 = nn.ReLU()
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1,res_ac1, res_fc2, res_bn2,res_ac2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_ac1 = nn.ReLU()
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_ac2 = nn.ReLU()
                self.blocks.append(nn.ModuleList([res_fc1,res_ac1, res_fc2, res_ac2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet
        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        # x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                # x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                # x = F.relu(x)
                x = self.blocks[block_num][1](x)

            # x = F.relu(x + res_inp)
            x = x + res_inp
        # output
        x = self.fc_out(x)
        return x

class actor_fleet(nn.Module):
    def __init__(
        self,
        out_dim,
        nodes_num,
        time_num,
        embedding_dim=32,
        h1_dim = 128, 
        resnet_dim = 64, 
        num_resnet_blocks = 4,
        batch_norm = False,
        seg_emb_dim = 24,
        lat_embedding_dim = 6,
        lng_embedding_dim = 6,
        dis_embedding_dim = 6,
        time_embedding_dim = 4,
        weekday_embedding_dim = 2,
        id_embedding_dim = 8
    ):

        super(actor_fleet, self).__init__()
        # self.bidirectional = True
        # self.GRU = nn.GRU(input_size = seg_emb_dim, hidden_size = out_dim, num_layers = 2, bidirectional = True)
        # self.fc1 = nn.Linear(out_dim*2,out_dim)
        self.h1_dim = h1_dim
        self.out_dim = out_dim
        self.resnet_dim = resnet_dim
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm

        self.gps_lat_embedding = nn.Linear(1, lat_embedding_dim)
        self.gps_lng_embedding = nn.Linear(1, lng_embedding_dim)
        self.dis_embdding = nn.Linear(1, dis_embedding_dim)
        
        self.time_embedding = nn.Embedding(time_num+1, time_embedding_dim)
        self.weekday_embedding = nn.Embedding(7+1, weekday_embedding_dim)
        self.id_embedding = nn.Embedding(nodes_num+1, id_embedding_dim)
        
        self.Positive = nn.ReLU()
        self.state_dim = lat_embedding_dim*2+lng_embedding_dim*2+dis_embedding_dim*2+time_embedding_dim+weekday_embedding_dim+seg_emb_dim*2+id_embedding_dim*2

        self.resnet = ResnetModel(state_dim = self.state_dim,  h1_dim = self.h1_dim, resnet_dim = self.resnet_dim, num_resnet_blocks = self.num_resnet_blocks, out_dim = self.out_dim, batch_norm = self.batch_norm)
        self.final_linear = nn.Sequential(nn.Linear(out_dim,out_dim),nn.ReLU(),nn.Linear(out_dim,1))
        self.dis_linear = nn.Sequential(nn.Linear(out_dim,out_dim),nn.ReLU(),nn.Linear(out_dim,1))
        # self.final_linear = nn.Linear(out_dim,1)
        # self.att_nn = MultiHeadAttention(n_head = 2, d_model = 32, d_q = 32, d_v = 32)
        # self.att_ns = MultiHeadAttention(n_head = 2, d_model = 32, d_q = 32, d_v = 24)
        self._init_embeddings()

    def _init_embeddings(self):
        for embedding in [self.gps_lat_embedding,self.gps_lng_embedding,self.dis_embdding,self.time_embedding,self.weekday_embedding,self.id_embedding]:
            embedding.weight.data.normal_(0, 0.1) 
    def forward(self,state):
        time_of_day = self.time_embedding(state[:,0].long())
        state[:,1] = 1
        weekday = self.weekday_embedding(state[:,1].long())
        id_s = self.id_embedding(state[:,3].long())
        id_e = self.id_embedding(state[:,4].long())
        gps_lat_s = self.gps_lat_embedding(state[:,5].unsqueeze(1))
        gps_lng_s = self.gps_lng_embedding(state[:,6].unsqueeze(1))
        gps_lat_e = self.gps_lat_embedding(state[:,7].unsqueeze(1))
        gps_lng_e = self.gps_lng_embedding(state[:,8].unsqueeze(1))
        if state.shape[1] == 43:
            dis_s = self.dis_embdding(state[:,25].unsqueeze(1))
            dis_e = self.dis_embdding(state[:,42].unsqueeze(1))
            state_emb = torch.cat([state[:,9:25],state[:,26:42]],-1)
        elif state.shape[1] == 59:
            dis_s = self.dis_embdding(state[:,33].unsqueeze(1))
            dis_e = self.dis_embdding(state[:,58].unsqueeze(1))
            state_emb = torch.cat([state[:,9:33],state[:,34:58]],-1)
            
        else:
            print('state.shape[1] is wrong',state.shape[1])
        


        state_f = torch.cat([time_of_day,weekday,id_s,gps_lat_s,gps_lng_s,id_e,gps_lat_e,gps_lng_e,dis_s,dis_e,state_emb],-1)

        x_representation = self.resnet(state_f)

        # x_representation = self.resnet(x)
        # x = self.Positive(x)
        # return x,x_representation
        x = self.final_linear(x_representation)
        x_dis = self.dis_linear(x_representation)
        return x,x_dis,x_representation

# class route_emb_net(nn.Module):
#     def __init__(
#         self,
#         out_dim,
#         h1_dim = 256, 
#         seg_emb_dim = 24
#     ): 
#         super(route_emb_net, self).__init__()
#         self.bidirectional = True
#         self.GRU = nn.GRU(input_size = seg_emb_dim, hidden_size = out_dim, num_layers = 2, bidirectional = True)
#         self.fc1 = nn.Sequential(nn.Linear(out_dim*2,out_dim),nn.ReLU(),nn.Linear(out_dim,out_dim))
#         self.h1_dim = h1_dim
#         self.out_dim = out_dim

#     def forward(self,astar_seg_embs_package):
#         _, hn = self.GRU(astar_seg_embs_package)
#         h = hn[-(1 + int(self.bidirectional)):] 
#         x_h = torch.cat(h.split(1), dim=-1).squeeze(0) 
#         x_route_emb = self.fc1(x_h) 
#         return x_route_emb
class route_emb_net(nn.Module):
    def __init__(
        self,
        out_dim,
        seg_emb_dim
    ): 
        super(route_emb_net, self).__init__()
        self.LSTM = nn.LSTM(input_size = seg_emb_dim, hidden_size = out_dim, num_layers = 2, batch_first = True)
        self.fc1 = nn.Sequential(nn.Linear(out_dim,out_dim*2),nn.ReLU(),nn.Linear(out_dim*2,out_dim))
        # self.to_travel_time = ResnetModel(state_dim = out_dim,  h1_dim = 128, num_resnet_blocks = 3, out_dim = 1, batch_norm = True)

    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim = 1, keepdim = False)
        lens = Variable(torch.unsqueeze(lens, dim = 1), requires_grad = False).float()
        hiddens = hiddens / lens  
        return hiddens

    def forward(self,packed_inputs,lens):
        packed_hiddens, (h_n, c_n) = self.LSTM(packed_inputs)
        hiddens, lens_ = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)
        hn = self.mean_pooling(hiddens, lens)
        x_route_emb = self.fc1(hn)
        return x_route_emb
    
class Whole_net(nn.Module):
    def __init__(
        self,
        out_dim,
        nodes_num,
        time_num,
        seg_emb_dim = 30
    ): 
        super(Whole_net, self).__init__()
        self.net1 = actor_fleet(out_dim,nodes_num,time_num)
        self.net2 = route_emb_net(out_dim,seg_emb_dim)
        self.LNT = True
        
    def forward(self,state,astar_seg_embs_package,lens):

        x,x_dis,x_representation = self.net1(state)
        if self.if_training:   
            lens_ = lens.clone().detach().cpu()
            times = astar_seg_embs_package[...,0].long()
            days = astar_seg_embs_package[...,1].long()
            times_emb = self.net1.time_embedding(times)
            days_emb = self.net1.weekday_embedding(days)
            astar_seg_embs_package = torch.cat([astar_seg_embs_package[...,2:],times_emb,days_emb],dim = -1).float()
            packed_inputs = nn.utils.rnn.pack_padded_sequence(astar_seg_embs_package, lens_, batch_first = True,enforce_sorted = False)
            x_route_emb = self.net2(packed_inputs,lens)
            # mean
            # x = self.net1.final_linear(x_representation)
            x_2 = self.net1.final_linear(x_route_emb)

            # x_dis = self.net1.dis_linear(x_representation)
            x_dis_2 = self.net1.dis_linear(x_route_emb)

            if self.LNT:
                # x_dis = (x_dis+x_dis_2)/2
                # x = (x+x_2)/2
                x_dis = x_dis_2
                x = x_2
            
            return x,x_dis,x_representation,x_route_emb
            # y = self.to_travel_time_r(x_route_emb)
            # return y,x_representation,x_route_emb
            
        else:
            return x,x_dis,x_representation
        # x_route_emb = self.net2(astar_seg_embs_package,astar_routes_len_batch)
        # return x,x_dis,x_representation,x_route_emb


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_q, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_model = d_model
        self.d_v = d_v

        self.w_qs = nn.Linear(d_q, n_head * d_model, bias=False)
        self.w_ks = nn.Linear(d_v, n_head * d_model, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_model, bias=False)
        self.fc = nn.Linear(n_head * d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_q, d_v, n_head = self.d_model, self.d_model, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_q)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_v)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        # return q, attn
        return q
