import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
class DataLoader(object):
    def __init__(self, data, batch_size, device, scaler = None,scaler_dis = None):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.device = device
        states_nnets = data["states_nnets"]
        self.states_nnets = np.concatenate(states_nnets,axis=1).squeeze()
        astar_labels = data["astar_labels"]
        if type(data["astar_dis_labels"]) == list:
            astar_dis_labels = np.concatenate(data["astar_dis_labels"])
        else:
            astar_dis_labels = data["astar_dis_labels"]
        self.astar_labels = np.concatenate(astar_labels)
        self.astar_seg_embs_not_pads = data["astar_seg_embs_not_pads"]
        self.astar_seg_days_not_pads = data["astar_seg_days_not_pads"]
        self.astar_seg_times_not_pads = data["astar_seg_times_not_pads"]
        # for seg in self.astar_seg_embs_not_pads:
        #     seg = seg.to(device)
        self.seg_embs_pad = pad_sequence(self.astar_seg_embs_not_pads, batch_first = True)
        self.seg_times_pad = pad_sequence(self.astar_seg_times_not_pads, batch_first = True)
        self.seg_days_pad = pad_sequence(self.astar_seg_days_not_pads, batch_first = True)
        self.seg_embs = torch.cat([self.seg_times_pad.unsqueeze(-1),self.seg_days_pad.unsqueeze(-1),self.seg_embs_pad],dim = -1)
        astar_routes_lens = data["astar_routes_lens"]
        self.astar_routes_lens = np.array(astar_routes_lens).flatten()
        self.size = len(self.states_nnets)
        self.num_batch = int(self.size // self.batch_size)
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler(mean = np.mean(self.astar_labels), std = np.std(self.astar_labels))
        if scaler_dis is not None:
            self.scaler_dis = scaler_dis
        else:
            self.scaler_dis = StandardScaler(mean = np.mean(astar_dis_labels), std = np.std(astar_dis_labels))
        self.astar_labels = self.scaler.transform(self.astar_labels)
        self.astar_dis_labels = self.scaler_dis.transform(astar_dis_labels)

    def get_scaler(self):
        return self.scaler,self.scaler_dis

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs,x3,x4,x5,x_dis = self.states_nnets[permutation],self.astar_labels[permutation],self.seg_embs[permutation],self.astar_routes_lens[permutation],self.astar_dis_labels[permutation]
        self.states_nnets = xs
        self.astar_labels = x3
        self.seg_embs = x4
        self.astar_routes_lens = x5
        self.astar_dis_labels = x_dis

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = [self.states_nnets[start_ind: end_ind, ...]]
                x3 = self.astar_labels[start_ind: end_ind]
                x_32 = self.astar_dis_labels[start_ind: end_ind]
                x4 = self.seg_embs[start_ind: end_ind].to(self.device)
                x5 = self.astar_routes_lens[start_ind: end_ind].tolist()
                yield (x_i, x3,x_32,x4,x5)
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return self.num_batch
    
    def __iter__(self):
        return self.get_iterator()

class DataLoader_Edit(object):
    def __init__(self, data, batch_size, device, env, if_use_mf, if_add_anode, scaler = None,scaler_dis = None):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.device = device
        states_nnets = data["states_nnets"]
        self.states_nnets = np.concatenate(states_nnets,axis=1).squeeze()
        self.states_nnets = env.edit_feas(self.states_nnets, if_use_mf, if_add_anode) 

        astar_labels = data["astar_labels"]
        astar_dis_labels = data["astar_dis_labels"]
        self.astar_labels = np.concatenate(astar_labels)
        self.astar_dis_labels = np.concatenate(astar_dis_labels)
        self.astar_seg_embs_not_pads = data["astar_seg_embs_not_pads"]
        self.astar_seg_days_not_pads = data["astar_seg_days_not_pads"]
        self.astar_seg_times_not_pads = data["astar_seg_times_not_pads"]
        # for seg in self.astar_seg_embs_not_pads:
        #     seg = seg.to(device)
        self.seg_embs_pad = pad_sequence(self.astar_seg_embs_not_pads, batch_first = True)
        self.seg_times_pad = pad_sequence(self.astar_seg_times_not_pads, batch_first = True)
        self.seg_days_pad = pad_sequence(self.astar_seg_days_not_pads, batch_first = True)
        self.seg_embs = torch.cat([self.seg_times_pad.unsqueeze(-1),self.seg_days_pad.unsqueeze(-1),self.seg_embs_pad],dim = -1)
        astar_routes_lens = data["astar_routes_lens"]
        self.astar_routes_lens = np.array(astar_routes_lens).flatten()
        self.size = len(self.states_nnets)
        self.num_batch = int(self.size // self.batch_size)
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler(mean = np.mean(self.astar_labels), std = np.std(self.astar_labels))
        if scaler_dis is not None:
            self.scaler_dis = scaler_dis
        else:
            self.scaler_dis = StandardScaler(mean = np.mean(self.astar_dis_labels), std = np.std(self.astar_dis_labels))

        self.astar_labels = self.scaler.transform(self.astar_labels)
        self.astar_dis_labels = self.scaler_dis.transform(astar_dis_labels)

    def get_scaler(self):
        return self.scaler,self.scaler_dis

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs,x3,x4,x5,x_dis = self.states_nnets[permutation],self.astar_labels[permutation],self.seg_embs[permutation],self.astar_routes_lens[permutation],self.astar_dis_labels[permutation]
        self.states_nnets = xs
        self.astar_labels = x3
        self.seg_embs = x4
        self.astar_routes_lens = x5

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = [self.states_nnets[start_ind: end_ind, ...]]
                x3 = self.astar_labels[start_ind: end_ind]
                x_32 = self.astar_dis_labels[start_ind: end_ind]
                x4 = self.seg_embs[start_ind: end_ind].to(self.device)
                x5 = self.astar_routes_lens[start_ind: end_ind].tolist()
                yield (x_i, x3,x_32, x4,x5)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss),torch.mean(preds),torch.mean(labels)

def masked_AEE(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = preds-labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_OE(preds, labels, null_val=np.nan):
   
    return sum((preds - labels)>0)/labels.shape[0]

