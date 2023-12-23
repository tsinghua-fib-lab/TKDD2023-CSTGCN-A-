import os
import sys
sys.path.append(r"/home/user/RILI_online")
import pickle
from argparse import ArgumentParser
from typing import Any, Dict
import environments.myenv_a as myenv
import numpy as np
from basic.astar_asnn import basic_label
import warnings

warnings.simplefilter("ignore") 
def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # hyper-parameters
    parser.add_argument('--time_num', type=int,default=288, help="")
    # parser.add_argument('--nodes_num', type=int,default=513, help="")
    parser.add_argument('--nodes_num', type=int,default=1053, help="")
    # Training
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    parser.add_argument('--mode', type=str, default="test", help="")
    parser.add_argument('--dataset_source', type=str, default="./data_paper/data_bd/data_notbaseline/", help="")
    parser.add_argument('--dataset_source2', type=str, default="./data_paper/data_bd_pred/", help="")
    parser.add_argument('--dataset_roadid_length', type=str, default="./data_paper/data_bd_pred/roadid_length.pkl", help="")
    parser.add_argument('--dataset_dir', type=str, default="./dataset_traj_bd/", help="")
    # parse arguments
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)
    print("Batch size: %i" % args_dict['batch_size'])
    if not os.path.exists(args_dict['dataset_dir']):
        os.makedirs(args_dict['dataset_dir'])
    return args_dict

def main(args_dict):

    #load data
    if args_dict["mode"] == "train":
        data_train = np.load(args_dict["dataset_source"]+"val.npz")['y'][:,0,:,:]#(2016, 6, 486, 4)
        args_dict["time_num"] = data_train.shape[0]
        args_dict["nodes_num"] = data_train.shape[1]
        
    elif args_dict["mode"] == "val":
        data_train = np.load(args_dict["dataset_source"]+"val.npz")['y'][:,0,:,:]
        args_dict["time_num"] = data_train.shape[0]
        args_dict["nodes_num"] = data_train.shape[1]
    elif args_dict["mode"] == "test":
        data_train = np.load(args_dict["dataset_source"]+"test.npz")['y'][:,0,:,:]#(288, 6, 486, 4)->(288, 486, 4)
        args_dict["time_num"] = data_train.shape[0]
        args_dict["nodes_num"] = data_train.shape[1]
    dataset_dir = args_dict["dataset_dir"]+args_dict["mode"]+".pkl"

    seg_embs = np.load(args_dict["dataset_source2"]+"TestSeg_emb.npy")
    if seg_embs.shape[1] == args_dict["nodes_num"]:
        # 在第一行添加一行全0的向量
        seg_embs = np.concatenate([np.zeros([seg_embs.shape[0],1,seg_embs.shape[2]]),seg_embs],axis=1)
    if args_dict["mode"] == "train":
        num_data = 100
    else:
        num_data = 30

    with open(args_dict["dataset_source2"]+"node_adj_c.pkl","rb") as f:
        state_tran = pickle.load(f)
    
    with open(args_dict["dataset_source2"]+"node_gps_dict.pkl","rb") as f:
        link_gps_start = pickle.load(f)
    for key in sorted(link_gps_start.keys()):
        if link_gps_start[key] == []:
            link_gps_start[key] = link_gps_start[key-1]

    with open(args_dict["dataset_source2"]+"nodeid_pair_to_roadid.pkl","rb") as f:
        np_to_rid = pickle.load(f)
    # node_adj = copy.deepcopy(state_tran)
    # node_adj[0] = [0,0,0,0]

    with open(args_dict["dataset_source2"]+"node_segs_c.pkl","rb") as f:
        node_segs = pickle.load(f)
    
    
    node_connected_list = np.load(args_dict["dataset_source2"]+"node_connected_list.npy").tolist()
    # 对路段长度进行补全
    with open(args_dict["dataset_roadid_length"],"rb") as f:
        dis_dict = pickle.load(f) # 节点编号从1开始的
    min_dis = np.mean(list(dis_dict.values()))
    # min_gps = np.mean(list(link_gps_start.values()),axis=0).tolist()
    link_num = data_train.shape[1]
    for i in range(1,link_num+1):
        if i not in dis_dict.keys():
            dis_dict[i] = min_dis
    # GPS 信息没有！！！！！！@@@
    # with open("./MYRL_bd/data_bd_pred/node_gps.pkl","rb") as f:
    #     node_gpsl = pickle.load(f)   

    #environment
    env = myenv.GridWorld(args_dict["time_num"], args_dict["nodes_num"])
    env.setStateTransition(state_tran,node_connected_list)
    env.setNtoR(np_to_rid)
    env.set_fea_data(seg_embs, node_segs,link_gps_start)
    env.set_mode(args_dict["mode"])
    env.setPassTimeMatrix(data_train[:,:,0])
    env.setRoadinfo(data_train)# 读取路段的time of day 和 day of week等信息
    env.setDistance(dis_dict)

    queries = []
    astar_labels = []#存放batch*1的数组
    astar_dis_labels = []
    astar_seg_embs_not_pads = []#列表存放 batch*不定长*24  用之前需要embedding
    astar_seg_days_not_pads = []
    astar_seg_times_not_pads = []
    astar_routes_lens = [] #列表存放 batch*1 长度
    states_nnets = []
    masks_nnets = []


    #生成1000*1的numpy数组
    for i in range(num_data):
        print("i:",i)
        # print("generating...")
        train_list_set,states_itr, mask_np = env.generate_states(args_dict["batch_size"])
        # print("generated")
        # 节点编号是从1开始的，通行时间矩阵是numpy数组，索引从0开始
        astar_label,astar_dis_label,astar_seg_embs_not_pad,astar_routes_len,seg_days,seg_times = basic_label(data_train,dis_dict,state_tran,train_list_set,seg_embs,args_dict["mode"],np_to_rid)
        states_nnet = env.state_to_nnet_input(states_itr)
        masks_nnet = env.mask_to_nnet_input(mask_np)
        states_nnets.append(states_nnet)
        masks_nnets.append(masks_nnet)
        astar_labels.append(astar_label)
        astar_dis_labels.append(astar_dis_label)
        astar_seg_embs_not_pads = astar_seg_embs_not_pads+astar_seg_embs_not_pad
        astar_seg_days_not_pads = astar_seg_days_not_pads + seg_days
        astar_seg_times_not_pads = astar_seg_times_not_pads + seg_times
        astar_routes_lens = astar_routes_lens+astar_routes_len
        queries = queries+train_list_set
    data = {}
    data["states_nnets"] = states_nnets
    data["masks_nnets"] = masks_nnets
    data["astar_labels"] = astar_labels
    data["astar_dis_labels"] = astar_dis_labels
    data["astar_seg_embs_not_pads"] = astar_seg_embs_not_pads
    data["astar_seg_days_not_pads"] = astar_seg_days_not_pads
    data["astar_seg_times_not_pads"] = astar_seg_times_not_pads
    data["astar_routes_lens"] = astar_routes_lens
    data["queries"] = queries

    with open(dataset_dir,"wb") as f:
        pickle.dump(data,f)
    print("Done")        


if __name__ == '__main__':
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)
    for item in ["test","val","train"]:
    # for item in ["train"]:
        args_dict["mode"] = item
        main(args_dict)
