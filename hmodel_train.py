
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"

import pickle
import time
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import environments.myenv_a as myenv
import numpy as np

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
torch.cuda._initialized = True
import torch.nn as nn

from environments.environment_abstract import Environment
from search_methods.astar_a import astar_test
from search_methods.astar_nh import astar_test_nh
from utils import data_utils
from utils import nnet_utils_dataset_a as nnet_utils
from utils.mae_dataset_a import DataLoader,DataLoader_Edit,StandardScaler
from utils.main_fun import *
import warnings
from utils.earlystopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
warnings.simplefilter("ignore")  
import copy
import json
ttc = time.time()
def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # hyper-parameters
    parser.add_argument('--time_num', type=int,default=288, help="")
    parser.add_argument('--nodes_num', type=int,default=1053, help="")#6796
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")#0。0001
    parser.add_argument('--lr_d', type=float, default=0.9999, help="Learning rate decay for every iteration. "
                                                                      "Learning rate is decayed according to: "
                                                                      "lr * (lr_d ^ itr)")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--nnet_name', type=str, default = "hmodel", help="Name of neural network")
    parser.add_argument('--save_dir', type=str, default="garage_traj_QT5", help="Director to which to save model")
    parser.add_argument('--if_training', type=bool, default=True, help="if training with road,if use if_train will change the state of net(train() or eval()")
    parser.add_argument('--if_save_args', type=bool, default= True, help="if save args/ if test only then do not save")
    parser.add_argument('--test_queries_path', type=str, default="./data_hmodel/data_QT_pred/test_queries.pkl", help="test_queries_dir")
    parser.add_argument('--use_nodeid', type=bool, default= True, help="if the net use node id")
    parser.add_argument('--loss_cat', type=str, default= "MAE", help="use which loss MAE or MAPE,default MAE")
    parser.add_argument('--w', type=float, default= 0.9, help="weight w")
    parser.add_argument('--if_use_mf', type=bool, default= False, help="if add the mean travel time of whole net to query feature")
    parser.add_argument('--if_add_anode', type=bool, default= False, help="if add the adjacent node embedding to query feature")
    parser.add_argument('--AutoWeightedLoss', type=bool, default= False, help="if use AutoWeightedLoss")
    parser.add_argument('--dataset_source', type=str, default="./data_hmodel/data_QT/data_notbaseline/", help="")
    parser.add_argument('--dataset_source2', type=str, default="./data_hmodel/data_QT_pred/", help="")
    parser.add_argument('--dataset_roadid_length', type=str, default="./data_hmodel/data_QT_pred/roadid_length.pkl", help="")
    parser.add_argument('--data_dir', type=str, default="./dataset_traj_QT/", help="")    
    # parse arguments
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)

    # make save directory
    model_dir: str = "%s/%s/" % (args_dict['save_dir'], args_dict['nnet_name'])
    args_dict['curr_dir'] = "%s%s/" % (model_dir, 'current')

    if not os.path.exists(args_dict['curr_dir']):
        os.makedirs(args_dict['curr_dir'])

    args_dict["output_save_loc"] = "%s/output.txt" % args_dict['save_dir']

    # save args
    if args_dict['if_save_args']:
        args_save_loc = "%s/args.pkl" % model_dir
        args_save_json = "%s/args.json" % model_dir
        print("Saving arguments to %s" % args_save_loc)
        with open(args_save_loc, "wb") as f:
            pickle.dump(args, f, protocol=-1)

        with open(args_save_json, 'w') as f:
            json.dump(args_dict, f, indent=4)
    print("Batch size: %i" % args_dict['batch_size'])

    return args_dict

def load_nnet(nnet_dir: str, env:Environment) -> Tuple[nn.Module, int, int]:
    nnet_file: str = "%s/model_state_dict.pt" % nnet_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_nnet_model())
    else:
        nnet: nn.Module = env.get_nnet_model()
    return nnet

def load_test_nnet(nnet_dir: str, env:Environment) -> Tuple[nn.Module, int, int]:
    nnet_file: str = "%s/model_state_dict.pt" % nnet_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_nnet_model())
    else:
        print("No model found in %s" % nnet_dir)
    return nnet

def test_dataset(args_dict,env,device,dataloader_test,scaler,scalar_dis):

    nnet: nn.Module
    nnet = load_test_nnet(args_dict['curr_dir'],env)
    nnet.to(device)

    nnet.if_training = args_dict["if_training"]
    nnet.loss_cat = args_dict["loss_cat"]
    nnet.AutoWeightedLoss = args_dict["AutoWeightedLoss"]
    for _ ,(states_nnet, astar_label,astar_dis_label, astar_seg_embs_pad, astar_routes_len) in enumerate(dataloader_test.get_iterator()):
        last_loss_test,mape_test,mae_test,mean_preds_test,mean_labels_test,std_preds_test,std_labels_test = nnet_utils.train_nnet(nnet, states_nnet,args_dict, astar_label,astar_dis_label, device,
                                            astar_seg_embs_pad,astar_routes_len,scaler,scalar_dis,if_train=False)
        # print("Test main loss is %f " % (last_loss_1_test))
        # print("Test Auxiliary loss is %f " % (last_loss_2_test))
        print("Test loss is %f " % (last_loss_test))
        print("Test mape is %f " % (mape_test))
        print("Test mae is %f " % (mae_test))
        print("Test mean_preds is %f " % (mean_preds_test))
        print("Test mean_labels is %f " % (mean_labels_test))
        print("Test std_preds is %f " % (std_preds_test))
        print("Test std_labels is %f " % (std_labels_test))

def test(args_dict, env, data_time_pre, data_time_static, data_time_label, state_tran, writer,scalar,scalar_dis):

    i=0
    
    weight = 1.0
    num_states = 32768 #batch size of heuristic
    batch_size = 1

    with open(args_dict["test_queries_path"],"rb") as f:
        test_queries_dict = pickle.load(f)

    env.setPassTimeMatrix(data_time_pre)
    for t in ["des1","des2","des3"]:
    # for t in ["des1"]:
        print("Calculating ",t,"......")
        time_test_1 = time.time()

        results_file = "%s/%s" % (args_dict['save_dir'], 'r_'+t+'_astar_info.pkl')
        results_file_p = "%s/%s" % (args_dict['save_dir'], 'result_'+t+'.pkl')
        results_file_p_nh = "%s/%s" % (args_dict['save_dir'], 'result_'+t+'_nh.pkl')
        test_queries = test_queries_dict[t]

        env.set_mode("test")
        #Astar
        weight_t = 1.5 # larger->more accurate but slower
        print("\nStartting Astar Test .....")

        start_time = time.time()
        found_rate,paths_list,ts, results,results_our = astar_test(args_dict, weight_t, batch_size, num_states, env,scalar, test_queries)
        start_time2 = time.time()
        print("Astar Test time: %.2f" % (start_time2 - start_time))

        print("\nStartting Astar Test without hn......")
        start_time3 = time.time()
        found_rate,paths_list,ts, results_nh,results_our_nh = astar_test_nh(args_dict, weight, batch_size, num_states, env,scalar, test_queries)
        print("Astar Test without hn time: %.2f" % (time.time() - start_time3))
        #basic
        print("\nStartting Basic Test without hn......")
        start_time2_2 = time.time()
        results_basic_file = "%s/%s" % (args_dict['save_dir'], 'result_basic_'+t+'.pkl')
        results_static_file = "%s/%s" % (args_dict['save_dir'], 'result_static_'+t+'.pkl')
        env.setPassTimeMatrix(data_time_label)
        found_rate,paths_list,ts, results_nh,results_basic = astar_test_nh(args_dict, weight, batch_size, num_states, env,scalar, test_queries)
        print("Basic Test without hn time: %.2f" % (time.time() - start_time2_2))

        time_our = time_cal(test_queries,results_our['paths_list'],data_time_label,env.nid_pair_to_rid)
        time_basic = time_cal(test_queries,results_basic['paths_list'],data_time_label,env.nid_pair_to_rid)
        score1,score2 = cal_sacore(time_our,time_basic)
        print("Dataset",t,"Score 1:",score1,"Score 2:",score2)


        start_time4 = time.time()
        env.setPassTimeMatrix(data_time_static)
        found_rate,paths_list,ts, results_nh,results_basic_static = astar_test_nh(args_dict, weight, batch_size, num_states, env,scalar, test_queries)
        print("Static Basic Test time: %.2f" % (time.time() - start_time4))

        
        time_nh = time_cal(test_queries,results_our_nh['paths_list'],data_time_label,env.nid_pair_to_rid)
        time_staic = time_cal(test_queries,results_basic_static['paths_list'],data_time_label,env.nid_pair_to_rid)

        if_save = False
        if if_save:
            pickle.dump(results, open(results_file, "wb"), protocol=-1)
            pickle.dump(results_our, open(results_file_p, "wb"), protocol=-1)
            pickle.dump(results_our_nh, open(results_file_p_nh, "wb"), protocol=-1)
            pickle.dump(results_basic, open(results_basic_file, "wb"), protocol=-1)
            pickle.dump(results_basic_static, open(results_static_file, "wb"), protocol=-1)

        print("Dataset",t,"Score 1:",score1,"Score 2:",score2)
        score1_nh,score2_nh = cal_sacore(time_nh,time_basic)
        print("Dataset",t,"Score 1_nh:",score1_nh,"Score 2_nh:",score2_nh)

        score1_staic,score2_staic = cal_sacore(time_staic,time_basic)
        print("Dataset",t,"Score 1_staic:",score1_staic,"Score 2_staic:",score2_staic)

        time_test_2 = time.time()
        print("Test Time of ",t," : ",time_test_2-time_test_1)



        if t == "des1":
            writer.add_scalars('score',{'score1_des1': score1,'score2_des1':score2} , i)
        elif t == "des2":
            writer.add_scalars('score',{'score1_des2': score1,'score2_des2':score2} , i)
        elif t == "des3":
            writer.add_scalars('score',{'score1_des3': score1,'score2_des3':score2} , i)

def train(args_dict, device, dataloader, dataloader_val, dataloader_test, env, scalar, scalar_dis, writer):  
    # load nnet
    nnet: nn.Module
    nnet = load_nnet(args_dict['curr_dir'],env)
    nnet.to(device)

    nnet.if_training = args_dict["if_training"]
    nnet.loss_cat = args_dict["loss_cat"]
    nnet.AutoWeightedLoss = args_dict["AutoWeightedLoss"]

    early_stopping = EarlyStopping(save_path = args_dict['curr_dir'], patience=100)
    for i in range(1000):
        print("i:",i)
        dataloader.shuffle()
        LSTT = 5
        if i>=LSTT:
            nnet.LNT = False
        for _ ,(states_nnet, astar_label,astar_dis_label, astar_seg_embs_pad, astar_routes_len) in enumerate(dataloader.get_iterator()):

            last_loss,mape,mae,mean_preds,mean_labels,std_preds,std_labels = nnet_utils.train_nnet(nnet, states_nnet,args_dict, astar_label,astar_dis_label, device,
                                            astar_seg_embs_pad,astar_routes_len,scalar,scalar_dis,if_train=True)

        for _ ,(states_nnet, astar_label,astar_dis_label, astar_seg_embs_pad, astar_routes_len) in enumerate(dataloader_val.get_iterator()):
        
            last_loss_val,mape_val,mae_val,mean_preds_val,mean_labels_val,std_preds_val,std_labels_val = nnet_utils.train_nnet(nnet, states_nnet,args_dict, astar_label,astar_dis_label, device,
                                                astar_seg_embs_pad,astar_routes_len,scalar,scalar_dis,if_train=False)
        for _ ,(states_nnet, astar_label,astar_dis_label, astar_seg_embs_pad, astar_routes_len) in enumerate(dataloader_test.get_iterator()):
        
            last_loss_test,mape_test,mae_test,mean_preds_test,mean_labels_test,std_preds_test,std_labels_test = nnet_utils.train_nnet(nnet, states_nnet,args_dict, astar_label,astar_dis_label, device,
                                                astar_seg_embs_pad,astar_routes_len,scalar,scalar_dis,if_train=False)

        writer.add_scalars('loss', {"train":last_loss,"val":last_loss_val,"test":last_loss_test}, i)
        writer.add_scalars('mape', {"train":mape,"val":mape_val,"test":mape_test}, i)
        writer.add_scalars('mae', {"train":mae,"val":mae_val,"test":mae_test}, i)
        writer.add_scalars('mean_preds', {"train":mean_preds,"val":mean_preds_val,"test":mean_preds_test}, i)
        writer.add_scalars('mean_labels', {"train":mean_labels,"val":mean_labels_val,"test":mean_labels_test}, i)
        writer.add_scalars('std_preds', {"train":std_preds,"val":std_preds_val,"test":std_preds_test}, i)
        writer.add_scalars('std_labels', {"train":std_labels,"val":std_labels_val,"test":std_labels_test}, i)

        # save nnet
        torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % args_dict['curr_dir'])

        if i>=LSTT:
            early_stopping(last_loss_val, nnet)
            if early_stopping.early_stop:
                print("Early stopping")
                break 
        # clear cuda memory
        torch.cuda.empty_cache()
        # print("Auxiliaryloss was %f / Mainloss is %f" % (last_loss_2,last_loss_1))
        print("Last loss was %f " % (last_loss))


def pad_dict_with_zeros(input_dict):
    output_dict = {}
    max_length = max(len(v) for v in input_dict.values())

    for key, value in input_dict.items():
        if len(value) < max_length:
            output_dict[key] = value + [0] * (max_length - len(value))
        else:
            output_dict[key] = value
    output_dict[0] = [0] * max_length
    return output_dict,max_length


def main():
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")
    for key in args_dict:
        print("%s: %s" % (key, args_dict[key]))

    # # get device
    # on_gpu: bool
    # device: torch.device
    # device, devices, on_gpu = nnet_utils.get_device()
    ###
    print("Device:")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # device = torch.device("cuda:%i" % DEVICECUDA)
    device = torch.device("cuda") 
    on_gpu = True
    print("device: %s, on_gpu: %s" % (device, on_gpu))

    # print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    #load data
    data_train = np.load(args_dict["dataset_source"]+"val.npz")['y'][:,0,:,:]
    segs_num = data_train.shape[1]
    with open(args_dict["dataset_source2"]+"node_adj_c.pkl","rb") as f:
        state_tran = pickle.load(f)
    seg_embs = np.load(args_dict["dataset_source2"]+"TestSeg_emb.npy")
    seg_embs_train = np.load(args_dict["dataset_source2"]+"TestSeg_emb.npy")
    if seg_embs.shape[1] == segs_num:
        seg_embs = np.concatenate([np.zeros([seg_embs.shape[0],1,seg_embs.shape[2]]),seg_embs],axis=1)
    if seg_embs_train.shape[1] == segs_num:
        seg_embs_train = np.concatenate([np.zeros([seg_embs_train.shape[0],1,seg_embs_train.shape[2]]),seg_embs_train],axis=1)

    node_adj_pad,max_adj_num = pad_dict_with_zeros(state_tran)
    
    with open(args_dict["dataset_source2"]+"node_segs_c.pkl","rb") as f:
        node_segs = pickle.load(f)

    node_connected_list = np.load(args_dict["dataset_source2"]+"node_connected_list.npy").tolist()

    with open(args_dict["dataset_source2"]+"node_gps_dict.pkl","rb") as f:
        link_gps_start = pickle.load(f)
    for key in sorted(link_gps_start.keys()):
        if link_gps_start[key] == []:
            link_gps_start[key] = link_gps_start[key-1]

    #environment
    with open(args_dict["dataset_roadid_length"],"rb") as f:
        dis_dict = pickle.load(f) 
    min_dis = np.mean(list(dis_dict.values()))
    # min_gps = np.mean(list(link_gps_start.values()),axis=0).tolist()
    link_num = data_train.shape[1]
    for i in range(1,link_num+1):
        if i not in dis_dict.keys():
            dis_dict[i] = min_dis
    
    with open(args_dict["dataset_source2"]+"nodeid_pair_to_roadid.pkl","rb") as f:
        np_to_rid = pickle.load(f)


    env = myenv.GridWorld(args_dict["time_num"], args_dict["nodes_num"])
    env.setStateTransition(state_tran,node_connected_list)
    env.setNtoR(np_to_rid)    
    env.setDistance(dis_dict)
    # env.set_fea_data(mean_traffic, seg_embs, node_adj_pad)
    env.set_fea_data(seg_embs, node_segs,link_gps_start)
    env.setPassTimeMatrix(data_train[:,:,0])   
    env.setRoadinfo(data_train)

    env.use_nodeid = args_dict["use_nodeid"]
    env.if_use_mf = args_dict["if_use_mf"]
    env.if_add_anode = args_dict["if_add_anode"]
    env.max_adj_num = max_adj_num


    if args_dict["if_training"]:
        with open(args_dict["data_dir"]+"/train.pkl","rb") as f:
            data  = pickle.load(f)
    
        with open(args_dict["data_dir"]+"/val.pkl","rb") as f:
            data_val  = pickle.load(f)

        dataloader = DataLoader(data, 1024*30, device)
        scalar,scalar_dis = dataloader.get_scaler()
        scalar_save_loc = "%s/scalar.pkl" % args_dict['save_dir']
        with open(scalar_save_loc, "wb") as f:
            pickle.dump((scalar.mean,scalar.std,scalar_dis.mean,scalar_dis.std),f)
        
        dataloader_val = DataLoader(data_val, 1024*30, device, scalar,scalar_dis)

    else:
        scalar_save_loc = "%s/scalar.pkl" % args_dict['save_dir']
        with open(scalar_save_loc, "rb") as f:
            mean,std,mean_dis,std_dis = pickle.load(f)
        scalar = StandardScaler(mean,std)
        scalar_dis = StandardScaler(mean_dis,std_dis)

    with open(args_dict["data_dir"]+"/test.pkl","rb") as f:
        data_test  = pickle.load(f)    

    dataloader_test = DataLoader(data_test, 1024*30, device, scalar,scalar_dis)



    writer = SummaryWriter(args_dict['save_dir']+"/log")
    if args_dict["if_training"]:
        train(args_dict, device, dataloader, dataloader_val, dataloader_test, env, scalar,scalar_dis, writer)

    model_dir: str = "%s/%s/" % (args_dict['save_dir'], args_dict['nnet_name'])
    args_save_loc = "%sargs.pkl" % model_dir
    print("Loading arguments from %s" % args_save_loc)
    with open(args_save_loc, "rb") as f:
        args_load = pickle.load(f)
    args_load_dict: Dict[str, Any] = vars(args_load)
    env.use_nodeid = args_load_dict["use_nodeid"]
    env.if_use_mf = args_load_dict["if_use_mf"]
    env.if_add_anode = args_load_dict["if_add_anode"]

    test_dataset(args_dict,env,device,dataloader_test,scalar,scalar_dis)

    data_time_pre = np.load(args_dict["dataset_source2"]+"TestTime_pre.npy")[...,0]
    data_time_pre = np.absolute(data_time_pre)
    data_time_static = np.load(args_dict["dataset_source"]+"test.npz")['x'][:,0,:,0]
    data_time_label = np.load(args_dict["dataset_source"]+"test.npz")['y'][:,0,:,0]
    test(args_dict, env, data_time_pre,data_time_static, data_time_label, state_tran, writer,scalar,scalar_dis)

    print("Done")        

if __name__ == '__main__':
    main()
