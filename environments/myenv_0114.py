import random
from random import randrange
from typing import Dict, List, Tuple, Union
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from utils.pytorch_models_0114 import actor_fleet,route_emb_net

from .environment_abstract import Environment, State
from geopy.distance import geodesic
query_random = False

class RFState(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.fea = None
        self.hash = None

    def get_meantime(self,cur_node,next_node,cur_time,pre_time = 0):
        # if cur_node < next_node:
        #     pair = (cur_node,next_node)
        # else:
        #     pair = (next_node,cur_node)  
        pair = (cur_node,next_node) 
        seg_id = self.nid_pair_to_rid[pair]
        passtime = self.passtime_matrix[cur_time, seg_id-1, pre_time]
        return passtime

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)
    


class GridWorld(Environment):

    def __init__(self, time_num, nodes_num):
        self.dtype = np.uint8
        self.dtype2 = np.float32
        self.action_space_size = 12
        self.observation_space_size = 2
        self.pre_slot = 6
        self.world_time = time_num
        self.world_nodes = nodes_num


        # self.segs_num = 486
        # self.passtime_matrix = np.zeros((time_num, segs_num , self.pre_slot))
        # self.reward_matrix = np.zeros((time_num, nodes_num))
        self.state_matrix = np.zeros((time_num, nodes_num))

        self.episode_reward = 0
        self.time_norm = 0

    # def setQuery(self):
    #     self.position = [self.query[0], self.query[1]]
    #     self.start_time = self.query[0]
    #     self.origin = self.query[1]
    #     self.destination = self.query[2]
    #     self.route = [self.query[1]]

    def setStateTransition(self, state_trans,nid_pair_to_rid,node_connected_list):
        #输入一个dict 每个state对应一个状态空间 相当于来邻接矩阵的作用
        self.state_trans = state_trans
        self.nid_pair_to_rid = nid_pair_to_rid
        self.node_connected_list = node_connected_list

    def setPassTimeMatrix(self, passtime_matrix):
        # if(passtime_matrix.shape != self.passtime_matrix.shape):
        #     raise ValueError('The shape of the two matrices must be the same.') 

        #对事件矩阵归一化
        X = passtime_matrix
        # self.time_norm = (X.max() - X.min() )/10.0
        # self.time_norm = X.mean()
        # self.time_norm = 100
        # X_nor = X / self.time_norm
        self.passtime_matrix = passtime_matrix

        # self.passtime_matrix = passtime_matrix
    def set_fea_data(self, mean_traffic, seg_embs, node_adj_pad, node_segs_pad):
        self.mean_traffic = mean_traffic
        self.seg_embs = seg_embs
        self.node_adj_pad = node_adj_pad
        self.node_segs_pad = node_segs_pad
    
    def get_seg_id(self,cur_node,next_node):
        # if cur_node < next_node:
        #     pair = (cur_node,next_node)
        # else:
        #     pair = (next_node,cur_node) 
        pair = (cur_node,next_node)  
        seg_id = self.nid_pair_to_rid[pair]
        return seg_id

    def get_passtime(self,cur_node,next_node,cur_time,pre_time = 0):
        # if cur_node < next_node:
        #     pair = (cur_node,next_node)
        # else:
        #     pair = (next_node,cur_node)   
        # seg_id = self.nid_pair_to_rid[pair]
        seg_id = self.get_seg_id(cur_node,next_node)
        passtime = self.passtime_matrix[cur_time, seg_id-1, pre_time]
        return passtime

    def next_state(self, states: List[RFState], action: int) -> Tuple[List[RFState], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)

        num_states: int = len(states)
        tc = []
        states_next = []
        cur_segs = states_np[:,1]
        cur_times = states_np[:,0]
        dess = states_np[:,2]

        for i in range(num_states):
            cur_seg = cur_segs[i]
            cur_time = cur_times[i]
            des = dess[i]
            # next_state_space = self.state_trans[cur_seg]
            # passtime = self.passtime_matrix[cur_time, cur_seg-1]
            # tc.append(passtime) 
            # next_seg = next_state_space[action]
            # next_time = cur_time + round(passtime/300)
            # if next_time>=self.world_time:
            #     next_time = self.world_time - 1
            # states_next.append(RFState(np.array([next_time,next_seg,des])))

            cur_node = cur_seg
            next_state_space = self.state_trans[cur_seg]
            next_node = next_state_space[action]
            # if cur_node < next_node:
            #     pair = (cur_node,next_node)
            # else:
            #     pair = (next_node,cur_node)
            # seg_id = self.nid_pair_to_rid[pair]
            # passtime = self.passtime_matrix[cur_time, seg_id-1]
            passtime = self.get_passtime(cur_node,next_node,cur_time)
            tc.append(passtime) 
            
            next_time = cur_time + int(passtime*self.time_norm/300)
            if next_time>=self.world_time:
                next_time = self.world_time - 1
            states_next.append(RFState(np.array([next_time,next_node,des])))

    
        return states_next,tc

    def prev_state(self, states: List[RFState], action: int) -> List[RFState]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[RFState], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[RFState] = [RFState(self.goal_colors.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[RFState]) -> np.ndarray:
        #已经重写完毕！
        states_np = np.stack([state.colors for state in states], axis=0)#将1000个状态类中的状态向量取出，按列堆积，拼接成1000*54的状态数组
        is_equal = (states_np[:,1] == states_np[:,2])#返回一个1000维的向量，元素是True 或者 False

        return is_equal

    def state_to_nnet_input(self, states: List[RFState]) -> List[np.ndarray]:
        #已经重写完毕！
        states_np = np.stack([state.colors for state in states], axis=0).astype(self.dtype2)
        fea_np = np.stack([self.get_fea(state.colors) for state in states], axis=0)
        states_np = np.hstack([states_np,fea_np])


        representation_np: np.ndarray = states_np.astype(self.dtype2)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def mask_to_nnet_input(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        #已经重写完毕！
        masks_np = np.stack(masks, axis=0)
        mask_np: np.ndarray = masks_np.astype(self.dtype)
        representation: List[np.ndarray] = [mask_np]

        return representation

    def get_fea(self,colors):

        mf = self.mean_traffic[int(colors[0])]/100
        #这个是当前节点和终点，不是一个路段，但是可以写一个文件，可以查出节点所在的所有路段，以及所有节点
        # seg_id = self.get_seg_id(int(colors[1]),int(colors[2]))
        # seg_emb = self.seg_embs[int(colors[0])][seg_id]

        anodes = np.array(self.node_adj_pad[int(colors[1])]) #编号从1开始的0是pad
        asegs = self.node_segs_pad[int(colors[1])] #编号从1开始的0是pad
        as_emb = self.seg_embs[int(colors[0])][asegs].flatten() #这里的seg_embs(514, 487, 16) 得到5*16
        
        self.fea = np.hstack([np.array([mf,mf]),as_emb])
        return self.fea
 

    def get_num_moves(self) -> int:
        return self.action_space_size

    def get_nnet_model(self) -> nn.Module:
        # state_dim: int = (self.cube_len ** 2) * 6
        # nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)
        nnet = nn.ModuleList([actor_fleet(out_dim = self.action_space_size,nodes_num = self.world_nodes,time_num = self.world_time),route_emb_net(self.action_space_size)])
        # nnet = actor_fleet(out_dim = self.action_space_size,nodes_num = self.world_nodes,time_num = self.world_time)
        return nnet
  
    def generate_states(self, num_states: int) -> Tuple[List[RFState], List[np.ndarray]]:
        #已经重新定义！
        #随机状态：随机当前时间，随机当前路段，随机终点
        #如果到达不了怎么办  例如23点出发？ 可以先设置 随机生成0-12时出发，但是 或者说就不会回传到相应的状态，所以没有影响？
        #A*会设置最大搜索长度，如果超过此长度就没有解
        # remove_list = [229, 312, 413, 414, 415, 416, 417, 418, 443]
        # nodes_list : List[int] =  list(range(1,self.world_nodes+1))
        # des_list : List[int] =  list(range(1,4))
        # for item in remove_list:
        #     nodes_list.remove(item)

        
        nodes_list : List[int] =  self.node_connected_list
        des_list : List[int] =   self.node_connected_list
        cur_time = np.random.randint(0, self.world_time, num_states)
        des = np.random.choice(des_list,num_states)
        cur_seg = np.random.choice(nodes_list,num_states)
        # start_time = np.random.randint(0, self.world_time-5, 1).item()
        
        # des = np.hstack([np.random.choice(des_list,1)]*num_states)
        # cur_seg = np.hstack([nodes_list]*5)[:num_states]
        # cur_time = np.repeat(np.array(range(start_time,start_time+5)), len(nodes_list))[:num_states]

        # # cur_seg = np.random.choice(nodes_list,num_states)

        is_same = (cur_seg == des) #目标时全变成False
        while np.any(is_same):
            idx: np.ndarray = np.where(is_same)[0]
            replace_array = np.random.choice(nodes_list,len(idx))
            cur_seg[idx] = replace_array
            is_same = (cur_seg == des)
        states_np = np.hstack([cur_time.reshape(-1,1),cur_seg.reshape(-1,1),des.reshape(-1,1)]) 
        states: List[RFState] = [RFState(x) for x in list(states_np)]   
        mask_l = self.get_mask(cur_seg)
        return states, mask_l

    def generate_test_states(self, queries: int) -> Tuple[List[RFState], List[np.ndarray]]:

        states_np = np.array(queries)
        cur_seg = states_np[:,1]
        states: List[RFState] = [RFState(x) for x in list(states_np)]   
        mask_l = self.get_mask(cur_seg)
        return states, mask_l

    def get_mask(self,cur_segs):
        mask_l = []
        for cur_seg in cur_segs:
            try:
                next_state_space = self.state_trans[cur_seg]
            except:
                print(cur_seg)
                print(self.state_trans)
                next_state_space = self.state_trans[cur_seg]
            action_num = len(next_state_space) 
            mask = np.zeros(self.action_space_size)
            mask[:action_num] = 1 #新动作空间的mask    
            mask_l.append(mask)   
        # mask_np = np.vstack(mask_l) 
        return mask_l

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray], List[List[np.ndarray]]]:
        # assert self.fixed_actions, "Environments without fixed actions must implement their own method"
        #已经重新定义！
        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]
        masks_exp: List[List[np.ndarray]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        cur_segs = states_np[:,1]
        cur_times = states_np[:,0]
        dess = states_np[:,2]
        for i in range(len(cur_segs)):
            cur_seg = cur_segs[i]
            cur_time = cur_times[i]
            des = dess[i]

            try:
                # next_state_space = self.state_trans[cur_seg]
                # action_num = len(next_state_space) 
                # passtime = self.passtime_matrix[cur_time, cur_seg-1]   

                cur_node = cur_seg
                next_state_space = self.state_trans[cur_node]
                action_num = len(next_state_space) 
                for action in range(self.action_space_size):
                    if action < action_num:
                        next_node = next_state_space[action]
                        # if cur_node < next_node:
                        #     pair = (cur_node,next_node)
                        # else:
                        #     pair = (next_node,cur_node)
                        # seg_id = self.nid_pair_to_rid[pair]
                        # passtime = self.passtime_matrix[cur_time, seg_id-1]    
                        passtime = self.get_passtime(cur_node,next_node,cur_time)                          
                        tc[i,action] = passtime
                    else:
                        INF = 10e6
                        tc[i,action] = INF
            except:
                print("ERROR! cur_seg = ", cur_seg)
                next_state_space = []
                action_num = 0
                INF = 10e6
                passtime = INF
                tc[i,:] = passtime 

            if (tc[i]<0).any():
                print("<0!!")
            # tc[i,:] = passtime    

            next_segs = next_state_space + (self.action_space_size - action_num)*[0]
            next_time = cur_time + int(passtime//300)
            if next_time>=self.world_time:
                next_time = self.world_time - 1

            for idx in range(len(next_segs)):
                s_e = np.array([next_time,next_segs[idx],des])
                states_exp[i].append(RFState(s_e))
                if next_segs[idx] != 0:
                    next_state_space_e = self.state_trans[next_segs[idx]]
                    action_num_e = len(next_state_space_e) 
                    mask = np.zeros(self.action_space_size)
                    mask[:action_num_e] = 1 #新动作空间的mask    
                else:
                    mask = np.zeros(self.action_space_size)
                masks_exp[i].append(mask)

        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        # masks_np_exp = np.vstack(masks_exp).reshape(-1,self.action_space_size)
        return states_exp, tc_l, masks_exp


