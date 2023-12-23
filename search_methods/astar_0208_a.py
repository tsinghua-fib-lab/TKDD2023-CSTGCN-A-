import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.utils import assert_all_finite

os.environ["CUDA_VISIBLE_DEVICES"]='7'
# 将包含parent包的路径添加进系统路径
sys.path.append(r"/data/suihongjie/DeepCubeA-master")
import os
import pickle
import sys
import time
from argparse import ArgumentParser
from heapq import heappop, heappush
import numpy as np
import torch
from environments.environment_abstract import Environment, State
from utils import data_utils, env_utils, misc_utils, search_utils
from utils import nnet_utils_0131_dataset_a as nnet_utils



class Node:
    __slots__ = ['state', 'path_cost', 'heuristic', 'cost', 'is_solved', 'parent_move', 'parent', 'transition_costs',
                 'children', 'bellman','out_openset']

    def __init__(self, state: State, path_cost: float, is_solved: bool,
                 parent_move: Optional[int], parent):
        self.state: State = state
        self.path_cost: float = path_cost
        self.heuristic: Optional[float] = None
        self.cost: Optional[float] = None
        self.is_solved: bool = is_solved
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent

        self.transition_costs: List[float] = []
        self.children: List[Node] = []

        self.bellman: float = np.inf
        self.out_openset = True

    def compute_bellman(self, mask):
        if self.is_solved:
            self.bellman = 0.0
        elif len(self.children) == 0:
            self.bellman = self.heuristic
        else:
            if mask.sum() == 0:
                self.bellman = self.heuristic
            else:
                for node_c, tc, mask_c in zip(self.children, self.transition_costs, mask.tolist()):
                    if mask_c == 1:
                        self.bellman = min(self.bellman, tc + node_c.heuristic)
    
    def __lt__(self, b):
        return self.cost < b.cost


OpenSetElem = Tuple[float, int, Node]


class Instance:

    def __init__(self, root_node: Node, root_mask: np.ndarray):
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        # self.closed_dict: Dict[State, float] = dict()
        self.closed_dict = dict()
        self.popped_nodes: List[Node] = []
        self.popped_masks: List[np.ndarray] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = root_node
        self.root_mask: np.ndarray = root_mask
        self.SearchNodes = self.SearchNodeDict()
        self.SearchNodes[root_node.state] = root_node
        self.closed_dict[self.root_node.state] = 0.0
        self.push_to_open([self.root_node],[root_mask])
    
    class SearchNodeDict(dict):

        def __missing__(self,state):
            v = Node(state, path_cost=1e6, is_solved=None, parent_move=None, parent=None)
            self.__setitem__(state, v)
            return v

    def push_to_open(self, nodes: List[Node], masks: List[np.ndarray]):
        for node, mask in zip(nodes, masks):
            # heappush(self.open_set, (node.cost, self.heappush_count, node, mask))
            # self.heappush_count += 1
            if node.out_openset:
                node.out_openset = False
                heappush(self.open_set, (node, mask.tolist()))
            else:
                # print("Replace Open")
                # print([self.open_set[i][0].state.colors for i in range(len(self.open_set))])
                # print(node.state.colors, mask.tolist())
                self.open_set.remove((node, mask.tolist()))
                heappush(self.open_set, (node, mask.tolist()))                


    def pop_from_open(self, num_nodes: int) -> Tuple[List[Node], List[np.ndarray]]:
        num_to_pop: int = min(num_nodes, len(self.open_set))
        popped_nodes = []
        popped_masks = []
        for _ in range(num_to_pop):
            # cost, count, node, mask = heappop(self.open_set)
            node, mask = heappop(self.open_set)
            node.out_openset = True
            popped_nodes.append(node)
            popped_masks.append(np.array(mask))
        # popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]
        self.goal_nodes.extend([node for node in popped_nodes if node.is_solved])
        self.popped_nodes.extend(popped_nodes)
        self.popped_masks.extend(popped_masks)

        return popped_nodes, popped_masks

    def remove_in_closed(self, nodes: List[Node], masks: List[np.ndarray]) -> Tuple[List[Node],List[np.ndarray]]:
        nodes_not_in_closed: List[Node] = []
        masks_not_in_closed: List[Node] = []
        # print("close dict:")
        # print([state.colors for state in self.closed_dict])
        for node, mask in zip(nodes,masks):
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if path_cost_prev is None:
                nodes_not_in_closed.append(node)
                masks_not_in_closed.append(mask)
                # self.closed_dict[node.state] = node.path_cost
            elif path_cost_prev > node.path_cost:
                pass
                # nodes_not_in_closed.append(node)
                # masks_not_in_closed.append(mask)
                # # self.closed_dict[node.state] = node.path_cost

        return nodes_not_in_closed, masks_not_in_closed

    def remove_not_able(self, nodes: List[Node], masks: List[np.ndarray]) -> Tuple[List[Node],List[np.ndarray]]:
        nodes_not_in_closed: List[Node] = []
        masks_not_in_closed: List[Node] = []
        for node, mask in zip(nodes,masks):
            if node.state.colors[1] != 0:
                nodes_not_in_closed.append(node)
                masks_not_in_closed.append(mask)

        return nodes_not_in_closed, masks_not_in_closed

def pop_from_open(instances: List[Instance], batch_size: int) -> Tuple[List[List[Node]], List[List[np.ndarray]]]:
    popped_nodes_all: List[List[Node]] = []
    popped_masks_all: List[List[np.ndarray]] = []
    for instance in instances:
        nodes, masks = instance.pop_from_open(batch_size)
        popped_nodes_all.append(nodes)
        popped_masks_all.append(masks)
    # popped_nodes_all: List[List[Node]] = [instance.pop_from_open(batch_size) for instance in instances]

    return popped_nodes_all, popped_masks_all


def expand_nodes(instances: List[Instance], popped_nodes_all: List[List[Node]], env: Environment):
    # Get children of all nodes at once (for speed)
    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_all)

    if len(popped_nodes_flat) == 0:
        return [[]],[[]]

    states: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[np.ndarray]
    masks_c_by_node: List[List[np.ndarray]]

    states_c_by_node, tcs_np, masks_c_by_node = env.expand(states)

    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]

    # Get is_solved on all states at once (for speed)
    states_c: List[State]

    states_c, split_idxs_c = misc_utils.flatten(states_c_by_node)
    masks_c, _ = misc_utils.flatten(masks_c_by_node)
    
    is_solved_c: List[bool] = list(env.is_solved(states_c))
    is_solved_c_by_node: List[List[bool]] = misc_utils.unflatten(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    parent_path_costs = np.expand_dims(np.array([node.path_cost for node in popped_nodes_flat]), 1)
    path_costs_c: List[float] = (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()

    path_costs_c_by_node: List[List[float]] = misc_utils.unflatten(path_costs_c, split_idxs_c)

    # Reshape lists
    tcs_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(tcs_by_node, split_idxs)
    patch_costs_c_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(path_costs_c_by_node,
                                                                               split_idxs)
    states_c_by_inst_node: List[List[List[State]]] = misc_utils.unflatten(states_c_by_node, split_idxs)
    masks_c_by_inst_node: List[List[List[np.ndarray]]] = misc_utils.unflatten(masks_c_by_node, split_idxs)
    is_solved_c_by_inst_node: List[List[List[bool]]] = misc_utils.unflatten(is_solved_c_by_node, split_idxs)

    # Get child nodes
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    masks_c_by_inst: List[List[np.ndarray]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        masks_c_by_inst.append([])
        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        tcs_by_node: List[List[float]] = tcs_by_inst_node[inst_idx]
        path_costs_c_by_node: List[List[float]] = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node: List[List[State]] = states_c_by_inst_node[inst_idx]
        masks_c_by_node: List[List[State]] = masks_c_by_inst_node[inst_idx]

        is_solved_c_by_node: List[List[bool]] = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        states_c: List[State]
        str_reps_c: List[str]
        for parent_node, tcs_node, path_costs_c, states_c, masks_c, is_solved_c in zip(parent_nodes, tcs_by_node,
                                                                              path_costs_c_by_node, states_c_by_node,masks_c_by_node,
                                                                              is_solved_c_by_node):
            state: State
            for move_idx, (state, mask) in enumerate(zip(states_c,masks_c)):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                node_c = instance.SearchNodes[state]
                if node_c.path_cost>path_cost:
                    node_c.path_cost = path_cost
                    node_c.is_solved = is_solved
                    node_c.parent_move = move_idx
                    node_c.parent = parent_node

                # node_c: Node = Node(state, path_cost, is_solved, move_idx, parent_node)

                nodes_c_by_inst[inst_idx].append(node_c)
                masks_c_by_inst[inst_idx].append(mask)

                parent_node.children.append(node_c)

            parent_node.transition_costs.extend(tcs_node)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])
    # masks_c_by_inst = masks_c_by_node #masks_c_by_inst,_ = misc_utils.flatten(masks_c_by_node)
    return nodes_c_by_inst, masks_c_by_inst


def remove_in_closed(instances: List[Instance], nodes_c_all: List[List[Node]], masks_c_all: List[List[np.ndarray]]) -> Tuple[List[List[Node]],List[List[np.ndarray]]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx], masks_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx], masks_c_all[inst_idx])

    return nodes_c_all, masks_c_all

def remove_not_able(instances: List[Instance], nodes_c_all: List[List[Node]], masks_c_all: List[List[np.ndarray]]) -> Tuple[List[List[Node]],List[List[np.ndarray]]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx], masks_c_all[inst_idx] = instance.remove_not_able(nodes_c_all[inst_idx], masks_c_all[inst_idx])

    return nodes_c_all, masks_c_all

def add_heuristic_and_cost(nodes: List[Node], scaler, heuristic_fn: Callable,
                           weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    # flatten nodes
    nodes: List[Node]

    if len(nodes) == 0:
        return np.zeros(0), np.zeros(0)

    # get heuristic
    states: List[State] = [node.state for node in nodes]

    # compute node cost
    heuristics = heuristic_fn(states,scaler)
    path_costs: np.ndarray = np.array([node.path_cost for node in nodes])
    is_solved: np.ndarray = np.array([node.is_solved for node in nodes])

    costs: np.ndarray = np.array(weights) * path_costs + heuristics * np.logical_not(is_solved)

    # add cost to node
    for node, heuristic, cost in zip(nodes, heuristics, costs):
        node.heuristic = heuristic
        node.cost = cost

    return path_costs, heuristics


def add_to_open(instances: List[Instance], nodes: List[List[Node]], masks: List[List[np.ndarray]]) -> None:
    nodes_inst: List[Node]
    masks_inst: List[np.ndarray]
    instance: Instance
    for instance, nodes_inst, masks_inst in zip(instances, nodes, masks):
        # if len(nodes_inst)>1:
        #     index = np.argmin([node.cost for node in nodes_inst])
        #     instance.push_to_open([nodes_inst[index]], [masks_inst[index]])
        # else:
        instance.push_to_open(nodes_inst, masks_inst)


def get_path(node: Node) -> Tuple[List[State], List[int], float]:
    path: List[State] = []
    moves: List[int] = []

    parent_node: Node = node
    # print("Start gen route!")
    while parent_node.parent is not None:
        path.append(parent_node.state)
        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent
        if len(path)>10000:
            print("error")

    path.append(parent_node.state)
    path = path[::-1]
    moves = moves[::-1]
    # print("Finish gen route!")
    return path, moves, node.path_cost


class AStar:

    def __init__(self, states: List[State],masks: List[np.ndarray], env: Environment,scalar, heuristic_fn: Callable, weights: List[float]):
        self.env: Environment = env
        self.weights: List[float] = weights
        self.step_num: int = 0

        self.timings: Dict[str, float] = {"pop": 0.0, "expand": 0.0, "check": 0.0, "heur": 0.0,
                                          "add": 0.0, "itr": 0.0}

        # compute starting costs
        root_nodes: List[Node] = []
        is_solved_states: np.ndarray = self.env.is_solved(states)
        for state, is_solved in zip(states, is_solved_states):
            root_node: Node = Node(state, 0.0, is_solved, None, None)
            root_nodes.append(root_node)

        add_heuristic_and_cost(root_nodes, scalar, heuristic_fn, self.weights)

        # initialize instances
        self.instances: List[Instance] = []
        for root_node, mask in zip(root_nodes, masks):
            self.instances.append(Instance(root_node, mask))

    def step(self, heuristic_fn: Callable, batch_size: int, scalar, include_solved: bool = False, verbose: bool = True):
        start_time_itr = time.time()
        instances: List[Instance]
        if include_solved:
            instances = self.instances
        else:
            instances = [instance for instance in self.instances if len(instance.goal_nodes) == 0]

        # Pop from open
        start_time = time.time()
        popped_nodes_all, popped_masks_all = pop_from_open(instances, batch_size)

        # for nl in popped_nodes_all:
        #     polist = [node.state.colors for node in nl]
        #     colist = [node.cost for node in nl]
        #     hlist = [node.heuristic for node in nl]
        # print("Pop List: ",polist,"Cost List:",colist,"Heuristic List",hlist)

        # List[List[Node]]
        pop_time = time.time() - start_time

        for inst_idx, instance in enumerate(instances):
            for node in popped_nodes_all[inst_idx]:
                instance.closed_dict[node.state] = node.path_cost

        # Expand nodes
        start_time = time.time()
        nodes_c_all, masks_c_all= expand_nodes(instances, popped_nodes_all, self.env)# 每次扩展新节点就会验证新节点是否到终点了


        # List[List[Node]]  List[List[np.ndarray]] 
        expand_time = time.time() - start_time





        # Check if children are abled
        start_time = time.time()
        nodes_c_all, masks_c_all = remove_not_able(instances, nodes_c_all, masks_c_all )
        check_time = time.time() - start_time
        # polist = []
        # for nl in nodes_c_all:
        #     polist = [node.state.colors for node in nl]
        # print("Expend List: ",polist)

        # Check if children are in closed
        start_time = time.time()
        nodes_c_all, masks_c_all = remove_in_closed(instances, nodes_c_all, masks_c_all )
        check_time = time.time() - start_time
        polist = []
        # for nl in nodes_c_all:
        #     polist = [node.state.colors for node in nl]
        # print("Expend Not in Closed List: ",polist)

        # Get heuristic of children, do heur before check so we can do backup
        start_time = time.time()
        nodes_c_all_flat, _ = misc_utils.flatten(nodes_c_all)
        masks_c_all_flat, _ = misc_utils.flatten(masks_c_all)
        weights, _ = misc_utils.flatten([[weight] * len(nodes_c) for weight, nodes_c in zip(self.weights, nodes_c_all)])
        path_costs, heuristics = add_heuristic_and_cost(nodes_c_all_flat, scalar, heuristic_fn, weights)
        heur_time = time.time() - start_time



        # Add to open
        start_time = time.time()
        add_to_open(instances, nodes_c_all, masks_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr
        # if heuristics.shape[0] <= 0:
        #     return True
        if len(instances) == 0:
            return True
        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = np.min(heuristics)# 计算的是没有去除假节点的heuristics
                min_heur_pc = path_costs[np.argmin(heuristics)]
                max_heur = np.max(heuristics[heuristics<10000])
                max_heur_pc = path_costs[heuristics<10000][np.argmax(heuristics[heuristics<10000])]
                print("Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                    "%.2f(%.2f)/%.2f(%.2f) " % (self.step_num, min_heur, min_heur_pc, max_heur, max_heur_pc))          

            # print("Times - pop: %.2f, expand: %.2f, check: %.2f, heur: %.2f, "
            #       "add: %.2f, itr: %.2f" % (pop_time, expand_time, check_time, heur_time, add_time, itr_time))

        # Update timings
        self.timings['pop'] += pop_time
        self.timings['expand'] += expand_time
        self.timings['check'] += check_time
        self.timings['heur'] += heur_time
        self.timings['add'] += add_time
        self.timings['itr'] += itr_time

        self.step_num += 1

    def has_found_goal(self) -> List[bool]:
        goal_found: List[bool] = [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]

        return goal_found

    def get_goal_nodes(self, inst_idx) -> List[Node]:
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx) -> Node:
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]

        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        return self.instances[inst_idx].num_nodes_generated

    def get_popped_nodes(self) -> List[List[Node]]:
        popped_nodes_all: List[List[Node]] = [instance.popped_nodes for instance in self.instances]
        return popped_nodes_all

    def get_popped_masks(self) -> List[List[np.ndarray]]:
        popped_masks_all: List[List[np.ndarray]] = [instance.popped_masks for instance in self.instances]
        return popped_masks_all

def bwas_t(args_dict, weight, batch_size, env: Environment, scaler, states: List[State], masks: List[np.ndarray]):


    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    if_parallel: bool = False

    # print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn_test(args_dict, device, if_parallel, env.get_nnet_model(),
                                                env, clip_zero=True)

    solns: List[List[int]] = []
    paths: List[List[State]] = []
    times: List = []
    num_nodes_gen: List[int] = []
    not_found_num = 0
    found_num = 0# 下面是一个instance一个instance训练
    weights = [weight]*len(states)

    start_time = time.time()
    num_itrs: int = 0
    astar = AStar(states, masks, env, scaler, heuristic_fn, weights)
    print("Num_instance:",len(astar.instances))
    solved_num = 0
    while not min(astar.has_found_goal()):#所有实例都完成
        if sum(astar.has_found_goal()) > solved_num:
            solved_num = sum(astar.has_found_goal())
            # print("itr_num:",num_itrs)
            # print("Solved Instance Number:",solved_num)
            is_end = astar.step(heuristic_fn, batch_size, scaler,verbose = False)
        else:
            is_end = astar.step(heuristic_fn, batch_size, scaler,verbose = False)
        if is_end:
            not_found_num = not_found_num +1
            break
        else:
            num_itrs += 1

    # print to screen
    timing_str = ", ".join(["%s: %.2f" % (key, val) for key, val in astar.timings.items()])
    print("Times - %s, num_itrs: %i" % (timing_str, num_itrs))
    solncost_list = []
    moves_list = []
    nodes_gen_list = []
    for state_idx in range(len(states)):
        state = states[state_idx]
        if is_end:
            pass
        else:#None
            found_num = found_num + 1
            path: List[State]
            soln: List[int]
            path_cost: float
            num_nodes_gen_idx: int
            goal_node: Node = astar.get_goal_node_smallest_path_cost(state_idx)
            path, soln, path_cost = get_path(goal_node)
            num_nodes_gen_idx: int = astar.get_num_nodes_generated(state_idx)
            solve_time = time.time() - start_time

            # record solution information
            solns.append(soln)
            paths.append(path)
            times.append(solve_time)
            num_nodes_gen.append(num_nodes_gen_idx)

            # check soln
            assert search_utils.is_valid_soln(state, soln, env)
            # print("State: %i, SolnCost: %.2f, # Moves: %i, "
            #     "# Nodes Gen: %s, Time: %.2f" % (state_idx, path_cost, len(soln),
            #                                     format(num_nodes_gen_idx, ","),
            #                                     solve_time))
            solncost_list.append(path_cost)
            moves_list.append(len(soln))
            nodes_gen_list.append(num_nodes_gen_idx)
            
    found_rate = found_num/(not_found_num+found_num)
        # print("not_found_num:",not_found_num,"found_num:",found_num)

    print("SolnCost:",round(np.mean(solncost_list),2),"Moves:",round(np.mean(moves_list),2),"Nodes Gen:",round(np.mean(nodes_gen_list),2))

    return solns, paths, times, num_nodes_gen, found_rate


def astar_test(args_dict, weight, batch_size, num_states: int, env: Environment, scalar, queries = None):

# get data
    if queries == None:
        states: List[State] = []
        states, mask_l = env.generate_states(num_states)
    else:
        states, mask_l = env.generate_test_states(queries)#已经加过1 了  

    # Do GBFS for each back step
    # print("Solving %i states with Astar" % len(states))

    results: Dict[str, Any] = dict()
    results["states"] = states
    solns, paths, times, num_nodes_gen, found_rate = bwas_t(args_dict, weight, batch_size, env, scalar, states, mask_l)

    results["solutions"] = solns
    results["paths"] = paths
    results["times"] = times
    results["num_nodes_generated"] = num_nodes_gen

    ts = []
    paths_list = []
    for path in results["paths"]:
        path_l = []
        tims_p = 0
        route = [s.colors[1] for s in path]
        start_time = path[0].colors[0]
        path_l.append(path[0].colors[1])
        for i in range(len(route)-1):
            # q_time = int(tims_p//300+start_time)
            # t1 = env.get_passtime(route[i],route[i+1],q_time)*100 

            # 基于查询时刻，预测未来半个小时的路况  
            passed_time = int(tims_p//300)
            if passed_time>5:
                passed_time = 5
            t1 = env.get_passtime(route[i],route[i+1],start_time,passed_time)  
            tims_p += t1
            path_l.append(route[i+1]) 
        # for s in path:
        #     q = s.colors
        #     path_l.append(q[1]) 
        #     t1 = env.get_passtime(q[1],q[2],q[0])       
        #     tims_p += t1
        paths_list.append(path_l)
        ts.append(round(tims_p,2))    

    results_p = {}
    results_p["paths_list"] = paths_list
    results_p["time_list"] = ts
    qs = []
    for state in results["states"]:
        query = state.colors
        qs.append(query)
    results_p["queries"] = np.vstack(qs)

    # print("solns, paths, times, num_nodes_gen: ",solns, paths, times, num_nodes_gen)
    print("found_rate:",found_rate)

    return found_rate,paths_list,ts, results, results_p


