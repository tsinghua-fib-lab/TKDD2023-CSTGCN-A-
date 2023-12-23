
import pickle
from heapq import heappop, heappush
import copy
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
Infinite = float('inf')
import torch


class AStar():
    def __init__(self,data,state_tran,nodeid_pair_to_roadid=None):
        self.data = data
        self.stateTran = state_tran
        self.nodeid_pair_to_roadid = nodeid_pair_to_roadid

    class SearchNode:
        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def heuristic_cost_estimate(self, current, goal):
        # gps_c = self.gpsList[current-1]
        # gps_g = self.gpsList[goal-1]
        # h_cost = np.linalg.norm(gps_g-gps_c)*self.gps_norm
        # return h_cost
        return 0

    def distance_between(self, start_time, gscore, n1, n2):
        # now_time = start_time + int(gscore/(60*5))
        # if now_time>=287:
        #     now_time = 287
        pre_time = int(gscore//300)
        # if pre_time>5:
        #     pre_time = 5
        pair = (n1,n2)
        seg_id = self.nodeid_pair_to_roadid[pair]
        now_time = (start_time+pre_time)%self.data.shape[0]
        return self.data[now_time,seg_id - 1]
        # now_time = (start_time+pre_time)%self.data.shape[0]
        # return self.data[now_time,n1 - 1]

    def neighbors(self, node):
        return self.stateTran[node]

    def is_goal_reached(self, current, goal):
        return current == goal

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return current == goal

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def reconstruct_path_time(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                slot_index = int(current.gscore//300)
                yield slot_index
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start_time, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start], 0
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath), current.fscore
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(start_time,current.gscore,current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None
    def astar_time(self, start_time, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start], 0
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath),self.reconstruct_path_time(current, reversePath), current.fscore
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(start_time,current.gscore,current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

class AStar_dis():
    def __init__(self,data,state_tran,nodeid_pair_to_roadid=None):
        self.data = data
        self.stateTran = state_tran
        self.max = max(self.data.values())
        self.nodeid_pair_to_roadid = nodeid_pair_to_roadid

    class SearchNode:
        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def heuristic_cost_estimate(self, current, goal):
        # gps_c = self.gpsList[current-1]
        # gps_g = self.gpsList[goal-1]
        # h_cost = np.linalg.norm(gps_g-gps_c)*self.gps_norm
        # return h_cost
        return 0


    def distance_between(self, start_time, gscore, n1, n2):
        pair = (n1,n2)
        seg_id = self.nodeid_pair_to_roadid[pair]
        try:
            dis = self.data[n1 - 1]
        except:
            dis = self.max # 因为对通行时间有补全操作，即某个路段如果所有时间片都没有通行时间，则用路网的平均代替。
        return dis

    def neighbors(self, node):
        return self.stateTran[node]

    def is_goal_reached(self, current, goal):
        return current == goal

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return current == goal

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start_time, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start], 0
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath), current.fscore
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(start_time,current.gscore,current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None


class AStar_static():
    def __init__(self,data,state_tran,nodeid_pair_to_roadid=None):
        self.data = data
        self.stateTran = state_tran
        self.nodeid_pair_to_roadid = nodeid_pair_to_roadid

    class SearchNode:
        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def heuristic_cost_estimate(self, current, goal):
        # gps_c = self.gpsList[current-1]
        # gps_g = self.gpsList[goal-1]
        # h_cost = np.linalg.norm(gps_g-gps_c)*self.gps_norm
        # return h_cost
        return 0


    def distance_between(self, start_time, gscore, n1, n2):
        pair = (n1,n2)
        seg_id = self.nodeid_pair_to_roadid[pair]
        return self.data[start_time,seg_id - 1]


    def neighbors(self, node):
        return self.stateTran[node]

    def is_goal_reached(self, current, goal):
        return current == goal

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return current == goal

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start_time, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start], 0
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath), current.fscore
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(start_time,current.gscore,current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None


class AStar_label():
    def __init__(self,data_pre,data,state_tran,nodeid_pair_to_roadid):
        self.data_pre = data_pre
        self.data = data
        self.stateTran = state_tran
        self.nodeid_pair_to_roadid = nodeid_pair_to_roadid

    class SearchNode:
        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def heuristic_cost_estimate(self,start_time, start, goal):
        findpath = AStar(self.data,self.stateTran)
        reversePath = False
        _, travel_time =findpath.astar(start_time, start, goal,reversePath)
        return travel_time

    def distance_between(self, start_time, gscore, n1, n2):
        pre_time = int(gscore//300)
        if pre_time>5:
            pre_time = 5
        pair = (n1,n2)
        seg_id = self.nodeid_pair_to_roadid[pair]
        now_time = (start_time+pre_time)%self.data.shape[0]
        return self.data_pre[now_time,seg_id - 1,pre_time]
        # return self.data[now_time,n1 - 1]
    
    def neighbors(self, node):
        return self.stateTran[node]

    def is_goal_reached(self, current, goal):
        return current == goal

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return current == goal

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def reconstruct_path_time(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                slot_index = int(current.gscore//300)
                yield slot_index
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start_time, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start], 0
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start_time, start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath), current.fscore
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(start_time,current.gscore,current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                now_time = int(start_time+tentative_gscore//300)
                if now_time>287:
                    now_time = 287
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(now_time, neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

def basic_test(data,state_tran,test_queries,nodeid_pair_to_roadid):

    findpath = AStar(data,state_tran,nodeid_pair_to_roadid)

    time_list = []
    paths_lists = []
    for query in test_queries:
        start_time, start, goal = query
        reversePath = False
        path, travel_time =findpath.astar(start_time, start, goal,reversePath)

        #visualize    
        path_list = []
        for id in path:
            path_list.append(id)
        # print("start_time:",start_time,"origin:",start,"destination:",goal)
        # print("path:",path_list)
        # print("Travel time:",int(travel_time//60),"min",travel_time%60,"s")
        time_list.append(travel_time)
        paths_lists.append(path_list)

    results = dict()
    results["paths_list"] = paths_lists
    results["time_list"] = time_list
    

    return paths_lists,time_list,results

def basic_test_static(data,state_tran,test_queries,nid_to_rid):

    findpath = AStar_static(data,state_tran,nid_to_rid)

    time_list = []
    paths_lists = []
    for query in test_queries:
        start_time, start, goal = query
        reversePath = False
        path, travel_time =findpath.astar(start_time, start, goal,reversePath)
        #visualize    
        path_list = []
        for id in path:
            path_list.append(id)
        time_list.append(travel_time)
        paths_lists.append(path_list)

    results = dict()
    results["paths_list"] = paths_lists
    results["time_list"] = time_list
    return results    

def basic_test_dis(data,state_tran,test_queries):

    findpath = AStar_dis(data,state_tran)

    time_list = []
    paths_lists = []
    for query in test_queries:
        start_time, start, goal = query
        reversePath = False
        path, travel_time =findpath.astar(start_time, start, goal,reversePath)
        #visualize    
        path_list = []
        for id in path:
            path_list.append(id)
        time_list.append(travel_time)
        paths_lists.append(path_list)

    results = dict()
    results["paths_list"] = paths_lists
    results["time_list"] = time_list
    return results   

def basic_test_label(data_pre,data,state_tran,nodeid_pair_to_roadid,test_queries):

    findpath = AStar_label(data_pre,data,state_tran,nodeid_pair_to_roadid)

    time_list = []
    paths_lists = []
    for query in test_queries:
        start_time, start, goal = query
        reversePath = False
        path, travel_time =findpath.astar(start_time, start, goal,reversePath)

        #visualize    
        path_list = []
        for id in path:
            path_list.append(id)
        # print("start_time:",start_time,"origin:",start,"destination:",goal)
        # print("path:",path_list)
        # print("Travel time:",int(travel_time//60),"min",travel_time%60,"s")
        time_list.append(travel_time)
        paths_lists.append(path_list)

    results = dict()
    results["paths_list"] = paths_lists
    results["time_list"] = time_list
    

    return results    