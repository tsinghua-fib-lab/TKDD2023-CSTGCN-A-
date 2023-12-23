
import pickle
from heapq import heappop, heappush

import numpy as np

Infinite = float('inf')


class AStar():
    def __init__(self,data,state_tran,nodeid_pair_to_roadid,gps_norm):
        self.gps_norm = gps_norm
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
        pre_time = int(gscore//900)
        if pre_time>5:
            pre_time = 5
        pair = (n1,n2)
        seg_id = self.nodeid_pair_to_roadid[pair]
        # return self.data[now_time,seg_id - 1]
        return self.data[start_time,seg_id - 1,pre_time]





        # return self.data[100,n1 - 1]


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
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

class AStar2():
    def __init__(self,data,state_tran,nodeid_pair_to_roadid,gps_norm):
        self.gps_norm = gps_norm
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
        gps_c = self.gpsList[current-1]
        gps_g = self.gpsList[goal-1]
        h_cost = np.linalg.norm(gps_g-gps_c)*self.gps_norm
        return h_cost


    def distance_between(self, start_time, gscore, n1, n2):
        # now_time = start_time + int(gscore/(60*5))
        # if now_time>=287:
        #     now_time = 287
        pre_time = int(gscore//900)
        if pre_time>5:
            pre_time = 5

        pair = (n1,n2)
        seg_id = self.nodeid_pair_to_roadid[pair]
        # return self.data[now_time,seg_id - 1]
        return self.data[start_time,seg_id - 1,pre_time]





        # return self.data[100,n1 - 1]


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
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None


def basic_test(data,state_tran,nodeid_pair_to_roadid,test_queries):

    gps_norm = 1000
    findpath = AStar(data,state_tran,nodeid_pair_to_roadid,gps_norm)

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

def basic_test2(data,state_tran,nodeid_pair_to_roadid,test_queries):

    gps_norm = 1000
    findpath = AStar2(data,state_tran,nodeid_pair_to_roadid,gps_norm)

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

def basic_label(data,state_tran,nodeid_pair_to_roadid,test_queries):

    gps_norm = 1000
    findpath = AStar(data,state_tran,nodeid_pair_to_roadid,gps_norm)

    time_list = []
    paths_lists = []
    for query in test_queries:
        start_time, start, goal = query
        reversePath = False
        path, travel_time =findpath.astar(start_time, start, goal,reversePath)

        time_list.append(travel_time)

    return np.array(time_list).reshape(-1,1)

if __name__ == "__main__":
    #prepare data
    dic = "MYRL/basic/"
    dic2 = "MYRL/data_peking_pred/"
    # with open(dic+"gps_list.pkl","rb") as f:
    #     gps_l = pickle.load(f)

    data = np.load(dic+"data_one_day.npy")[:,:,0]


    with open(dic2+"node_adj.pkl","rb") as f:
        state_tran = pickle.load(f)
    with open(dic2+"nodeid_pair_to_roadid.pkl","rb") as f:
        nodeid_pair_to_roadid = pickle.load(f)
    with open(dic2+"roadid_length.pkl","rb") as f:
        roadid_length = pickle.load(f)

    gps_norm = 1000
    findpath = AStar(data,state_tran,nodeid_pair_to_roadid,gps_norm)

    #
    start_time = 10
    start = 1
    goal = 513
    reversePath = False
    path, travel_time =findpath.astar(start_time, start, goal,reversePath)

    #visualize    
    path_list = []
    for id in path:
        path_list.append(id)
    print("start_time:",start_time,"origin:",start,"destination:",goal)
    print("path:",path_list)
    print("Travel time:",int(travel_time//60),"min",travel_time%60,"s")

    #**h()函数是通过gps对的距离计算出来的，最远的两个点的距离是0.1左右，乘以一个因子gps_norm后作为h()函数。
    # 但是这个因子对搜索结果是有影响的 例如取10 100 1000,quest =（10,10,20）,gps_norm = 10,1000 的最快路径要大于 gps_norm = 100