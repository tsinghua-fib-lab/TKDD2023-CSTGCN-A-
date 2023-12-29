import numpy as np
import os
import shutil
from typing import Any, Dict, List, Tuple
import torch.nn as nn
import pickle
import copy
from environments.environment_abstract import Environment
import time
def copy_files(src_dir: str, dest_dir: str):
    src_files: List[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)


def time_cal(queries,pathlist,data,nid_to_rid):
    query_time = []
    for query in queries:
        query_time.append(query[0])
    ts = []
    for de_time,pa in zip(query_time,pathlist):
        tims_p = 0
        # for i in range(len(pa)):
        #     rid = pa[i]
        for i in range(len(pa)-1):
            rid = nid_to_rid[(pa[i],pa[i+1])]
            q_time = int(tims_p//300)
            if q_time>5:
                q_time = 5
            now_time = (q_time+de_time)%data.shape[0]
            # t1 = data[de_time,rid-1,q_time]
            t1 = data[now_time,rid-1]
            tims_p += t1
        ts.append(tims_p)    
    return ts

def get_seg_embs(queries,pathlist,nid_to_rid,seg_embs):
    query_time = []
    for query in queries:
        query_time.append(query[0])
    ts = []
    for de_time,pa in zip(query_time,pathlist):
        tims_p = 0
        for i in range(len(pa)):
            # if pa[i]<=pa[i+1]:
            # rid = nid_to_rid[(pa[i],pa[i+1])]
            rid = pa[i]
            # else:
            #     rid = nid_to_rid[(pa[i+1],pa[i])]
            q_time = int(tims_p//300)
            seg_time = de_time+q_time
            if seg_time>287:
                seg_time = 287
            t1 = seg_embs[seg_time,rid-1]
            tims_p += t1
        ts.append(tims_p)    
    return ts

def cal_sacore(ts,time_list):
    score_list = [ts[i]-time_list[i] for i in range(len(ts))]
    # print(ts,time_list)
    count1 = 0
    for i in range(len(score_list)):
        if score_list[i] <= 0 :
            count1 += 1
    score1 = count1/len(score_list)
    score2_list = []
    for i in range(len(score_list)):
        if score_list[i]>=0:
            s2 = score_list[i]/time_list[i]
            score2_list.append(s2)
    score2 = np.mean(score2_list)    
    return score1,score2