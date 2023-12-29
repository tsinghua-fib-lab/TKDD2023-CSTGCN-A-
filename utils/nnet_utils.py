from typing import List, Tuple, Optional
import numpy as np
import os
import torch
from torch import nn
from environments.environment_abstract import Environment, State
from collections import OrderedDict
import re
from random import shuffle
from torch import Tensor

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from torch.multiprocessing import Queue, get_context
from utils.mae_dataset_a import masked_mae,masked_mape
import time


# training
def states_nnet_to_pytorch_input(states_nnet: List[np.ndarray], device) -> List[Tensor]:
    states_nnet_tensors = []
    for tensor_np in states_nnet:
        tensor = torch.tensor(tensor_np, device=device)
        states_nnet_tensors.append(tensor)

    return states_nnet_tensors

def masks_nnet_to_pytorch_input(masks_nnet: List[np.ndarray], device) -> List[Tensor]:
    masks_nnet_tensors = []
    for tensor_np in masks_nnet:
        tensor = torch.tensor(tensor_np, device=device)
        masks_nnet_tensors.append(tensor)
    return masks_nnet_tensors


def make_batches(states_nnet: List[np.ndarray], masks_nnet: List[np.ndarray], outputs: np.ndarray,astar_label: np.ndarray,
                 batch_size: int) -> List[Tuple[List[np.ndarray], np.ndarray]]:
    num_examples = outputs.shape[0]
    rand_idxs = np.random.choice(num_examples, num_examples, replace=False)
    outputs = outputs.astype(np.float32)
    astar_label = astar_label.astype(np.float32)

    start_idx = 0
    batches = []
    while (start_idx + batch_size) <= num_examples:
        end_idx = start_idx + batch_size

        idxs = rand_idxs[start_idx:end_idx]

        inputs_batch = [x[idxs] for x in states_nnet]
        inputs_batch2 = [x[idxs] for x in masks_nnet]
        outputs_batch = outputs[idxs]
        astar_label_batch = astar_label[idxs]

        batches.append((inputs_batch, inputs_batch2, outputs_batch,astar_label_batch))

        start_idx = end_idx

    return batches


def train_nnet(nnet: nn.Module, states_nnet: List[np.ndarray], masks_nnet: List[np.ndarray], outputs: np.ndarray,astar_label: np.ndarray, device: torch.device,
               batch_size: int, num_itrs: int, train_itr: int, lr: float, lr_d: float, display: bool = True) -> float:
    
    # optimization
    display_itrs = 100
    criterion = nn.MSELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # initialize status tracking
    start_time = time.time()

    # train network
    batches: List[Tuple[List, List, np.ndarray]] = make_batches(states_nnet, masks_nnet, outputs,astar_label, batch_size)

    nnet.train()
    max_itrs: int = train_itr + num_itrs

    last_loss: float = np.inf
    batch_idx: int = 0
    mape_list = []
    mae_list = []
    preds_list = []
    labels_list = []
    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        inputs_batch, inputs_batch2, targets_batch_np,astar_label_batch_np = batches[batch_idx] # states mask target
        targets_batch_np = targets_batch_np.astype(np.float32)
        astar_label_batch_np = astar_label_batch_np.astype(np.float32)
        # send data to device
        states_batch: List[Tensor] = states_nnet_to_pytorch_input(inputs_batch, device)
        masks_batch: List[Tensor] = masks_nnet_to_pytorch_input(inputs_batch2, device)

        targets_batch: Tensor = torch.tensor(targets_batch_np, device=device)
        astar_label_batch: Tensor = torch.tensor(astar_label_batch_np, device=device)

        # forward
        nnet_outputs_batch_: Tensor = nnet(*states_batch,*masks_batch)
        nnet_outputs_batch = nnet_outputs_batch_.min(1)[0].reshape(-1, 1)#[batch_size(5000),12]->[5000,1]
        # cost
        nnet_cost_to_go = nnet_outputs_batch[:, 0]
        target_cost_to_go = targets_batch[:, 0]
        astar_label_to_go = astar_label_batch[:, 0]
        mape,preds,labels = masked_mape(nnet_cost_to_go,astar_label_to_go,0.0)
        mape = mape.item()
        preds = preds.item()
        labels = labels.item()
        mae = masked_mae(nnet_cost_to_go,astar_label_to_go,0.0).item()
        mape_list.append(mape)
        mae_list.append(mae)
        preds_list.append(preds)
        labels_list.append(labels)
        loss = criterion(nnet_cost_to_go, target_cost_to_go)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        last_loss = loss.item()
        # display progress
        if (train_itr % display_itrs == 0) and display:
            print("Itr: %i, lr: %.2E, loss: %.2f, targ_ctg: %.2f, nnet_ctg: %.2f, "
                  "Time: %.2f" % (
                      train_itr, lr_itr, loss.item(), target_cost_to_go.mean().item(), nnet_cost_to_go.mean().item(),
                      time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

        batch_idx += 1
        if batch_idx >= len(batches):
            shuffle(batches)
            batch_idx = 0
    
    mean_mape = np.mean(mape_list)
    mean_mae = np.mean(mae_list)
    mean_preds = np.mean(preds_list)
    mean_labels = np.mean(labels_list)
    print("MAE:",mean_mae,"MAPE:",mean_mape)
    print("Preds:",mean_preds,"s Labels:",mean_labels,"s")
    return last_loss,mean_mape,mean_mae,mean_preds,mean_labels


# pytorch device
def get_device() -> Tuple[torch.device, List[int], bool]:
    device: torch.device = torch.device("cpu")
    devices: List[int] = get_available_gpu_nums()
    on_gpu: bool = False
    if devices and torch.cuda.is_available():
        device = torch.device("cuda:%i" % 0)
        on_gpu = True

    return device, devices, on_gpu
 

# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


# heuristic
def get_heuristic_fn(nnet: nn.Module, device: torch.device, env: Environment, clip_zero: bool = False,
                     batch_size: Optional[int] = None):
    nnet.eval()

    def heuristic_fn(states: List, masks: np.ndarray, is_nnet_format: bool = False) -> np.ndarray:
        cost_to_go: np.ndarray = np.zeros(0)
        if not is_nnet_format:
            num_states: int = len(states)
        else:
            num_states: int = states[0].shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            # convert to nnet input
            if not is_nnet_format:
                states_batch: List = states[start_idx:end_idx]
                masks_batch: List = masks[start_idx:end_idx]
                states_nnet_batch: List[np.ndarray] = env.state_to_nnet_input(states_batch)
                masks_nnet_batch: List[np.ndarray] = env.mask_to_nnet_input(masks_batch)
            else:
                states_nnet_batch = [x[start_idx:end_idx] for x in states]
                masks_nnet_batch = [x[start_idx:end_idx] for x in masks]

            # get nnet output
            states_nnet_batch_tensors = states_nnet_to_pytorch_input(states_nnet_batch, device)
            masks_nnet_batch_tensors = masks_nnet_to_pytorch_input(masks_nnet_batch, device)
            cost_to_go_batch: np.ndarray = nnet(*states_nnet_batch_tensors,*masks_nnet_batch_tensors).min(1)[0].reshape(-1,1).cpu().data.numpy()#10000*54输入

            cost_to_go: np.ndarray = np.concatenate((cost_to_go, cost_to_go_batch[:, 0]), axis=0)

            start_idx: int = end_idx

        assert (cost_to_go.shape[0] == num_states)

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def get_available_gpu_nums() -> List[int]:
    # devices: Optional[str] = os.environ.get('CUDA_VISIBLE_DEVICES')
    # return [int(x) for x in devices.split(',')] if devices else []

    num = torch.cuda.device_count()
    return [int(x) for x in range(num)] if (num !=0) else []
    # return [int(x) for x in range(4)] 


def load_heuristic_fn(nnet_dir: str, device: torch.device, on_gpu: bool, nnet: nn.Module, env: Environment,
                      clip_zero: bool = False, gpu_num: int = -1, batch_size: Optional[int] = None):
    if (gpu_num >= 0) and on_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    model_file = "%s/model_state_dict.pt" % nnet_dir

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn = get_heuristic_fn(nnet, device, env, clip_zero=clip_zero, batch_size=batch_size)

    return heuristic_fn

# parallel training
def heuristic_fn_queue(heuristic_fn_input_queue, heuristic_fn_output_queue, proc_id, env: Environment):
    def heuristic_fn(states,masks):
        states_nnet = env.state_to_nnet_input(states)
        masks_nnet = env.mask_to_nnet_input(masks)
        heuristic_fn_input_queue.put((proc_id, states_nnet, masks_nnet))
        heuristics = heuristic_fn_output_queue.get()

        return heuristics

    return heuristic_fn


def heuristic_fn_runner(heuristic_fn_input_queue: Queue, heuristic_fn_output_queues, nnet_dir: str,
                        device, on_gpu: bool, gpu_num: int, env: Environment, all_zeros: bool,
                        clip_zero: bool, batch_size: Optional[int]):
    heuristic_fn = None
    if not all_zeros:
        heuristic_fn = load_heuristic_fn(nnet_dir, device, on_gpu, env.get_nnet_model(), env, gpu_num=gpu_num,
                                         clip_zero=clip_zero, batch_size=batch_size)#env.get_nnet_model() 一个Resnet网络

    while True:
        proc_id, states_nnet, masks_nnet = heuristic_fn_input_queue.get()
        if proc_id is None:
            break

        if all_zeros:
            heuristics = np.zeros(states_nnet[0].shape[0], dtype=np.float)
        else:
            heuristics = heuristic_fn(states_nnet, masks_nnet, is_nnet_format=True)

        heuristic_fn_output_queues[proc_id].put(heuristics)

    return heuristic_fn


def start_heur_fn_runners(num_procs: int, nnet_dir: str, device, on_gpu: bool, env: Environment,
                          all_zeros: bool = False, clip_zero: bool = False, batch_size: Optional[int] = None):
    ctx = get_context("spawn")

    heuristic_fn_input_queue: ctx.Queue = ctx.Queue()
    heuristic_fn_output_queues: List[ctx.Queue] = []
    for _ in range(num_procs):
        heuristic_fn_output_queue: ctx.Queue = ctx.Queue(1)
        heuristic_fn_output_queues.append(heuristic_fn_output_queue)

    # initialize heuristic procs
    gpu_nums = get_available_gpu_nums() or [-1]

    heur_procs: List[ctx.Process] = []
    for gpu_num in gpu_nums:
        heur_proc = ctx.Process(target=heuristic_fn_runner,
                                args=(heuristic_fn_input_queue, heuristic_fn_output_queues,
                                      nnet_dir, device, on_gpu, gpu_num, env, all_zeros, clip_zero, batch_size))
        heur_proc.daemon = True
        heur_proc.start()
        heur_procs.append(heur_proc)

    return heuristic_fn_input_queue, heuristic_fn_output_queues, heur_procs


def stop_heuristic_fn_runners(heur_procs, heuristic_fn_input_queue):
    for _ in heur_procs:
        heuristic_fn_input_queue.put((None, None, None))

    for heur_proc in heur_procs:
        heur_proc.join()
