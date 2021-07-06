import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import sys
import random
import math
import logging
from statistics import mean
from functools import reduce

import torchdocker exec -it bsh-server2_0 bash
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

from net import Net

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO, 
    format = '%(asctime)s:%(levelname)s:%(message)s'
)


class MNISTResNet(ResNet):

    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=7548):
        self.data = data
        self.partitions = []
        rng = random
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset():
    dataset = datasets.MNIST('./dataset', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ]))
    size = dist.get_world_size()
    bsz = 1024 * 4 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size = int(bsz), shuffle=True)
    
    return train_set, bsz


def logInfo(*value):
    if dist.is_initialized() :
        logging.info(f'[Rank {dist.get_rank()}] {" ".join(map(str, value))}')
    else :
        logging.info(' '.join(map(str, value)))


def ring_allreduce(epoch, send):
    start_time = time.time()
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = send.clone().cuda()
    recv_buff = send.clone().cuda()
    accum = send.clone().cuda()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        logInfo(epoch, 'Rank ', dist.get_rank(), i, "start")
        if i % 2 == 0:
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv_buff[:]
        else:
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send_buff[:]
        logInfo(epoch, 'Rank ', dist.get_rank(), i, "finish")
        send_req.wait()
    
    send.set_(accum)

    logInfo(epoch, 'Rank ', dist.get_rank(), ', time ', time.time() - start_time)


def allreduce(epoch, data, group = None):
    start_time = time.time()
    dist.all_reduce(data, op=dist.ReduceOp.SUM, group=group)
    #print(epoch, 'Rank ', dist.get_rank(), ', time ', time.time() - start_time)


def broadcast_and_avg(no, model, group = None):
    #state = torch.Tensor([100*dist.get_rank()]).cuda()
    #logInfo('state', state)
    #start_time = time.time()
    # 각 노드의 상태를 확인하고
    # dist.broadcast(state, src = 0)
    # dist.broadcast(state, src = 1)
    # dist.broadcast(state, src = 2)
    # dist.broadcast(state, src = 3)
    #logInfo('update state', state, time.time() - start_time)
    
    # 그룹을 만들고
    start_time = time.time()
    group = dist.new_group([0, 2, 3], backend="nccl")
    #print('create group', time.time() - start_time)

    start_time = time.time()
    average_gradients(no, model, group)
    #print('avg', time.time() - start_time)


def average_gradients(no, model, group = None):
    size = float(dist.get_world_size())
    start_time = time.time()
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
        param.grad.data /= size
    
 
def run(communicate_func):
    torch.manual_seed(7548)
    train_set, bsz = partition_dataset()
    device = torch.device("cuda:{}".format(dist.get_rank()))
    torch.cuda.set_device(dist.get_rank())
    
    #model = Net().to(device)

    model = MNISTResNet().cuda()

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    no = 0 

    for epoch in range(5):
        epoch_loss = 0.0
        start_time = time.time()
        step = 0
        for data, target in train_set:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()

            communicate_func(no, model)

            no += 1
            optimizer.step()

            logInfo('epoch ', epoch, '(step', step ,'): ', epoch_loss / num_batches)
            step+=1


def init_process(rank, size, fn, backend='nccl'):
    start_time = time.time()
    logInfo("========= start init_process =========")
    os.environ['MASTER_ADDR'] = '127.0.0.1'


    os.environ['MASTER_PORT'] = '7548'
#    os.environ['NCCL_DEBUG'] = 'INFO'
#    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
#    os.environ['USE_SYSTEM_NCCL'] = '1'
#    os.environ['NCCL_IB_DISABLE'] = '1'

    #os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    logging.disable(logging.CRITICAL)
    
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    device = torch.device("cuda:{}".format(rank))
    torch.cuda.set_device(rank)
 
    run(fn)
    elaped_time = time.time() - start_time
    logInfo("========= done init_process =========", elaped_time)


def main(run_func):
    #logging.disable(logging.CRITICAL)
    start_time = time.time()
    #logInfo("========= start main =========")
    size = 4
    processes = []
    #mp.set_start_method("spawn")

    # for rank in range(size):
    #     p = mp.Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
    torch.multiprocessing.spawn(init_process, nprocs=size, args=(size, run_func,))

    elaped_time = time.time() - start_time
    logInfo("========= done main =========", elaped_time)
    return elaped_time


if __name__ == "__main__":
    lst_elaped_time = []
    
#    for _ in range(5):
#       lst_elaped_time.append(main(average_gradients))
#    avg1 = reduce(lambda a, b: a+b, lst_elaped_time) / len(lst_elaped_time)
    logInfo("run_allreduce avg time ", avg1)

    lst_elaped_time = []
    for _ in range(5):
        lst_elaped_time.append(main(broadcast_and_avg))
    avg2 = reduce(lambda a, b: a+b, lst_elaped_time) / len(lst_elaped_time)
    logInfo("run_broadcast avg time ", avg2)

    #logInfo('avg1-avg2', avg1 - avg2)
