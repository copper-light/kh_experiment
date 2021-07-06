import os
import logging
import random
import math
import time

from net import Net

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms

def run(rank, size):
    print('run', rank)
    torch.manual_seed(7548)
    train_set, bsz = partition_dataset()
    device = torch.device("cuda:{}".format(rank))
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    no = 0
    for epoch in range(10):
        epoch_loss = 0.0
        start_time = time.time()

        step = 0
        for data, target in train_set:
            data, target = data.to(device), target.to(device)              
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()

            average_gradients(no, model)
            no += 1
            optimizer.step()

            print('Rank ', dist.get_rank(), ', epoch ', epoch, '(step', step ,'): ', epoch_loss / num_batches)
            step+=1

    # if rank == '0':
    #     logging.info('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

    print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches, 'tiem : ', time.time() - start_time)

def broadcast(tensor, size):
    start_time = time.time()
    if rank == 0:
        tensor = torch.zeros(3)
    else:
        tensor = torch.ones(3)
    group = dist.new_group([0, 2, 3])
    dist.broadcast(tensor, src = 0, group= group)
    print('Rank ', dist.get_rank(), tensor, time.time() - start_time)

def average_gradients(no, model):
    size = float(dist.get_world_size())
    start_time = time.time()
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
    if dist.get_rank() == 0:
        print('Rank ', dist.get_rank(), ', time ', time.time() - start_time)
        

def init_process(rank, size, fn, backend='nccl'):
    print('init_process', rank)
    
    # os.environ.setdefault("NCCL_DEBUG", "INFO")
    # os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")


    # os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '7548'
    dist.init_process_group(backend, rank=rank, world_size=size)
    if rank == 0:
        logging.basicConfig(filename='./logs/rank_0.log', encoding='utf-8', level=logging.DEBUG)
    fn(rank, size)


def partition_dataset():
    dataset = datasets.MNIST('./dataset', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ]))
    size = dist.get_world_size()
    bsz = 1024 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size = int(bsz), shuffle=True)
    
    return train_set, bsz


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


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

