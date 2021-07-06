import os
import sys
import random
import time
import logging
import math
import numpy

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from core.data import partition_dataset, get_test_dataset

class MNISTResNet(ResNet):
    def __init__(self):
        #super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

def log_info(*value):
    if dist.is_initialized() :
        logging.info(f'[Rank-{dist.get_rank()}] {" ".join(map(str, value))}')
    else :
        logging.info(' '.join(map(str, value)))

def average_gradients(model, group = None):
    size = float(dist.get_world_size())
    for param in model.parameters():
        # s = torch.cuda.Stream()
        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
        param.grad.data /= size
        # with torch.cuda.stream(s):
        #     s.wait_stream(torch.cuda.default_stream())

def std_mean_loss(loss):
    cur_losses = torch.Tensor([loss,loss,loss,loss]).cuda()
    cur_losses = list(cur_losses.chunk(4))
    all_losses = list(torch.empty([4], dtype=torch.float32).cuda().chunk(4))

    torch.distributed.all_to_all(all_losses, cur_losses)
    # rank = dist.get_rank()

    std_loss, _ = torch.std_mean(torch.Tensor(all_losses))

    # group = []
    # for idx, l in enumerate(all_losses):
    #     if l > std_loss

    
    return std_loss

def run():
    rank = dist.get_rank()
    seed = 7548
    group =None

    model = MNISTResNet().cuda()
    model = torchvision.models.resnet50().cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr = 0.01)
    
    pre_std_loss = 0
    count = 0
    
   
    for epoch in range(1):
        seed += epoch
        torch.manual_seed(seed)
        train_set, batch_size = partition_dataset(batch_size=256, seed=seed)
        num_batches = math.ceil(len(train_set.dataset) / float(batch_size))

        epoch_loss = 0.0
        start_time = time.time()
        step = 0
        avg_loss = 0.0
        std_loss = 0.0

        reduce_trues = torch.tensor(0.0).cuda()
        reduce_num_dataset_per_step = 0
        #group = dist.new_group([0, 2, 3], backend='nccl')
        for data, target in train_set:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            
            avg_loss = epoch_loss / num_batches
            std_loss = std_mean_loss(avg_loss)
            
            if std_loss > pre_std_loss:
                average_gradients(model, group)
            else:
                count += 1

            #if (step == 48):
            #average_gradients(model, group)

            pre_std_loss = std_loss
            
            optimizer.step()

            # calc aucc
            _, predicted = torch.max(output, 1)
            trues = (target.data.view_as(predicted) == predicted).squeeze().sum()
            dist.all_reduce(trues, op=dist.ReduceOp.SUM, group=group)
            reduce_trues += trues
            length_target = torch.tensor(len(target)).cuda()
            dist.all_reduce(length_target,  op=dist.ReduceOp.SUM)
            reduce_num_dataset_per_step +=  length_target

            step += 1
        log_info('train', 'epoch', epoch, 'step', step, epoch_loss / num_batches, float(reduce_trues / float(reduce_num_dataset_per_step)), count)
        print("epoch", epoch)

        
        with torch.no_grad():
            val_t = torch.tensor(0.0).cuda()
            len_y = 0
            val_set, batch_size = get_test_dataset(batch_size=256, seed=seed)
            epoch_loss = 0.0
            for data, target in val_set:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                epoch_loss += loss.item()

                _, predicted = torch.max(output, 1)
                v_trues = (target.data.view_as(predicted) == predicted).squeeze().sum()
                dist.all_reduce(v_trues, op=dist.ReduceOp.SUM, group=group)
                val_t += v_trues

                length_target = torch.tensor(len(target)).cuda()
                dist.all_reduce(length_target,  op=dist.ReduceOp.SUM)

            log_info('val', epoch_loss / len(val_set.dataset), float(val_t / float(len(val_set.dataset))))
    
    ret = [10]
    torch.distributed.broadcast_object_list(ret, 0)    

    #average_gradients(model)

def init_process(rank):
    #os.environ['MASTER_ADDR'] = '192.168.1.1'
    #os.environ['MASTER_PORT'] = '7548'
    #os.environ['NCCL_DEBUG'] = 'INFO'
    #os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['NCCL_ALGO'] = 'Ring'
    #os.environ["NCCL_BLOCKING_WAIT"] = '1'

    dist.init_process_group(init_method='tcp://192.168.1.1:7548', backend='nccl', rank=rank, world_size=4)
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    
    #torch.distributed.broadcast_object_list([current_time], 0)
    
    logging.basicConfig(
        filename= f'logs/log_{current_time}_{rank}.log',
        level=logging.INFO, 
        format = '%(asctime)s:%(levelname)s:%(message)s'
    )

    #model = Tor.cuda()

    # cpu = torch.device("cpu")

    exp_time = []

    loop = 100

    # for i in range(loop):
    #     if rank == 0 : 
    #         start_time = time.time()
    #         for param in model.parameters():
    #             torch.distributed.isend(param, 1)
    #         exp_time.append(time.time() - start_time)
    #     else:
    #         start_time = time.time()
    #         for param in model.parameters():
    #             torch.distributed.irecv(param, 0)
    #         exp_time.append(time.time() - start_time)

    #     if i % 100 == 0:
    #         log_info(i)
            
    # log_info("1 {0:0.10f}".format(sum(exp_time) / loop))

    # exp_time = []
    # for _ in range(loop):
    #     if rank == 0 : 
    #         start_time = time.time()
    #         for param in model.parameters():
    #             data = param.to(cpu)
    #             torch.distributed.isend(data, 1)
    #             param = data.cuda()
    #         #torch.distributed.isend(data, 1)
    #         exp_time.append(time.time() - start_time)
    #     else:
    #         start_time = time.time()
    #         for param in model.parameters():
    #             data = param.to(cpu)
    #             torch.distributed.irecv(data, 0)
    #             param = data.cuda()
    #         exp_time.append(time.time() - start_time)
            
    # log_info("2 {0:0.10f}".format(sum(exp_time) / loop))

    # input = torch.Tensor([rank,rank,rank,rank]).cuda()
    # input = list(input.chunk(4))
    # output = list(torch.empty([4], dtype=torch.float32).cuda().chunk(4))

    # torch.distributed.all_to_all(output, input)
    # std_loss = torch.std_mean(torch.Tensor(output))

    # print(std_loss)
    start_time = time.time()
    run()
    log_info(time.time() - start_time)


global v

def test():
    print("1",  data)
    

import torch.multiprocessing as mp
#from multiprocessing import shared_memory

if __name__ == "__main__":
    rank = int(sys.argv[1])
    
    # model = MNISTResNet().cuda()
    # cpu = torch.device("cpu")
    # for param in model.parameters():
    #     data = param.to(cpu).share_memory_()

    init_process(rank)
    # v = 10
    # print("0", data)  
    # p = mp.Process(target=test, args=())
    # print("2", data)
    # p.start()
    # print("3", data)
    # p.join()
    # print("4", data)
    