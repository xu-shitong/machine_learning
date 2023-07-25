import torch
import torch.nn as nn
import os

import os,sys,humanize,psutil,GPUtil

# Define function
def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

import torch.distributed as dist
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, batch_size=32, pin_memory=False):
    X = torch.randn(1024, 2)
    Y = torch.randn(1024, 1)
    dataset = torch.utils.data.TensorDataset(X, Y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

from torch.nn.parallel import DistributedDataParallel as DDP
def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    torch.cuda.set_device(rank)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = nn.Sequential(
        nn.Linear(2, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1)
    )
    model = model.to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss_fn = nn.L1Loss()
    for epoch in range(1):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        
        for step, x in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(x[0].cuda())
            label = x[1].cuda()
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

    torch.distributed.barrier() 
    print("rank: ", rank)
    mem_report()

import torch.multiprocessing as mp
if __name__ == '__main__':
    # suppose we have 3 gpus
    mp.spawn(
        main,
        args=(torch.cuda.device_count(),),
        nprocs=torch.cuda.device_count()
    )
