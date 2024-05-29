import os
import sys
import torch
import argparse
from datetime import datetime
from torch.optim import lr_scheduler
from os.path import join, isdir, abspath, dirname

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from create_args import create_args
from model_msmsfnet import Network
from prepare_datasets_ddp import prepare_datasets
from utils import Logger, AverageMeter, load_checkpoint, save_checkpoint
from test_single_multi import single_scale_test, multi_scale_test
from create_opt_msmsfnet import create_opt
from load_net_state_dict import load_net_state_dict
from train_loop_ddp import train_loop

parser=argparse.ArgumentParser(description='edge model training.')
args=create_args(parser)
#device=torch.device('cpu' if args.cpu else 'cuda')


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size):
    ddp_setup(rank, world_size)
    current_dir=abspath(dirname(__file__))
    output_dir=join(current_dir, args.output)
    if not isdir(output_dir):
        os.makedirs(output_dir)

    now_str=datetime.now().strftime('%y%m%d%H%M%S')
    log=Logger(join(output_dir, 'log-{}.txt'.format(now_str)))
    sys.stdout=log
    train_loader, test_loader=prepare_datasets(args)
    net=Network()
    load_net_state_dict(net, args)
    net=net.to(rank)
    net=DDP(net, device_ids=[rank])
    #net.apply(weights_init)
    opt=create_opt(net, args)
    lr_schd=lr_scheduler.StepLR(opt, step_size=args.lr_stepsize, gamma=args.lr_gamma)
    #load_net_state_dict(net, args)
    device=torch.device(rank)
    if args.test:
        single_scale_test(test_loader, net, save_dir=join(output_dir, 'test'), device=device)
        multi_scale_test(test_loader, net, save_dir=join(output_dir, 'test'), scales=[0.5, 1.5], device=device)
        
    else:
        train_epoch_losses=train_loop(train_loader, test_loader, net, opt, lr_schd, output_dir, args,device, log, rank)
    destroy_process_group()

if __name__=='__main__':
    world_size=torch.cuda.device_count()

    mp.spawn(main, args=(world_size, ), nprocs=world_size)
