import torch
import os
from tqdm import tqdm
from utils import AverageMeter
from loss_RCF import Cross_entropy_loss

def train(train_loader, net, opt, lr_schd, epoch, save_dir, args, device):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    net.train()
    opt.zero_grad()
    batch_loss_meter=AverageMeter()
    counter=0
    for batch_index, (images, edges) in enumerate(tqdm(train_loader)):
        images, edges=images.to(device), edges.to(device)
        preds_list=net(images)
        loss=torch.zeros(1).to(device)
        for o in preds_list:
            loss=loss+Cross_entropy_loss(o, edges)
        counter+=1
        loss/=args.itersize
        loss.backward()
        if counter==args.itersize:
            opt.step()
            opt.zero_grad()
            counter=0
            batch_loss_meter.update(loss.item())
        if batch_index%args.print_freq==args.print_freq-1:
            print('Epoch: {}/{}, batch: {}/{}, Epoch_average_batch_loss: {}, lr: {}'.format(epoch, args.max_epoch, batch_index, len(train_loader), batch_loss_meter.avg, lr_schd.get_last_lr()))

    lr_schd.step()
    return batch_loss_meter.avg
