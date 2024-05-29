import torch
from organize_opt_lr_msmsfnet import organize_opt_lr

def create_opt(net, args):
    if args.organize_lr:
        print('organize opt lr')
        opt=organize_opt_lr(net, args, args.use_sgd)
    else:
        if args.SGD:
            print('use SGD optimizer without organizing lr')
            opt=torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            print('use adam optimizer without organizing lr')
            opt=torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
    return opt