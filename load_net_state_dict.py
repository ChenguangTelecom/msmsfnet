import torch
import os

def load_net_state_dict(net, args):
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print('load pretrained model {}'.format(args.pretrained))
            net.load_state_dict(torch.load(args.pretrained)['net'], strict=False)
        else:
            raise ValueError('checkpoint {} not found'.format(args.pretrained))

    if args.pretrained_imagenet:
        if os.path.isfile(args.pretrained_imagenet):
            print('loading pretrained imagenet model {}'.format(args.pretrained_imagenet))
            net.load_state_dict(torch.load(args.pretrained_imagenet)['state_dict'], strict=False)
        else:
            raise ValueError('pretrained imagenet model not found {}'.format(args.pretrained_imagenet))

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print('loading checkpoint {}'.format(args.checkpoint))
            net.load_state_dict(torch.load(args.checkpoint)['net'])
        else:
            raise ValueError('checkpoint not found {}'.format(args.checkpoint))
        
    return True