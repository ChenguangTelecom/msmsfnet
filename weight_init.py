import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        if m.weight.data.shape==torch.Size([1, 5, 1, 1]):
            torch.nn.init.constant_(m.weight, 0.2)
        else:
            torch.nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            m.bias.data.zero_()
