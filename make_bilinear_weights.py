import numpy as np
import torch

def make_bilinear_weights(size, num_channels):
    factor=(size+1)//2
    if size%2==1:
        center=factor-1.0
    else:
        center=factor-0.5

    og=np.ogrid[:size, :size]
    filt=(1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    filt=torch.from_numpy(filt)
    w=torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad=False
    for i in range(num_channels):
        for j in range(num_channels):
            if i==j:
                w[i,j]=filt

    return w
