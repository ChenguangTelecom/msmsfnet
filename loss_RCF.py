import torch
import torch.nn.functional as F

def Cross_entropy_loss(prediction, label):
    mask=label.clone()
    num_positive=torch.sum((mask==1).float()).float()
    num_negative=torch.sum((mask==0).float()).float()
    num_tot=num_positive+num_negative
    mask[mask==1]=1.0*num_negative/num_tot
    mask[mask==0]=1.1*num_positive/num_tot
    mask[mask==2]=0
    loss_v=F.binary_cross_entropy(prediction, label, weight=mask, reduction='none')
    return torch.sum(loss_v)