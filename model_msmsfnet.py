import torch
import torch.nn as nn
import torch.nn.functional as F

from make_bilinear_weights import make_bilinear_weights
from crop_image import crop_image
class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kx1, ky1, kx2,  ky2, stride=1, bias=True, with_bn=False, with_relu=False):
        super(TwoConv, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=(kx1, ky1), stride=stride, padding=(int(kx1/2), int(ky1/2)), bias=bias)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=(kx2, ky2), stride=1, padding=(int(kx2/2), int(ky2/2)), bias=bias)
        self.with_bn=with_bn
        self.with_relu=with_relu
        if with_bn:
            self.bn1=nn.BatchNorm2d(out_channels)
            self.bn2=nn.BatchNorm2d(out_channels)

        if with_relu:
            self.relu=nn.ReLU(inplace=True)


    def forward(self, x):
        if self.with_bn:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.conv2(x)
            x=self.bn2(x)
        else:
            x=self.conv1(x)
            x=self.conv2(x)

        if self.with_relu:
            x=self.relu(x)

        return x

class MsmsfBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, with_bn=False):
        super(MsmsfBlock, self).__init__()
        out_dim=int(out_channels/4)

        self.conv1_1=TwoConv(in_channels, out_dim, 1, 3, 3, 1, bias=bias, with_bn=with_bn)
        self.conv1_2=TwoConv(out_dim, out_dim, 1, 3, 3, 1, bias=bias, with_bn=with_bn)
        
        self.conv2_1=TwoConv(in_channels, out_dim, 1, 3, 5, 1, bias=bias, with_bn=with_bn)
        self.conv2_2=TwoConv(out_dim, out_dim, 1, 5, 3, 1, bias=bias, with_bn=with_bn)

        self.fuse12=TwoConv(out_dim*2, out_dim*2, 1, 3, 3, 1, bias=bias, with_bn=with_bn)

        self.conv3_1=TwoConv(in_channels, out_dim, 1, 3, 7, 1, bias=bias, with_bn=with_bn)
        self.conv3_2=TwoConv(out_dim, out_dim, 1, 7, 3, 1, bias=bias, with_bn=with_bn)

        self.conv4_1=TwoConv(in_channels, out_dim, 1, 3, 9, 1, bias=bias, with_bn=with_bn)
        self.conv4_2=TwoConv(out_dim, out_dim, 1, 9, 3, 1, bias=bias, with_bn=with_bn)
        
        self.fuse34=TwoConv(out_dim*2, out_dim*2, 1, 3, 3, 1, bias=bias, with_bn=with_bn)

        self.fuse1234=TwoConv(out_channels, out_channels, 1, 3, 3, 1, bias=bias, with_bn=with_bn, with_relu=True)


    def forward(self, x):
        x1=self.conv1_1(x)
        x1=self.conv1_2(x1)

        x2=self.conv2_1(x)
        x2=self.conv2_2(x2)

        x3=self.conv3_1(x)
        x3=self.conv3_2(x3)

        x4=self.conv4_1(x)
        x4=self.conv4_2(x4)

        x1=torch.cat([x1, x2], 1)
        x3=torch.cat([x3, x4], 1)

        x1=self.fuse12(x1)
        x3=self.fuse34(x3)

        x=torch.cat([x1, x3], 1)
        x=self.fuse1234(x)
        return x

class MultiBlock(nn.Module):
    def __init__(self, block, num_layers, in_channels, out_channels, bias=True, with_bn=False):
        super(MultiBlock, self).__init__()

        self.num_layers=num_layers
        for i in range(num_layers):
            setattr(self, 'conv{}'.format(i+1), block(in_channels, out_channels, bias=bias, with_bn=with_bn))
            in_channels=out_channels

    def forward(self, x):
        for i in range(self.num_layers):
            x=getattr(self, 'conv{}'.format(i+1))(x)
            #out.append(x0)

        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.n_stages=3
        self.n_blocks=3
        in_channels=3
        out_channels=128

        for i in range(self.n_stages):
            setattr(self, 'conv{}'.format(i+1), MultiBlock(MsmsfBlock, self.n_blocks, in_channels, out_channels, bias=True, with_bn=False))
            setattr(self, 'score_dsn{}'.format(i+1), nn.Conv2d(out_channels, 1, 3, 1, 1))
            if i>0:
                setattr(self, 'weight_deconv{}'.format(i+1), make_bilinear_weights(2**(i+1), 1))
            in_channels=out_channels
            out_channels*=2

        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse=nn.Conv2d(self.n_stages, 1, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()

    def forward(self, x):
        _,_,h,w=x.shape
        out=[]
        for i in range(self.n_stages):
            x=getattr(self, 'conv{}'.format(i+1))(x)
            #out_all=out_all+x
            #x=x[-1]
            out.append(getattr(self, 'score_dsn{}'.format(i+1))(x))
            if i<self.n_stages-1:
                x=self.maxpool(x)

        for i in range(1, self.n_stages):
            #out_last[i]=getattr(self, 'score_dsn{}'.format(i+1))(out_last[i])
            out[i]=F.conv_transpose2d(out[i], getattr(self, 'weight_deconv{}'.format(i+1)).to(out[i].device), stride=2**i)#[:,:,2**(i-1):-2**(i-1), 2**(i-1):-2**(i-1)]#nn.Upsample(size=img.size()[2:], mode='bilinear', align_corners=True)(out_last[i])
            out[i]=crop_image(out[i], h, w)

            
        x=torch.cat(out, 1)
        x=self.fuse(x)

        out.append(x)

        out=[torch.sigmoid(r) for r in out]
        return out

    
            

        
