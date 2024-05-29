import os
import scipy.io as sio
import torch
from PIL import Image
from os.path import join
from tqdm import tqdm
import cv2
import numpy as np

def create_folder(save_dir, mode='single'):
    png_avg_dir=join(save_dir, mode, 'png_avg')
    os.makedirs(png_avg_dir, exist_ok=True)
    png_fuse_dir=join(save_dir, mode, 'png_fuse')
    os.makedirs(png_fuse_dir, exist_ok=True)
    mat_avg_dir=join(save_dir, mode, 'mat_avg')
    os.makedirs(mat_avg_dir, exist_ok=True)
    mat_fuse_dir=join(save_dir, mode, 'mat_fuse')
    os.makedirs(mat_fuse_dir, exist_ok=True)
    return png_avg_dir, png_fuse_dir, mat_avg_dir, mat_fuse_dir

def single_scale_test(test_loader, net, save_dir, device=torch.device('cuda')):
    os.makedirs(save_dir, exist_ok=True)
    png_avg_dir, png_fuse_dir, mat_avg_dir, mat_fuse_dir=create_folder(save_dir)
    net.eval()
    with torch.no_grad():
        for batch_index, images in enumerate(tqdm(test_loader)):
            images=images.to(device)
            preds_list=net(images)
            avg=torch.mean(torch.stack(preds_list).detach(), dim=0).cpu().numpy()[0,0]
            fuse=preds_list[-1].detach().cpu().numpy()[0,0]
            imagename=test_loader.dataset.images_name[batch_index]
            sio.savemat(join(mat_avg_dir, imagename+'.mat'), {'result': avg})
            sio.savemat(join(mat_fuse_dir, imagename+'.mat'), {'result': fuse})
            Image.fromarray((avg*255).astype(np.uint8)).save(join(png_avg_dir, imagename+'.png'))
            Image.fromarray((fuse*255).astype(np.uint8)).save(join(png_fuse_dir, imagename+'.png'))

def multi_scale_test(test_loader, net, save_dir, scales=[0.5, 1.5], device=torch.device('cuda')):
    os.makedirs(save_dir, exist_ok=True)
    _,png_fuse_dir, _, mat_fuse_dir=create_folder(save_dir, mode='multi')
    net.eval()
    with torch.no_grad():
        for batch_index, images in enumerate(tqdm(test_loader)):
            _,_,h,w=images.shape
            images=images.to(device)
            preds_list=net(images)
            msfuse=preds_list[-1].detach().cpu().numpy()[0,0]
            if len(scales)>0:
                in_=images[0].cpu().numpy().transpose((1,2,0))
                for scale in scales:
                    im_=cv2.resize(in_, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    im_=im_.transpose((2, 0, 1))
                    im_=torch.unsqueeze(torch.from_numpy(im_), 0).to(device)
                    preds_list=net(im_)
                    fuse=preds_list[-1].detach().cpu().numpy()[0, 0]
                    fuse=cv2.resize(fuse, (w, h), interpolation=cv2.INTER_LINEAR)
                    msfuse+=fuse

            fuse=msfuse/(len(scales)+1)
            imagename=test_loader.dataset.images_name[batch_index]
            sio.savemat(join(mat_fuse_dir, imagename+'.mat'), {'result': fuse})
            Image.fromarray((fuse*255).astype(np.uint8)).save(join(png_fuse_dir, imagename+'.png'))



        


