from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, dataset_dir='./data', trainlist='bsds_pascal_train_pair.lst', mean_pixel_value=[104.00698793, 116.66876762, 122.67891434]):
        self.dataset_dir=dataset_dir
        self.mean_pixel_value=mean_pixel_value
        self.listpath=os.path.join(self.dataset_dir, trainlist)
        f=open(self.listpath, 'r')
        lines=f.readlines()
        f.close()
        lines=[line.strip() for line in lines]
        pairs=[line.split() for line in lines]
        self.images_path=[pair[0] for pair in pairs]
        self.edges_path=[pair[1] for pair in pairs]
        #self.images_name=[]
        #for img_path in self.images_path:
            #img_folder, img_name=os.path.split(img_path)
            #img_name_o, img_ext=os.path.splitext(img_name)
            #self.images_name.append(img_name_o)
            
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        edge_path=os.path.join(self.dataset_dir, self.edges_path[idx])
        image_path=os.path.join(self.dataset_dir, self.images_path[idx])
        edge=cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        image=cv2.imread(image_path)
        image, edge=self.transform(image, edge)
        return image, edge
    
    def transform(self, image, edge):
        edge=edge.astype(np.float32)
        image=image.astype(np.float32)
        edge=edge[np.newaxis, :, :]
        edge[edge==0]=0
        edge[np.logical_and(edge>0, edge<127.5)]=2
        edge[edge>=127.5]=1
        edge=edge.astype(np.float32)
        image=image-np.array(self.mean_pixel_value)
        image=np.transpose(image, (2, 0, 1))
        image=image.astype(np.float32)
        edge=edge.astype(np.float32)
        return image, edge



