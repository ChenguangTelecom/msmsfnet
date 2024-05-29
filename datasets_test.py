from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class TestDataset(Dataset):
    def __init__(self, dataset_dir='./data', mean_pixel_value=[104.00698793, 116.66876762, 122.67891434]):
        self.dataset_dir=dataset_dir
        self.mean_pixel_value=mean_pixel_value
        self.imagelist=os.listdir(self.dataset_dir)
        self.images_name=[]
        for imagename_full in self.imagelist:
            imagename, imageext=os.path.splitext(imagename_full)
            self.images_name.append(imagename)
            
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, idx):
        image_path=os.path.join(self.dataset_dir, self.imagelist[idx])
        image=cv2.imread(image_path)
        image=self.transform(image)
        return image
    
    def transform(self, image):
        image=image.astype(np.float32)
        image=image-np.array(self.mean_pixel_value)
        image=np.transpose(image, (2, 0, 1))
        image=image.astype(np.float32)
        return image