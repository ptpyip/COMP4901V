import os
import torch

import numpy as np

from enum import Enum
from errno import ENOENT
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import RandomAffine

import PA1.src.tools.dense_transforms as dense_transforms

TENSOR_DTYPE = torch.float

def load_data(dataset_dir:str, transforms, num_workers=0, batch_size=32, **kwargs):
    dataset = VehicleClassificationDataset(dataset_dir, transforms=transforms)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseCityscapesDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


class VehicleClassificationDataset(Dataset):
    VEHICLE_TYPES = ['Bicycle', 'Car', 'Taxi', 'Bus', 'Truck', 'Van']
    def __init__(self, dataset_dir, transforms):
        """
        Your code here
        Hint: load your data from provided dataset (VehicleClassificationDataset) to train your designed model
        """
        self.dataset_dir = dataset_dir
        self.data_paths = []
        self.labels = []   
        self.transforms = transforms
        
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), dataset_dir)
        
        for i, vehicle in enumerate(self.VEHICLE_TYPES):
            # ls file names + cleaning
            vehicle_path = os.path.join(self.dataset_dir, vehicle)
            paths = [path for path in os.listdir(vehicle_path) if path.endswith(".jpg")]
            
            self.data_paths.extend(paths)
            self.labels.extend([i] * len(paths))  

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        img_path = os.path.join(self.dataset_dir, self.VEHICLE_TYPES[label], self.data_paths[idx])
        img = Image.open(img_path).convert("RGB")     # - image: a PIL Image (H, W)
        
        # necessary transform
        img_tensor = F.pil_to_tensor(img)
        img_tensor = F.convert_image_dtype(img_tensor, TENSOR_DTYPE)
        img_tensor = self.transforms(img_tensor)

        return img_tensor, label

class DenseCityscapesDataset(Dataset):
    BASE_LINE = 0.222384
    FOCAL_LENGTH = 2273.82
    B_f = BASE_LINE*FOCAL_LENGTH
    
    def __init__(self, dataset_dir, transforms, depth_reqd=False):
        # from glob import glob
        # from os import path
        # self.files = []
        # for im_f in glob(path.join(dataset_path, '*_im.jpg')):
        #     self.files.append(im_f.replace('_im.jpg', ''))
        # self.transform = transform

        """
        Your code here
        """
        self.dataset_dir = dataset_dir 
        self.transforms = transforms
        self.depth_reqd = depth_reqd
        
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), dataset_dir)
        
        self.img_dir = os.path.join(self.dataset_dir, "image")
        self.label_dir = os.path.join(self.dataset_dir, "label")
        self.depth_dir = os.path.join(self.dataset_dir, "depth")
        
        ls = os.listdir(self.img_dir)
        if (ls.count('.DS_Store')): ls.remove('.DS_Store')
        self.img_paths = ls
        # for i in range(19):
        #     # ls file names + cleaning
        #     vehicle_path = os.path.join(self.dataset_dir, vehicle)
        #     paths = [path for path in os.listdir(vehicle_path) if path.endswith(".jpg")]
            
        #     self.data_paths.extend(paths)
        #     self.labels.extend([i] * len(paths))  

    def __len__(self):

        # return len(self.files)

        """
        Your code here
        """
        return len(self.img_paths)

    def __getitem__(self, idx):

        """
        Hint: generate samples for training
        Hint: return image, semantic_GT, and depth_GT
        """
        tgt_file_name = self.img_paths[idx]
        img = self.__loadNumpy(os.path.join(self.img_dir, tgt_file_name)).permute(2, 0, 1).to(TENSOR_DTYPE)
        label = self.__loadNumpy(os.path.join(self.label_dir, tgt_file_name)).to(torch.long)
        
        ## clean undefined label
        label[label < 0] = 255
        
        if self.depth_reqd:
            inverse_depth = self.__loadNumpy(os.path.join(self.depth_dir, tgt_file_name)).permute(2, 0, 1).to(TENSOR_DTYPE)
        
            ## handel depth
            depth = self.__getDepth(inverse_depth)
            img, label, depth = self.transforms(img, label, depth)
            return img, (label, depth)

        return self.transforms(img, label)
    
    @staticmethod
    def __getDepth(inverse_depth):
        ''' Stack segmentic labels and depth labels'''
        # conver disparity 
        disparity = (inverse_depth * 65535 - 1) / 256
        depth = DenseCityscapesDataset.B_f / disparity
        depth[depth.isinf()] = -1
        
        return depth
    
    @staticmethod
    def __getLabelMasks(label, depth):
        ''' Stack segmentic labels and depth labels'''
        H, W, _ = depth.shape
        depth_reshpaed = depth.reshape((H, W))
        return np.stack([label, depth_reshpaed], axis=2)
    
    @staticmethod
    def __loadNumpy(path):
        tgt_ndarray = np.load(path)
        return torch.from_numpy(tgt_ndarray)
        



