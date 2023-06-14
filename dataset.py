import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import random
from random import randrange
import glob
import cv2
import imageio
from torchvision.transforms import  ToTensor
import scipy.io as scio



"test ahmf in training"
class DatasetFromFolderTest(data.Dataset):
    def __init__(self, dataset, scale, input_transform=None, input_rgb_transform=None, target_transform=None):
        super(DatasetFromFolderTest, self).__init__()
        # self.image_filepaths = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.dataset = dataset
        self.scale =scale
        self.all_files = sorted(glob.glob(self.dataset+ '*.mat'))

    def __getitem__(self, index):

        "Load the data by determining the type of the dataset"
        i_file = self.all_files[index]
        i_name = i_file.split('/')[-1].split('.')[0]
        lr = scio.loadmat(self.dataset+  i_name + '.mat')['bic_x16']
        rgb =scio.loadmat(self.dataset+  i_name + '.mat')['rgb_x16']
        # gt =scio.loadmat(self.dataset+  i_name + '.mat')['gt_x16']
        # lr = scio.loadmat(self.dataset+  i_name + '.mat')['bic_x4']
        # rgb =scio.loadmat(self.dataset+  i_name + '.mat')['rgb_x4']
        # gt =scio.loadmat(self.dataset+  i_name + '.mat')['gt_x4']
        h, w = rgb.shape[: 2]
        rgb = rgb [: h - (h % 16), :w - (w % 16)]
        img_max = 255
        delta = 0
        lr = lr/(img_max + delta)
        rgb =np.transpose(rgb, (2, 0, 1)) / 255        
        rgb = torch.from_numpy(rgb).float()
        lr = torch.from_numpy(lr).unsqueeze(0).float()
        return  lr,rgb, i_name+'.png'
    def __len__(self):
        return len(self.all_files)

