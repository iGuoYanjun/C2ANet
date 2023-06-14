from __future__ import print_function
import argparse
import math
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pdb
import socket
import time
import numpy as np
import cv2
from imageio import imread
import scipy.io as scio
from torch.nn.functional import interpolate
from dataset import DatasetFromFolderTest
# from PIL import Image

# from ddcnet_nodense_x8 import Net as DDCNDX8
# from ddcnet_x8 import Net as DDCX8
# from dbpn8 import Net as DBPN8
# from ddcnet_parallel import Net as DDCP
# from ddcnet_edgedecouple_v2 import Net as DDCE
# from hrdsrnet_x8v2 import Net as PMBAX4
# from models_ahmf.ahmf import AHMF as ahmf
from models.C2ANet_x16 import Net as C2ANet

parser = argparse.ArgumentParser(description='PyTorch Super-Resolution Example')
# Device
parser.add_argument('--gpu_mode',           default=True, type=bool)
parser.add_argument('--gpus',               default=2, type=float, help='total number of gpus in device')
parser.add_argument('--gpu',                default=0, type=int,  help='the choosed gpu')
parser.add_argument('--DataParallel',       default=True, type=bool, help='whether to use parallelly training')

# Model
parser.add_argument('--model_type',         default='C2ANet', type=str, help='the choosed model')
parser.add_argument('--prefix',             default='x16_', help='mark of the model')
# parser.add_argument('--upscale_factor',     default=16, type=int, help='upscale factor of the model')

# Weights
parser.add_argument('--pretrained',         default=False, type=bool, help='whether to load pre-training weights')
parser.add_argument('--pretrained_sr',      default="", help='sr pretrained weights')

# Data In

parser.add_argument('--data_dir',           default='', type=str, help='root of test data')
parser.add_argument('--gt_path',            default='', type=str, help='root of test_gt data')
parser.add_argument('--save_path',           default='./result/', type=str, help='root of data')
# Data Out
parser.add_argument('--testBatchSize',      default=1, type=int, help='testing batch size')
parser.add_argument('--threads',            default=0, type=int,help='number of threads for data loader to use')
parser.add_argument('--seed',               default=123, type=int,help='random seed to use. Default=123')
opt = parser.parse_args()
print(opt)

"Obtain device info"
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
def test():
    sum_rmse = []
    model.eval()
    with torch.no_grad():
  
        for batch in testing_data_loader:

            lr_depth,  input_rgb, name = batch[0],batch[1],batch[2]
            if cuda:
                lr_depth = lr_depth.cuda()
                input_rgb = input_rgb.cuda()
            # lr_up = interpolate(lr_depth, scale_factor=16, mode='bicubic', align_corners=False)
            "Output"
            t0 = time.time()
            prediction = model(lr_depth,input_rgb)
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))           
            save_img(prediction.cpu().numpy(), opt.save_path,name[0])         
    calc_pred_rmse(opt.save_path, opt.gt_path)
    calc_pred_mad(opt.save_path, opt.gt_path)
         
def calc_pred_rmse(oo_path, gg_path):
    rmse_list= []
    rmse=0
    output = sorted((glob(oo_path+'*.png')))
    print('test_numbers_forRMSE: ',len(output))
    for i in range(len(output)):
        oo = imread(output[i]).astype(np.float)
        name = output[i].split('/')[-1].split('.')[0]
        gg = scio.loadmat(gg_path + name + '.mat')['gt_x16'].astype(np.float)
        h, w = gg.shape[: 2]
        gg = gg [: h - (h % 16), :w - (w % 16)]
        im_occ = np.ones_like(gg)
        if name.find('doll') != -1:
            im_occ[gg <= 9] = 0
        rmse_per = np.sqrt(np.mean(np.power((gg - oo)*im_occ, 2)))
        rmse_list.append(rmse_per)
        rmse += rmse_per
    print(rmse / len(output), rmse_list) 
def calc_pred_mad(oo_path, gg_path):
    mad_list= []
    mad=0
    output = sorted(glob(oo_path+'*.png'))
    print('test_numbers_forMAD: ',len(output))
    for i in range(len(output)):
        oo = imread(output[i]).astype(np.float32)
        # print(oo.shape)
        name = output[i].split('/')[-1].split('.')[0]
        gg = scio.loadmat(gg_path + name + '.mat')['gt_x16'].astype(np.float32)
        h, w = gg.shape[: 2]
        gg = gg [: h - (h % 16), :w - (w % 16)]
        mad_per = np.mean(abs(gg - oo))
        mad_list.append(mad_per)
        mad += mad_per
    print(mad / len(output), mad_list) 



def save_img(img, save_dir, img_name):

    img= np.clip(img * 255, 0, 255).round().astype(np.uint8)
    img = img.squeeze()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)       
    save_fn = './'+save_dir + img_name.split('/')[-1]

    cv2.imwrite(save_fn,img)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

"Check cuda"
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

"Choose Seed"
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


"Loading Dataset"

print('=====> Loading Test Dataset: ', opt.data_dir)


test_set = DatasetFromFolderTest(opt.data_dir,opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

"Building Net"
print('===> Building model: ', opt.model_type)
if opt.model_type == 'C2ANet':
    model = C2ANet(num_channels=1, base_filter=64, scale_factor=opt.upscale_factor)
"DataParallel"
if opt.DataParallel:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

"Loading Pre-trained weights"

flag = 1
if os.path.exists(opt.pretrained_sr):
    model.load_state_dict(torch.load(opt.pretrained_sr, map_location=lambda storage, loc: storage))
    flag=0
    print('<--------------Pre-trained SR model is loaded.-------------->')
if flag == 1:
    print('!--------------Cannot load pre-trained model! --------------!')

"To cuda"
if cuda:
    model = model.cuda(gpus_list[opt.gpu])
    criterion_L1 = criterion_L1.cuda(gpus_list[opt.gpu])
    criterion_L2 = criterion_L2.cuda(gpus_list[opt.gpu])
"Testing"
test()

