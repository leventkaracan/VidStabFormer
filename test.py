#test_cyclesttn_selfie_pre_raft
import torch
import numpy as np
import math
import torch.optim as optim
import torch.nn as nn
from datasetVTF import SelfieVTF
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

import cv2
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.io import imread
from model.VidStabFormerTest_selfie import  VidStabFormer
from torch.backends import cudnn
from torchvision import transforms, utils
cudnn.benchmark = True

import argparse

import time
from tqdm import tqdm


def current_milli_time():
    return round(time.time() * 1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
model = VidStabFormer(batch_size=BATCH_SIZE,istrain=False)

model = model.cuda()

max_epochs = 1000
PRINT_FREQ = 100
torch.autograd.set_detect_anomaly(True)

def require_grads(model,val):
    for param in model.parameters():
        param.requires_grad = val

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
####################



def cvReadGrayImg(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def saveOptFlowToImage(flow, basename, merge):
    bound=15
    flow = np.round((flow + bound) / (2. * bound) * 255.)
    flow[flow < 0] = 0
    flow[flow > 255] = 255
    flow = np.concatenate((flow, np.zeros((flow.shape[0],flow.shape[1],1))), axis=2)
    if merge:
        # save x, y flows to r and g channels, since opencv reverses the colors
        #print(flow[:,:,::-1].shape)
        cv2.imwrite(basename+'.png', flow[:,:,::-1])
    else:
        cv2.imwrite(basename+'_x.JPEG', flow[...,0])
        cv2.imwrite(basename+'_y.JPEG', flow[...,1])
######################

parser = argparse.ArgumentParser(description="Hscnet")


parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to saved model to resume from')
parser.add_argument('--start_epoch', nargs='?', type=int, default=0,
                        help='Path to saved model to resume from')




args = parser.parse_args()


#############################3
start_epoch=0

if(args.resume!=None):
    if args.start_epoch:
        model.load_state_dict(torch.load(args.resume+"G_e"+str(args.start_epoch)+".pth"))

        start_epoch = args.start_epoch
    else:
        model.load_state_dict(torch.load(args.resume+".pth"))


model.eval()

path2 = './data'

img_size = (480,856)
transform=transforms.Compose([
                                     transforms.Resize(img_size),
                                      transforms.ToTensor()
                                           ])

iter=0

train_dataset = SelfieVTF(path2, type="test", transform=transform, num_skips=5,img_size = img_size)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)

SAVE_PATH = "./results/"+str(start_epoch)+""

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
cur_vid_id = -1
with torch.no_grad():
    for epoch in range(1):
        running_total_loss = 0
        running_img_loss = 0
        running_feat_loss = 0


        epoch_total_loss = 0
        epoch_img_loss =0
        epoch_feat_loss = 0

        cnt = 0
        ccc = 15
        prelist = []
        for inputs,warpfields,vid_id in tqdm(train_loader):

            if(vid_id.item() != cur_vid_id):
                cur_vid_id = vid_id.item()
                ccc=15
                cnt=0
                if not os.path.exists(SAVE_PATH+"/"+str(vid_id.item())):
                    os.mkdir(SAVE_PATH+"/"+str(vid_id.item()))
                cur_save_path=SAVE_PATH+"/"+str(vid_id.item())

            pre_list = []
            stab_pre_list = []
            input_list = []
            width = inputs.size(4)
            height = inputs.size(3)
            bs = inputs.size(0)
            ttt = inputs.size(1)

            t=15
            if(cnt==0):
                prelist = [inputs[:,t,:,:,:].cuda(),inputs[:,t,:,:,:].cuda()]

            cnt = cnt + 1
            prevs = torch.cat((prelist[0],prelist[1]),dim=1).view(bs,6,height,width)

            input = inputs[:,t,:,:,:].cuda()
            warpfield = warpfields[:,t,:,:,:].cuda()
            perm = torch.randperm(30)

            inter = torch.cat((inputs[:,t-3:t,:,:,:],inputs[:,t+1:t+4,:,:,:]),dim=1).reshape(bs,18,height,width)# inters[:,t,:,:,:].reshape(bs,12,height,width)
            inter = inter.cuda()


            input_cat = torch.cat((input,inter,prevs),dim=1)

            output,_ = model(input_cat,warpfield=warpfield.detach())

            prelist.append(output[:,:,:,:])
            del prelist[0]

            iter = iter + 1
            save_image(output,cur_save_path+ "/frame"+str(ccc-15).zfill(6)+".png")
            ccc=ccc+1
