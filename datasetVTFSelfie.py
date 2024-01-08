import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import os

import torch.nn.functional as F

def read_homography(H_path):
    xv, yv = np.meshgrid(np.linspace(0, 832 + 2 * 64 - 1, 832 + 2 * 64), np.linspace(0, 448 + 2 * 64 - 1, 448 + 2 * 64))
    H_inv = np.load(H_path)
    #print(H_inv)
    #print(H_inv.shape)

    if np.sum(np.abs(H_inv)) == 0.0:
        H_inv[0, 0] = 1.0
        H_inv[1, 1] = 1.0
        H_inv[2, 2] = 1.0
    xv_prime = (H_inv[0, 0] * xv + H_inv[0, 1] * yv + H_inv[0, 2]) / (H_inv[2, 0] * xv + H_inv[2, 1] * yv + H_inv[2, 2])
    yv_prime = (H_inv[1, 0] * xv + H_inv[1, 1] * yv + H_inv[1, 2]) / (H_inv[2, 0] * xv + H_inv[2, 1] * yv + H_inv[2, 2])
    flow = np.stack((xv_prime - xv, yv_prime - yv), -1)
    return flow
backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end
def read_flo(flo_path):
    #print(flo_path)
    xv, yv = np.meshgrid(np.linspace(-1, 1, 832 + 2 * 64), np.linspace(-1, 1, 448 + 2 * 64))

    flow = np.load(flo_path)
    if(flow.shape[1]!=832 + 2 * 64):
        flow = np.zeros((576,960,2))


    flow_u = ((flow[:, :, 0] + xv) + 1.0) / 2.0 * float(832+2*64-1)
    flow_v = ((flow[:, :, 1] + yv) + 1.0) / 2.0 * float(448+2*64-1)
    flow_u -= ((xv + 1.0) / 2.0 * float(832+2*64-1))
    flow_v -= ((yv + 1.0) / 2.0 * float(448+2*64-1))
    flow = np.stack((flow_u, flow_v), -1)
    return flow

class SelfieVTF(Dataset):

    def __init__(self, img_root, type="test", transform=None,num_skips=5,vid_id=1,frame_num=0,seq_len=30,img_size=(180,320)):
        self.seq_len = seq_len
        self.img_transform = transform
        self.img_root = img_root
        if self.img_transform is None:
            self.img_transform = transforms.ToTensor()
        self.type = type
        self.num_skips = num_skips
        self.vid_id = vid_id
        self.base = img_root+"/Selfie"
        self.img_size = img_size

        self.frame_start=0
        self.frame_num = frame_num
        self.prev_frame_num = 5
        self.prev_frame_skip = [7,13,19,25,31]
        #self.data = self._load_dataset()
        self.generatedSteady = []
        self.vids_list = open(os.path.join(img_root+"/Selfie", "test.txt")).readlines()
        self.A_paths = []
        self.WF_paths = []
        self.H_paths = []


        #self.B_paths = []
        #self.int_paths = []
        #self.prev_skip_paths = []

        self.num_of_seq =0
        self.num_of_frame =0
        self.vid_len_list= []
        self.vid_id_list = []
        for idx, line in enumerate(self.vids_list):
            #print(idx)
            self.A_paths.append([])
            self.WF_paths.append([])
            self.H_paths.append([])

            #self.B_paths.append([])

            #self.int_paths.append([])
            #self.prev_skip_paths.append([])
            x = line.split()
            self.vid_id_list.append(int(x[0]))
            fr_nm = int(x[1])
            self.vid_len_list.append(fr_nm)
            for i in range(fr_nm):

                filename = os.path.join(self.base + "/" + x[0] + "/frame" + str(i).zfill(6)+".png")
                self.A_paths[idx].append(filename)
                wf_filename = os.path.join(self.base + "/" + x[0] + "_wf/"+ str(i).zfill(5)+".npy")
                self.WF_paths[idx].append(wf_filename)
                h_inv_filename = os.path.join(self.base + "/" + x[0] + "_wf/"+ str(i).zfill(5)+"_H_inv.npy")
                self.H_paths[idx].append(h_inv_filename)

                #filename2 = os.path.join(self.base + "/stable/" + x[0] +"/frame" + str(i).zfill(5)+".jpg")
                #self.B_paths[idx].append(filename2)
                #image_list = []
                image_prev = []
                #for j in range(1, self.num_skips):
                #    filename = "/" +  x[0] + "/skip" + str(j) + "/"  #1/skip0/frame00000,1/skip1/frame00000,1/skip2/frame00000...
                #    filename1 = os.path.join(self.base + filename+"frame" + str(i).zfill(5)+".jpg") #/mnt/sdb/users/semiha/MyProjects/VGG/dataloeder_skip/1/skip0/frame00000.jpg
                #    image_list.append(filename1)
                #for j in range(0, len(self.prev_frame_skip)):
                #    filename = "/stable/" + x[0] + "/"  #1/skip0/frame00000,1/skip1/frame00000,1/skip2/frame00000...
                #    filename1 = os.path.join(self.base + filename+"frame" + str((i-self.prev_frame_skip[j])).zfill(5)+".jpg") #/mnt/sdb/users/semiha/MyProjects/VGG/dataloeder_skip/1/skip0/frame00000.jpg
                #    image_prev.append(filename1)
                #self.int_paths[idx].append(image_list)
                #self.prev_skip_paths[idx].append(image_prev)
                self.num_of_frame=self.num_of_frame+1

            self.num_of_seq = self.num_of_seq +1

    def __len__(self):
        if(self.type=="train"):
            return self.num_of_seq*10
        else:
            return self.num_of_frame

    def name(self):
        return 'NUS'
    def get_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return img

    def resize_nearest_torch(self,mask,size):
        w,h,_ = mask.shape
        mask = torch.Tensor(mask).permute(2,0,1).unsqueeze(0)
        #print("Mask shape : ",mask.shape)
        mask = F.interpolate(mask, size, mode='nearest')
        #print("Mask shape 2 : ",mask.shape)
        return mask

    def __getitem__(self, index):
        if(self.type =="train"):
            A_paths = self.A_paths[index % self.num_of_seq]
            WF_paths = self.WF_paths[index % self.num_of_seq]
            H_paths = self.H_paths[index % self.num_of_seq]

            #B_paths = self.B_paths[index % self.num_of_seq]

            #prev_paths = self.prev_skip_paths[index % self.num_of_seq]
            #int_paths = self.int_paths[index % self.num_of_seq]
            #print(A_paths[index% self.num_of_seq])
            n_frames_total, start_idx, t_step = get_video_params(self.seq_len, len(A_paths), index)

            A = 0
            B = 0
            P = 0
            I = 0
            for i in range(self.seq_len):
                #print(i)
                prev_list = []
                int_list = []
                A_path = A_paths[start_idx + i * t_step]
                #print(A_path)
                Ai=Image.open(A_path)
                H,W = self.img_size
                #print("H ",H, "W ",W)
                Ai =self.img_transform(Ai).unsqueeze(0)
                A = Ai if i == 0 else torch.cat([A, Ai], dim=0)

                WF_path = WF_paths[start_idx + i * t_step]
                H_path = H_paths[start_idx + i * t_step]

                tenH_inv = torch.FloatTensor(np.ascontiguousarray(read_homography(H_path).transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenFlow = torch.FloatTensor(np.ascontiguousarray(read_flo(WF_path).transpose(2, 0, 1)[None, :, :, :])).cuda()
                #print("TF: ",tenFlow.shape)

                tenBackFlow = backwarp(tenInput=tenH_inv, tenFlow=tenFlow)
                totalFlowIn832 = (tenBackFlow+tenFlow)[:, :, 64:-64, 64:-64]
                #print("832?: ",totalFlowIn832.shape)
                """second backward warping in full resolution"""
                W_ratio = W/(832)
                H_ratio = H/(448)
                totalFlow = F.upsample(totalFlowIn832, size=self.img_size, mode='bilinear')
                #print("totalFlow : ",totalFlow.shape)
                F_kprime_to_k = torch.stack((totalFlow[:, 0]*W_ratio, totalFlow[:, 1]*H_ratio), dim=1)
                '''
                if H % 4 == 0:
                    boundary_cropping_h = 4
                else:
                    boundary_cropping_h = 3
                if W % 4 == 0:
                    boundary_cropping_w = 4
                else:
                    boundary_cropping_w = 3
                '''

                #WFi = F_kprime_to_k[:, :, boundary_cropping_h:-boundary_cropping_h, boundary_cropping_w:-boundary_cropping_w]
                WFi = F_kprime_to_k
                #print(WFi.shape)
                #print(A_path)
                #WFi=np.load(WF_path)

                #print(tenH_inv.shape)

                #print(WFi.shape)
                #WFi =self.resize_nearest_torch(WFi,self.img_size)
                WF = WFi if i == 0 else torch.cat([WF, WFi], dim=0)

                #B_path =B_paths[start_idx + i * t_step]
                #print(A_path)
                #Bi=Image.open(B_path)
                #Bi =self.img_transform(Bi).unsqueeze(0)
                #B = Bi if i == 0 else torch.cat([B, Bi], dim=0)

                #pimg = ""
                #for j in range(0,self.prev_frame_num):
                #    prev= prev_paths[start_idx + i * t_step][j]
                #    img = ImageOps.grayscale(Image.open(prev))
                #    prev =self.img_transform(img)
                #    prev_list.append(prev)
                #Pi = torch.stack(prev_list)
                #Pi = Pi.unsqueeze(0)
                #P = Pi if i == 0 else torch.cat([P, Pi], dim=0)

                #intimg = ""
                #for j in range(0,self.num_skips-1):
                #    inter= int_paths[start_idx + i * t_step][j]
                #    img = Image.open(inter)
                #    inter =self.img_transform(img)
                #    int_list.append(inter)
                #Ii = torch.stack(int_list)
                #Ii = Ii.unsqueeze(0)
                #I = Ii if i == 0 else torch.cat([I, Ii], dim=0)
            return A,WF#A,B,WF,P,I
        else:
            pre_sub_tot=0
            sub_tot=0
            which_vid=0
            which_frame=0
            vid_len = 0

            for k in range(self.num_of_seq):
                pre_sub_tot = sub_tot
                sub_tot = sub_tot+self.vid_len_list[k]
                which_vid = k
                vid_len = self.vid_len_list[k]
                if(index<sub_tot):
                    which_frame = index-pre_sub_tot
                    break
            #print(str(which_vid)+" "+str(which_frame))


            A_paths = self.A_paths[which_vid]
            WF_paths = self.WF_paths[which_vid]
            H_paths = self.H_paths[which_vid]

            I = 0
            start_frm = which_frame-15
            for i in range(30):

                if(start_frm<0):
                    get_frm = 0
                elif(start_frm>vid_len-1):
                    get_frm = vid_len-1
                else:
                    get_frm = start_frm


                int_list = []
                A_path = A_paths[get_frm]
                #print(A_path)
                Ai=Image.open(A_path)
                Ai =self.img_transform(Ai).unsqueeze(0)
                A = Ai if i == 0 else torch.cat([A, Ai], dim=0)
                H,W = self.img_size
                #print("H ",H, "W ",W)
                WF_path = WF_paths[get_frm]
                H_path = H_paths[get_frm]

                tenH_inv = torch.FloatTensor(np.ascontiguousarray(read_homography(H_path).transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenFlow = torch.FloatTensor(np.ascontiguousarray(read_flo(WF_path).transpose(2, 0, 1)[None, :, :, :])).cuda()
                #print("TF: ",tenFlow.shape)

                tenBackFlow = backwarp(tenInput=tenH_inv, tenFlow=tenFlow)
                totalFlowIn832 = (tenBackFlow+tenFlow)[:, :, 64:-64, 64:-64]
                #print("832?: ",totalFlowIn832.shape)
                """second backward warping in full resolution"""
                W_ratio = W/(832)
                H_ratio = H/(448)
                totalFlow = F.upsample(totalFlowIn832, size=(H,W), mode='bilinear')
                #print("totalFlow : ",totalFlow.shape)
                F_kprime_to_k = torch.stack((totalFlow[:, 0]*W_ratio, totalFlow[:, 1]*H_ratio), dim=1)

                WFi = F_kprime_to_k

                WF = WFi if i == 0 else torch.cat([WF, WFi], dim=0)
                start_frm = start_frm + 1

            return A,WF,self.vid_id_list[which_vid]


def get_video_params( n_frames_total, cur_seq_len, index):
    tG = 3
    isTrain=True
    #print(n_frames_total)
    #print(cur_seq_len)
    #print(index)

    if isTrain:
        n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)
        n_gpus =1       # number of generator GPUs for each batch
        n_frames_per_load = n_gpus        # number of frames to load into GPUs at one time (for each batch)
        n_frames_per_load = min(n_frames_total, n_frames_per_load)
        n_loadings = n_frames_total // n_frames_per_load           # how many times are needed to load entire sequence into GPUs
        n_frames_total = n_frames_per_load * n_loadings + tG - 1   # rounded overall number of frames to read from the sequence
        max_t_step = min(1, (cur_seq_len-1) // (n_frames_total-1))
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible index for the first frame
        start_idx = np.random.randint(offset_max)                 # offset for the first frame to load
    else:
        n_frames_total = tG
        start_idx = index
        t_step = 1
    return n_frames_total, start_idx, t_step
