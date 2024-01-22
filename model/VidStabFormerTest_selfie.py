import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import softsplat
import cv2

from .pytorch_pwc.pwc import backwarp_fusion as backwarp
import matplotlib.pyplot as plt
from PIL import Image
import flow_vis
import math
from torch.autograd import Variable
from .raft import RAFT
import argparse

# RAFT
parser = argparse.ArgumentParser()
parser.add_argument('--small', nargs='?', type=bool, default=False,
                        help='Path to saved model to resume from')
parser.add_argument('--mixed_precision', nargs='?', type=bool, default=False,
                        help='Path to saved model to resume from')
parser.add_argument('--alternate_corr', nargs='?', type=bool, default=False,
                        help='Path to saved model to resume from')
parser.add_argument('--resume', nargs='?', type=str, default=".",
                        help='Path to saved model to resume from')
parser.add_argument('--start_epoch', nargs='?', type=int, default=-1,
                        help='Path to saved model to resume from')

args = parser.parse_args()

flow_model = torch.nn.DataParallel(RAFT(args))
flow_model.load_state_dict(torch.load('./model/raft_models/raft-things.pth'))
flow_model = flow_model.module
flow_model.to('cuda')
flow_model.eval()


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def load_image_list(image_files):
    images = []
    for imfile in image_files:
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

def flowestimate(img1, img2):
    with torch.no_grad():

        #images = load_image_list([img1, img2])

        flow_low, flow_up = flow_model(img1, img2, iters=20, test_mode=True)
    return flow_up

def flowwarp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv_function(in_c, out_c, k, p, s):
    return torch.nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s)

def get_conv_layer():
    conv_layer_base = conv_function
    return conv_layer_base

def get_bn_layer(output_sz):
    return torch.nn.BatchNorm2d(num_features=output_sz)

class ResNet_Block(torch.nn.Module):
    def __init__(self, in_c, in_o, downsample=None):
        super().__init__()
        bn_noise1 = get_bn_layer(output_sz=in_c)
        bn_noise2 = get_bn_layer(output_sz=in_o)

        conv_layer = get_conv_layer()

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = torch.nn.Identity()

        self.ch_a = torch.nn.Sequential(
            bn_noise1,
            torch.nn.ReLU(),
            conv_aa,
            bn_noise2,
            torch.nn.ReLU(),
            conv_ab,
            norm_downsample,
        )

        if downsample or (in_c != in_o):
            self.ch_b = torch.nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = torch.nn.Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b

class SpatialFeatureNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = ResNet_Block(3, 8)
        self.layer1 = ResNet_Block(8, 8)
        self.layer2 = ResNet_Block(8, 8)
        self.layer3 = ResNet_Block(8, 16)
        self.layer4 = ResNet_Block(16, 16)
        self.layer5 = ResNet_Block(16, 16)
        self.layer6 = ResNet_Block(16, 16)
        self.layer7 = ResNet_Block(16, 32)

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)

        return x_7

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, m, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, height*width)
            mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value, mm)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        #print("out : ",output.shape)
        x = self.output_linear(output)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c = x['x'], x['m'], x['b'], x['c']

        #m = torch.ones(m.size()).cuda()
        x = torch.nn.ReflectionPad2d((1, 1, 0, 0))(x)
        m = torch.nn.ReflectionPad2d((1, 1, 0, 0))(m)

        ###

        x = x + self.attention(x, m, b, c)
        x = x + self.feed_forward(x)
        x = x[:,:,:,1:-1]
        m = m[:,:,:,1:-1]

        return {'x': x, 'm': m, 'b': b, 'c': c}

class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        #print("x ",x.shape)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)

class AggregationWeightingNetwork(torch.nn.Module):
    def __init__(self, noDL_CNNAggregation=False, CNN_flowError=False):
        super().__init__()
        input_ch = 32 + 32 + 1 + 1
        if noDL_CNNAggregation:
            input_ch = 3 + 3 + 2 + 1 + 1 + 3
        if CNN_flowError:
            input_ch = 32 + 32 + 1 + 1 + 1
        self.layer0 = GatedConv2d_ResNet_Block(input_ch, 16)
        self.layer1 = GatedConv2d_ResNet_Block(16, 64, 'Down')
        self.layer2 = GatedConv2d_ResNet_Block(64, 64, 'Down')
        self.layer3 = GatedConv2d_ResNet_Block(64, 32)
        self.layer4 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        self.layer5 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        self.layer6 = GatedConv2d_ResNet_Block(32, 32)
        self.layer7 = GatedConv2d_ResNet_Block(32, 1)

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)

        return x_7

class GatedConv2d(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_c, out_c, k, p, s):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_c, out_c, k, s, p)
        self.mask_conv2d = torch.nn.Conv2d(in_c, out_c, k, s, p)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_c)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        #print(input.shape)
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.gated(mask)
        return x

class GatedConv2d_ResNet_Block(torch.nn.Module):
    def __init__(self, in_c, in_o, downsample=None):
        super().__init__()
        bn_noise1 = get_bn_layer(output_sz=in_c)
        bn_noise2 = get_bn_layer(output_sz=in_o)

        conv_layer = GatedConv2d

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = torch.nn.Identity()

        self.ch_a = torch.nn.Sequential(
            bn_noise1,
            torch.nn.ReLU(),
            conv_aa,
            bn_noise2,
            torch.nn.ReLU(),
            conv_ab,
            norm_downsample,
        )

        if downsample or (in_c != in_o):
            self.ch_b = torch.nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = torch.nn.Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b
j_idx=0

class VidStabFormer(BaseNetwork):
    def __init__(self, batch_size=1, FOV_expansion=1,istrain=True):
        super(VidStabFormer, self).__init__()
        self.spatialFeatureNetwork = SpatialFeatureNetwork()
        self.FOV_expansion = FOV_expansion
        self.istrain = istrain
        self.backwarp_tenGrid = {}
        self.aggregationWeightingNetwork = AggregationWeightingNetwork(0, True)

        channel = 256
        stack_num = 8
        patchsize = [(216, 120), (108, 60), (54, 30),(27,15)]
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)
        self.spatialFeatureNetwork = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.temporalFeatureNetwork = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.temporalResNet =  nn.Sequential(

            ResNet_Block(256, 256),
            ResNet_Block(256, 256),
            ResNet_Block(256, 256),
            ResNet_Block(256, 256),
            ResNet_Block(256, 256),
            ResNet_Block(256, 256),
            ResNet_Block(256, 256),
            ResNet_Block(256, 256),

            )
        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.pooling = nn.Sequential(
            nn.Conv2d(channel*7,channel,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.init_weights()




    def Img_pyramid(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
        return img_pyr

    def forward(self, frames,target=0, warpfield=None):

        input = frames
        unstable = input[:, :3, :, :]  # First frame

        global j_idx
        inter1 = input[:, 3:6, :, :]  # Second frame
        inter2 = input[:, 6:9, :, :]  # Third frame
        inter3 = input[:, 9:12, :, :]  # Fourth frame
        inter4 = input[:, 12:15, :, :]  # Fifth frame
        inter5 = input[:, 15:18, :, :]  # Fourth frame
        inter6 = input[:, 18:21, :, :]  # Fifth frame

        prev1 = input[:, 21:24, :, :]
        prev2 = input[:, 24:27, :, :]

        image_list = []
        image_list.append(unstable)
        image_list.append(inter1)
        image_list.append(inter2)
        image_list.append(inter3)
        image_list.append(inter4)
        image_list.append(inter4)
        image_list.append(inter5)
        image_list.append(inter6)


        if(self.istrain):
            F_kprime_to_k = flowestimate(target,unstable)
        else:
            F_kprime_to_k = warpfield


        img1_second = backwarp(tenInput=unstable, tenFlow=F_kprime_to_k)

        inter0_to_unstab = flowestimate(unstable,unstable)
        inter1_to_unstab = flowestimate(inter1,unstable)
        inter2_to_unstab = flowestimate(inter2,unstable)
        inter3_to_unstab = flowestimate(inter3,unstable)
        inter4_to_unstab = flowestimate(inter4,unstable)
        inter5_to_unstab = flowestimate(inter5,unstable)
        inter6_to_unstab = flowestimate(inter6,unstable)


        prev1_to_unstab = flowestimate(prev1,unstable)
        prev2_to_unstab = flowestimate(prev2,unstable)

        F_n_to_k_s = []
        _, _, img_h , img_w = inter0_to_unstab.size()
        F_n_to_k_s.append(F.interpolate(inter0_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(inter1_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(inter2_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(inter3_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(inter4_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(inter5_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(inter6_to_unstab, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(prev1_to_unstab, scale_factor=1.0/4.0, mode='nearest')*0.25)
        F_n_to_k_s.append(F.interpolate(prev2_to_unstab, scale_factor=1.0/4.0, mode='nearest')*0.25)

        F_n_to_k_s_orig = []
        F_n_to_k_s_orig.append(inter0_to_unstab)
        F_n_to_k_s_orig.append(inter1_to_unstab)
        F_n_to_k_s_orig.append(inter2_to_unstab)
        F_n_to_k_s_orig.append(inter3_to_unstab)
        F_n_to_k_s_orig.append(inter4_to_unstab)
        F_n_to_k_s_orig.append(inter5_to_unstab)
        F_n_to_k_s_orig.append(inter6_to_unstab)

        F_n_to_k_s_orig.append(prev1_to_unstab)
        F_n_to_k_s_orig.append(prev2_to_unstab)

        GAUSSIAN_FILTER_KSIZE = 5
        gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)

        h_padded = False
        w_padded = False
        minimum_size = 4


        features = []
        for i in range(7):
            features.append(self.spatialFeatureNetwork(frames[:,i*3:i*3+3]))

        prevs = input[:, 21:27, :, :]

        prev_feats = self.temporalFeatureNetwork(prevs)

        prev_feats = self.temporalResNet(prev_feats)

        W = 256
        H = 256
        tenOnes = torch.ones_like(features[0])[:, 0:1, :, :]
        tenOnes_orig = torch.ones_like(unstable)[:, 0:1, :, :]

        orig_F_kprime_to_k = F_kprime_to_k.clone()
        for_mask = F_kprime_to_k.clone()
        orig_F_kprime_to_k = torch.nn.ReplicationPad2d((W, W, H, H))(orig_F_kprime_to_k)
        F_kprime_to_k = F.interpolate(F_kprime_to_k, size=(int(np.ceil(img_h/4)),int(np.ceil(img_w/4))), mode='nearest')*0.25
        if self.FOV_expansion > 0:
            F_kprime_to_k_pad = torch.nn.ReplicationPad2d((W, W, H, H))(F_kprime_to_k)
        else:
            F_kprime_to_k_pad = torch.nn.ZeroPad2d((W, W, H, H))(F_kprime_to_k)

        tenWarpedFeat = []
        tenWarpedMask = []


        all_mask = []

        tenMaskFirst_orig = softsplat.softsplat(tenIn=torch.ones_like(unstable)[:, 0:1, :, :], tenFlow=F_n_to_k_s_orig[0], tenMetric=None, strMode='avg')
        tenMaskFirst_orig =  backwarp(tenInput=tenMaskFirst_orig, tenFlow=for_mask)


        m1 = torch.roll(tenMaskFirst_orig,4,0)
        m2 = torch.roll(tenMaskFirst_orig,-4,0)
        m3 = torch.roll(tenMaskFirst_orig,4,1)
        m4 = torch.roll(tenMaskFirst_orig,-4,1)

        tenMaskFirst_orig = tenMaskFirst_orig*m1*m2*m3*m4

        unstab_warped = backwarp(tenInput=unstable, tenFlow=for_mask)



        for idx, feat in enumerate(features):
            """padding for forward warping"""

            ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_n_to_k_s[idx])
            ref_frame_flow_orig = torch.nn.ReplicationPad2d((W, W, H, H))(F_n_to_k_s_orig[idx])

            tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(feat)

            """first forward warping"""
            tenWarpedFirst = softsplat.softsplat(tenIn=tenRef, tenFlow=ref_frame_flow, tenMetric=None, strMode='avg')

            tenWarpedSecond = backwarp(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad)

            tenWarped = tenWarpedSecond[:, :, H:-H, W:-W]

            tenWarpedFeat.append(tenWarped)
            tenWarpedMask.append(tenOnes)

        def length_sq(x):
            return torch.sum(x ** 2, dim=1, keepdim=True)
        j_idx = j_idx+1

        encoded_feat = torch.stack(tenWarpedFeat) # t b c h w

        t,bs,c,h,w = encoded_feat.size()
        encoded_feat = encoded_feat.permute(1,0,2,3,4)#bs,t,c,h,w
        encoded_feat = encoded_feat.contiguous().view(bs*t,c,h,w)
        masks = torch.stack(tenWarpedMask)
        masks = masks.view(bs*t, 1, h, w)

        trans_feat = self.transformer({"x": encoded_feat,"m": masks ,"b":bs, "c":256 })['x']

        fff = trans_feat.view(bs,t,c,h,w).view(bs,t*c,h,w)

        fff = self.pooling(fff)

        output = (self.decoder(fff+ prev_feats)+1)*0.5

        output = output[:,:,:,:] * torch.abs(tenMaskFirst_orig-1) + unstab_warped *tenMaskFirst_orig


        return output, torch.abs(tenMaskFirst_orig-1)
