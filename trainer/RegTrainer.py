#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset,ValDataset,TestDataset
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
import torchvision
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import lpips
import lpips
# loss_fn_alex = lpips.LPIPS(net='vgg')#也可以选择alex
loss_fn_alex = lpips.LPIPS(net='alex')
import pydicom
import shutil
import copy
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr, compare_mse
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def to_windowdata(image,WC,WW):
    image = (image + 1) * 0.5 * 4095
    image[image == 0] = -2000
    image=image-1024
    center = WC #40 400//60 300
    width = WW# 200
    try:
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    except:
        # print(WC[0])
        # print(WW[0])
        center = WC[0]  # 40 400//60 300
        width = WW[0]  # 200
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = np.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0
    image=image/255#np.uint8(image)
    image = (image - 0.5)/0.5
    return image

#平扫CT [30 80]
def to_windowdat1(image):
    image = (image + 1) * 0.5 * 4095
    image[image == 0] = -2000
    center = 30
    width = 85
    win_min = (2 * center - width) / 2.0 + 0.5
    win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = np.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0
    return image

def pplot(a,b,c):
    plt.subplot(2, 2, 1)
    plt.imshow(a, cmap='gray')  # ,vmin=0,vmax=255
    plt.subplot(2, 2, 2)
    plt.imshow(b, cmap='gray')  # ,vmin=0,vmax=255
    plt.subplot(2, 2, 3)
    plt.imshow(c, cmap='gray')  # ,vmin=0,vmax=255
    plt.show()  # c h w

#窗融合(xx:不加残差，xxx:加残差)+多尺度判别
class Reg_Trainer():#引入多尺度判别器
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks #x:窗数据融合，xx:引入pix2pixHD,xxx:
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),lr=config['lr'], betas=(0.5, 0.999))
        self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
        self.spatial_transform = Transformer_2D().cuda()
        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        # self.L_GM_loss = torch.nn.LGMLoss()
        self.L1_loss = torch.nn.L1Loss()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor


        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level
        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level], fillcolor=-1),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level], fillcolor=-1),  #
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoaderX(
            ImageDataset(config['train_list'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], pin_memory=True)#config['n_cpu']

        val_transforms = [ToTensor(),
                          Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = DataLoaderX(ValDataset(config['val_list'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'],pin_memory=True)
        self.test_data = DataLoaderX(TestDataset(config['test_list'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])
        # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs']+config['decay_epoch'], len(self.dataloader))

    def update_learning_rate(self):
        lrd = self.config['lr'] / self.config['decay_epoch']#学习率衰减
        lr = self.config['lr'] - lrd
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_R_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.config['lr'], lr))
        self.config['lr'] = lr

    def train(self):
        ###### Training ######
        # self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B_xxx_19_26.1403_0.8204.pth'))
        # self.netD_B.load_state_dict(torch.load(self.config['save_root'] + 'netD_B_xxx_19_26.1403_0.8204.pth'))
        # self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'R_A_xxx_19_26.1403_0.8204.pth'))

        for epoch in range(self.config['epoch']+1, self.config['n_epochs']+1+self.config['decay_epoch']):
            if epoch > self.config['n_epochs']:
                self.update_learning_rate()
            for i, batch in enumerate(self.dataloader):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                #### regist sys loss
                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B, real_B)
                SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                pred_fake0 = self.netD_B(fake_B)
                adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                ####smooth loss
                SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                toal_loss = SM_loss + adv_loss + SR_loss
                toal_loss.backward()
                self.optimizer_R_A.step()
                self.optimizer_G.step()

                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake0 = self.netD_B(fake_B)
                pred_real = self.netD_B(real_B)
                loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake) + self.config[
                    'Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                loss_D_B.backward()
                self.optimizer_D_B.step()

                ###################################

                self.logger.log({'loss_D_B': loss_D_B, },
                                images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})  # ,'SR':SysRegist_A2B

            ############val###############
            if epoch%5==0:
                with torch.no_grad():
                    SSIM = 0
                    PSNR=0
                    num = 0
                    for i, batch in enumerate(self.val_data):
                        real_A = Variable(self.input_A.copy_(batch['A']))  ###注意generator输入是[1,3,512,512]
                        real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                        fake_B= self.netG_A2B(real_A,real_A).detach().cpu().numpy().squeeze()
                        # real_B = (real_B * 0.5 + 0.5) * 255
                        # fake_B = (fake_B * 0.5 + 0.5) * 255
                        psnr = self.PSNR(fake_B, real_B)  # fake_B
                        PSNR += psnr
                        ssim = measure.compare_ssim(fake_B, real_B)
                        SSIM += ssim
                        num += 1
                    print('PSNR:', PSNR / num)
                    print('SSIM:', SSIM / num)
                #         # Save models checkpoints
                if not os.path.exists(self.config["save_root"]):
                    os.makedirs(self.config["save_root"])
                st=str(epoch)+'_'+ str(round(PSNR / num, 4))+'_'+str(round(SSIM / num, 4))
                torch.save(self.netG_A2B.state_dict(),
                           self.config['save_root'] + "netG_A2B_"+st+ ".pth")
                torch.save(self.R_A.state_dict(),
                           self.config['save_root'] + "R_A_" +st+".pth")
                torch.save(self.netD_B.state_dict(),
                           self.config['save_root'] + "netD_B_"+st+".pth")
            else:
                if not os.path.exists(self.config["save_root"]):
                    os.makedirs(self.config["save_root"])
                st=str(epoch)
                torch.save(self.netG_A2B.state_dict(),self.config['save_root'] + "netG_A2B_"+st+ ".pth")
                torch.save(self.R_A.state_dict(),self.config['save_root'] + "R_A_" +st+".pth")
                torch.save(self.netD_B.state_dict(),self.config['save_root'] + "netD_B_"+st+".pth")

    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B_5b.pth'))#43netG_A2Ba512.pth netG_A2Bc512.pth
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        path = 'f'
        ii = 0
        with torch.no_grad():
                MAEw = 0
                PSNRw = 0
                SSIMw = 0
                LPIPSw=0
                UQIw = 0

                MAE = 0
                PSNR = 0
                SSIM = 0
                LPIPS=0
                UQI = 0
                num = 0
                for i, batch in enumerate(self.test_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    A_path=batch['A_path']
                    file_path = A_path[0].replace('../../../', '../../')
                    # ds = pydicom.dcmread(file_path, force=True)  # 读取头文件
                    file_pathE=file_path.replace('SE0','SE1')
                    ds = pydicom.dcmread(file_pathE, force=True)  # 读取头文件
                    # name=os.path.split(A_path[0])[-1].split('.')[0]
                    name = A_path[0].split('SE0/')[1]
                    if path != file_path.split('IM')[0]:
                        path = file_path.split('IM')[0]  # patients
                        ii = ii + 1
                        out_path0 = self.config['image_save']+'/ST0/' +str(ii)+'/SE0/'
                        out_path1 = self.config['image_save']+'/ST0/'  +str(ii)+'/SE1/'
                        out_path2 = self.config['image_save']+'/ST1/' +str(ii)+'/SE2/'
                        if not os.path.isdir(out_path0):
                            os.makedirs(out_path0)
                        if not os.path.isdir(out_path1):
                            os.makedirs(out_path1)
                        if not os.path.isdir(out_path2):
                            os.makedirs(out_path2)
                    file_path0 = os.path.join(out_path0, name)
                    file_path1 = os.path.join(out_path1, name)
                    # print(name)
                    fake_B = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    # real_A = real_A.detach().cpu().numpy().squeeze()

                    # best window
                    # WC = 50#ds.WindowCenter
                    # WW = 400#ds.WindowWidth
                    WC = ds.WindowCenter
                    WW = ds.WindowWidth
                    # aa = to_windowdata(real_A, WC, WW)
                    # aa[aa < 0.3] = 0
                    # aa[aa >= 0.3] = 1

                    b = to_windowdata(real_B, WC, WW)
                    bb = copy.deepcopy(b)
                    bb[bb < 0.3] = 0
                    bb[bb >= 0.3] = 1
                    # bb = bb - aa
                    # bb[bb < 0.3] = 0
                    b = b * bb
                    b[b == 0] = -1

                    c = to_windowdata(fake_B, WC, WW)*bb
                    cc=copy.deepcopy(c)
                    cc[cc < 0.3] = 0
                    cc[cc >= 0.3] = 1
                    # cc=cc-aa
                    # cc[cc < 0.3] = 0
                    c = c * cc
                    c[c == 0] = -1

                    # a2 = (c + 1) * 0.5 * 255
                    # plt.imshow(a2, cmap='gray')  # ,vmin=0,vmax=255
                    # plt.show()
                    maew = self.MAE(c, b)
                    psnrw = self.PSNR(c, b)  # fake_B
                    # lpips_loss=loss_fn_alex.forward(c, b)
                    ssimw = measure.compare_ssim(c, b)
                    lpips_lossw = loss_fn_alex.forward(torch.tensor(c), torch.tensor(b))
                    uqiw = self.UQI(c, b)
                    MAEw += maew
                    PSNRw += psnrw
                    SSIMw += ssimw
                    UQIw += uqiw
                    # print(lpips_loss.numpy()[0][0][0][0])
                    LPIPSw = LPIPSw + lpips_lossw.numpy()[0][0][0][0]

                    # dicom
                    real_BB = copy.deepcopy(real_B)
                    real_B=real_B*bb
                    real_B[real_B==0]=-1

                    fake_BB=copy.deepcopy(fake_B)
                    fake_B=fake_B*cc
                    fake_B[fake_B==0]=-1

                    # abc = (real_B + 1) * 0.5 * 255
                    # plt.imshow(abc, cmap='gray')  # ,vmin=0,vmax=255
                    # plt.show()
                    mae = self.MAE(fake_B, real_B)
                    psnr = self.PSNR(fake_B, real_B)
                    ssim = measure.compare_ssim(fake_B, real_B)
                    lpips_loss=loss_fn_alex.forward(torch.tensor(fake_B),torch.tensor(real_B))
                    uqi = self.UQI(fake_B, real_B)
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim
                    LPIPS = LPIPS + lpips_loss.numpy()[0][0][0][0]
                    UQI += uqi
                    num+=1

                    # cv2.imwrite(self.config['image_save']+name+'a.png',a)
                    # cv2.imwrite(self.config['image_save']+name+'b.png',b)
                    # cv2.imwrite(self.config['image_save']+name+'c.png',c)

                    newimg = (fake_BB + 1) * 0.5 * 4095
                    # newimg[newimg == 0] = -2000
                    if ds[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
                        newimg = newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm
                    elif ds[0x0028, 0x0100].value == 8:
                        newimg = newimg.astype(np.int8)
                    else:
                        raise Exception("unknow Bits Allocated value in dicom header")
                    ds.PixelData = newimg.tobytes()  # 替换矩阵

                    # shutil.copy(file_path, file_path0)
                    # shutil.copy(file_path.replace('SE0','SE1'), file_path1)
                    pydicom.dcmwrite(out_path2+name,ds)

                print ('MAEw',MAEw/num)
                print ('PSNRw:',PSNRw/num)
                print ('SSIMW:',SSIMw/num)
                print('LPIPSW:', LPIPSw/num)
                print ('UQIW:',UQIw/num)

                print('\n')
                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
                print('LPIPS:', LPIPS / num)
                print('UQI:', UQI / num)

    def PSNR(self, fake, real):
        a = np.where(real != -1)  # Exclude background
        x = a[0]
        y = a[1]
        if x.size==0 or y.size==0:
            mse = np.mean(((fake+ 1) / 2. - (real + 1) / 2.) ** 2)+1e-10
        else:
            mse = np.mean(((fake[x, y] + 1) / 2. - (real[x, y] + 1) / 2.) ** 2)
        # mse = np.mean(((fake + 1) / 2. - (real+ 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX /(np.sqrt(mse)+1e-10))

    def MAE(self, fake, real):
        a = np.where(real != -1)  # Exclude background
        x=a[0]
        y=a[1]
        # print(a[2].shape)
        # print(x.shape)
        # print(y.shape)
        if x.size==0 or y.size==0:
            mae = np.nanmean(np.abs(fake- real))+1e-10
        else:
            mae = np.nanmean(np.abs(fake[x, y] - real[x, y]))
        return mae / 2  # from (-1,1) normaliz  to (0,1)

    def UQI(self, fake, real):
        meanf = np.mean(fake)
        meanr = np.mean(real)
        m, n = np.shape(fake)
        varf = np.sqrt(np.sum((fake - meanf) ** 2) / (m * n - 1))
        varr = np.sqrt(np.sum((real - meanr) ** 2) / (m * n - 1))
        cov = np.sum((fake - meanf) * (real - meanr)) / (m * n - 1)
        UQI = 4 * meanf* meanr * cov / ((meanf ** 2 + meanr ** 2) * (varf ** 2 + varr ** 2)+1e-10)
        return UQI
