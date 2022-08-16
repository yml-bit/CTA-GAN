import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pydicom
import SimpleITK as sitk
from matplotlib import pyplot as plt
import torch

#输入输出都是窗数据
def read_w(file_path):
    image = np.load(file_path)
    image = (image + 1) * 0.5 * 4095
    image[image == 0] = -2000
    image = image - 1024
    #if "C+" in st:#宽对窄
    center = 40#ds.WindowCenter
    width = 400#ds.WindowWidth
    win_min = (2 * center - width) / 2.0 + 0.5
    win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = np.trunc(image * dFactor)
    image[image>255]=255
    image[image<0]=0
    image=image/255#np.uint8(image)
    image = (image - 0.5)/0.5
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap='gray')#,vmin=0,vmax=255
    # plt.show()
    return image

def read_ori_w(file_path):#read_dicom_mw
    file_path=file_path.replace('../../../', '../../')
    dicom = sitk.ReadImage(file_path)
    data1 = np.squeeze(sitk.GetArrayFromImage(dicom))
    data=data1+1024
    # ds = pydicom.dcmread(file_path, force=True)  # 读取头文件
    # data=(ds.pixel_array).astype(np.int)
    # data1=data-1024
    #if "C+" in st:#宽对窄
    center =50# ds.WindowCenter 50
    width = 400#ds.WindowWidth # 400
    win_min = (2 * center - width) / 2.0 + 0.5#-149.5
    win_max = (2 * center + width) / 2.0 + 0.5#250.5
    dFactor = 255.0 / (win_max - win_min)#把窗内
    image = data1 - win_min #sitk读取的数值比pydicom读取的数值小1024
    # image=data1+149.5
    image1 = np.trunc(image * dFactor)#dFactor
    image1[image1>255]=255
    image1[image1<0]=0
    image1=image1/255#np.uint8(image)
    image1 = (image1 - 0.5)/0.5

    image2=data#sitk读取的数值比pydicom读取的数值小1024
    image2[image2<0]=0#-2000->0
    image2=image2/4095
    image2 = (image2 - 0.5)/0.5

    # image1=(image1*2-1)*255
    # image2=(image2*2-1)*255
    # plt.subplot(2, 2, 1)
    # plt.imshow(image1*255, cmap='gray')#,vmin=0,vmax=255
    # plt.subplot(2, 2, 2)
    # plt.imshow(image2*255, cmap='gray')#,vmin=0,vmax=255
    # plt.show()

    return image1,image2

#真实平扫头部数据
def read_dicom(file_path):
    ds = pydicom.dcmread(file_path.replace('../../../','../../'), force=True)  # 读取头文件
    image2=(ds.pixel_array).astype(np.int)

    #sitk读取的数值比pydicom读取的数值小1024
    image2[image2<0]=0
    image2=image2/4095
    image2 = (image2 - 0.5)/0.5
    return image2

#使用原始数据
class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        f = open(root)#test train
        train_lista = []
        train_listb = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            train_lista.append(line)
            train_listb.append(line.replace("SE0","SE1"))
        f.close()  # 关
        self.files_A=sorted(train_lista)
        self.files_B = sorted(train_listb)
        # self.files_A = sorted(glob.glob("%s/A/*" % root))
        # self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned

    def __getitem__(self, index):
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img tranfsorms
        item_A = read_dicom(self.files_A[index % len(self.files_A)])
        item_A = self.transform1(item_A.astype(np.float32))
        #random.seed(seed)
        if self.unaligned:
            item_B = read_dicom(self.files_B[random.randint(0, len(self.files_B) - 1)])
            item_B = self.transform2(item_B.astype(np.float32))
        else:
            item_B = read_dicom(self.files_B[index % len(self.files_B)])
            item_B = self.transform1(item_B.astype(np.float32))
        # a=item_B.max()
        # b=item_B.min()
        # print(item_B)
        # a=item_B.detach().cpu().numpy().squeeze()
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        f = open(root)
        train_lista = []
        train_listb = []
        for line in f.readlines():
            line = line.strip('\n')
            train_lista.append(line)
            train_listb.append(line.replace("SE0","SE1"))
        f.close()  # 关
        self.files_A=train_lista
        self.files_B = train_listb
        # self.files_A = sorted(glob.glob("%s/A/*" % root))
        # self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        item_A = read_dicom(self.files_A[index % len(self.files_A)])
        item_A = self.transform(item_A.astype(np.float32))
        if self.unaligned:
            item_B = read_dicom(self.files_B[random.randint(0, len(self.files_B) - 1)])
            item_B = self.transform(item_B.astype(np.float32))
        else:
            item_B = read_dicom(self.files_B[index % len(self.files_B)])
            item_B = self.transform(item_B.astype(np.float32))
        return {'A': item_A, 'B': item_B,'A_path':self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class TestDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        f = open(root)
        train_lista = []
        train_listb = []
        for line in f.readlines():
            line = line.strip('\n')#.replace('CT_CTA','CT_CTAa')  # 直接将文件中按行读到list里，效果与方法2一样
            train_lista.append(line)
            train_listb.append(line.replace("SE0","SE1"))
        f.close()  # 关
        self.files_A=train_lista
        self.files_B = train_listb
        # self.files_A = sorted(glob.glob("%s/A/*" % root))
        # self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        item_A=read_dicom(self.files_A[index % len(self.files_A)])
        item_A= self.transform(item_A.astype(np.float32))
        if self.unaligned:
            item_B= read_dicom(self.files_B[random.randint(0, len(self.files_B) - 1)])
            item_B = self.transform(item_B.astype(np.float32))
        else:
            item_B = read_dicom(self.files_B[index % len(self.files_B)])
            item_B = self.transform(item_B.astype(np.float32))
        return {'A': item_A, 'B': item_B,'A_path':self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset_x(Dataset):
    def __init__(self, root, count=None, transforms_1=None, transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        f = open(root)  # test train train_np abd_train
        train_lista = []
        train_listb = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            train_lista.append(line)
            train_listb.append(line.replace("SE0", "SE1"))
        f.close()  # 关
        fd1 = open(root.replace('train', 'traind1'))
        for line in fd1.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            train_lista.append(line)
            train_listb.append(line.replace("SE0", "SE1"))
        fd1.close()  # 关
        # ind=np.random.randint(-6,6)+7
        for i in range(np.random.randint(3,6)):#[4 10]-->3 epoch效果不错.model5 [2 7]
            fd2 = open(root.replace('train', 'traind2'))
            for line in fd2.readlines():
                line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
                train_lista.append(line)
                train_listb.append(line.replace("SE0", "SE1"))
            fd2.close()  # 关
        self.files_A = sorted(train_lista)
        self.files_B = sorted(train_listb)
        # self.files_A = sorted(glob.glob("%s/A/*" % root))
        # self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        a1,a2=read_ori_w(self.files_A[index % len(self.files_A)])#index % len(self.files_A) 50
        item_A1 = self.transform1(a1.astype(np.float32))
        item_A2 = self.transform1(a2.astype(np.float32))
        # random.seed(seed)
        b1, b2 = read_ori_w(self.files_B[index % len(self.files_A)])  # index % len(self.files_A) 50
        item_B1 = self.transform2(b1.astype(np.float32))
        item_B2 = self.transform2(b2.astype(np.float32))
        # a=item_B.max()
        # b=item_B.min()
        # print(item_B)

        return {'A1': item_A1,'A2': item_A2, 'B1': item_B1,'B2': item_B2}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ValDataset_x(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        f = open(root)#test_np abd_val
        train_lista = []
        train_listb = []
        for line in f.readlines():
            line = line.strip('\n')
            train_lista.append(line)
            train_listb.append(line.replace("SE0", "SE1"))
        f.close()  # 关
        self.files_A = train_lista
        self.files_B = train_listb
        # self.files_A = sorted(glob.glob("%s/A/*" % root))
        # self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        a1, a2 = read_ori_w(self.files_A[index % len(self.files_A)])  # index % len(self.files_A) 50
        item_A1 = self.transform(a1.astype(np.float32))
        item_A2 = self.transform(a2.astype(np.float32))
        # random.seed(seed)
        b1, b2= read_ori_w(self.files_B[index % len(self.files_B)])  # self.files_B[index % len(self.files_B)
        item_B1 = self.transform(b1.astype(np.float32))
        item_B2 = self.transform(b2.astype(np.float32))
        return {'A1': item_A1,'A2': item_A2, 'B1': item_B1,'B2': item_B2,'A_path': self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class TestDataset_x(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        f = open(root)#test_np abd_val
        train_lista = []
        train_listb = []
        for line in f.readlines():
            line = line.strip('\n')#.replace('CT_CTA','CT_CTAa')   # ANTSY配准
            train_lista.append(line)
            train_listb.append(line.replace("SE0", "SE1"))
        f.close()  # 关
        self.files_A = train_lista
        self.files_B = train_listb

    def __getitem__(self, index):
        a1, a2= read_ori_w(self.files_A[index % len(self.files_A)])  # index % len(self.files_A) 50
        item_A1 = self.transform(a1.astype(np.float32))
        item_A2 = self.transform(a2.astype(np.float32))
        # random.seed(seed)
        b1, b2 = read_ori_w(self.files_B[index % len(self.files_B)])  # self.files_B[index % len(self.files_B)
        item_B1 = self.transform(b1.astype(np.float32))
        item_B2 = self.transform(b2.astype(np.float32))
        return {'A1': item_A1,'A2': item_A2, 'B1': item_B1,'B2': item_B2,'A_path': self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))