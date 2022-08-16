import numpy as np
import matplotlib.pyplot as plt
import pydicom
import shutil
import random
import torch.nn as nn
import os
import glob
import SimpleITK as sitk
import ants
import dicom2nifti
import cv2

#get dicom list
def get_neck_list():
    path="../../../data/CT_CTA/neck/"#neckk
    catch="../../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)
    path_list=[]
    for root, dirs, files in os.walk(path, topdown=False):
        if"SE0" in root :
            path_list.append(root)
    random.shuffle(path_list)
    file_path_list=[]
    # f = open("error.txt", "w")
    f1 = open("train1.txt", "w")#564
    f2 = open("val1.txt", "w")#187
    f3 = open("test1.txt", "w")#188
    i=0
    id=[]
    for sub_path in path_list:
        i=i+1
        try:
            aa=os.path.join(sub_path.split('SE0')[0], 'SE2')
            if os.path.isdir(aa):
                shutil.rmtree(aa)
                # os.remove(aa)
            data_files = os.listdir(sub_path)
            traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        except:
            continue
        if len(traget_files)!=len(data_files):
            if len(data_files)%len(traget_files)==0:
                os.rename(sub_path, sub_path.replace("SE0", "SE33"))#0 3
                os.rename(sub_path.replace("SE0", "SE1"), sub_path)#1 0
                os.rename(sub_path.replace("SE0", "SE33"), sub_path.replace("SE0", "SE1"))#3 1
            for j in range(len(traget_files)):
                file_path = os.path.join(sub_path, traget_files[j])  #
                flag=os.path.exists(file_path)#只要平扫没有的则删除
                if not flag:
                    os.remove(file_path.replace("SE0", "SE1"))
        data_files = os.listdir(sub_path)
        traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        if len(data_files)==0:
            shutil.rmtree(sub_path.split("/ST0")[0])
        if len(traget_files) != len(data_files):
            # os.remove(sub_path.split("/ST0")[0])
            shutil.rmtree(sub_path.split("/ST0")[0])
            print(sub_path+':len(data_files) != len(traget_files)')
            continue

        for j in range(len(data_files)):
            #注释部分用于挑选特定部分数据，如头部数据
            # a = int(data_files[j].split("M")[1])
            # if a > int(len(data_files)/2+75) and (a+1)<len(data_files):  # 只挑选头部数据
            #     continue
            try:
                file_path = os.path.join(sub_path, data_files[j])  # 直接将文件中按行读到list里，效果与方法2一样
                dsA = pydicom.dcmread(file_path, force=True)  # 读取头文件
                dsB = pydicom.dcmread(file_path.replace("SE0", "SE1"), force=True)  # 读取头文件
                if dsA.PatientID not in id:
                    id.append(dsA.PatientID)
                elif id[-1]!=dsA.PatientID:
                    shutil.rmtree(sub_path.split("/ST0")[0])
                    # print("repeat id:", sub_path)
                    break
                # if 'Head' in dsA.ProtocolName:  # check ProtocolName BodyPartExamined
                #     print("dsA.BodyPartExamined==head:", file_path)
                if dsA.RescaleIntercept != -1024 or dsB.RescaleIntercept != -1024:  # check
                    print("dsA.RescaleIntercept!=-1024:", file_path)
                    continue
                if dsA.AccessionNumber != dsB.AccessionNumber:
                    print("dsA.AccessionNumber != dsB.AccessionNumber:", file_path)
                    continue
                if dsA.SliceLocation != dsB.SliceLocation:
                    print("dsA.SliceLocation != dsB.SliceLocation:", file_path)
                    continue
                if "C+" not in dsA.SeriesDescription and "C+" in dsB.SeriesDescription:
                    a = 1
                elif "C+" not in dsB.SeriesDescription and "C+" in dsA.SeriesDescription:
                    file_pathb = file_path
                    file_patha = file_path.replace("SE0", "SE1")
                    catch_path = os.path.join(catch, file_patha.split("/")[-1])
                    shutil.move(file_patha, catch_path)
                    shutil.move(file_pathb, file_patha)
                    shutil.move(catch_path, file_pathb)
                    print("exchange path:", file_patha)
                else:
                    print("CTA not in dsA/dsB.SeriesDescription path:", file_path)
                    continue
            except:
                continue
            if i <= int(len(path_list) * 0.6):#564
                f1.writelines(file_path + "\n")
            elif i>int(len(path_list) * 0.6) and i<int(len(path_list) * 0.8):#187
                f2.writelines(file_path + "\n")
            else:#188
                f3.writelines(file_path + "\n")
    f1.close()
    f1.close()
    f2.close()
    print('id numbers:',len(id))

def get_abd_list():
    path="../../../data/CT_CTA/abd/"#neckk
    catch="../../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)
    path_list=[]
    for root, dirs, files in os.walk(path, topdown=False):
        if"SE0" in root :
            path_list.append(root)
    random.shuffle(path_list)

    f1 = open("train2.txt", "w")#564
    f2 = open("val2.txt", "w")#187
    f3 = open("test2.txt", "w")#188
    i=0
    id=[]
    for sub_path in path_list:
        i=i+1
        try:
            aa=os.path.join(sub_path.split('SE0')[0], 'SE2')
            if os.path.isdir(aa):
                shutil.rmtree(aa)
                # os.remove(aa)
            data_files = os.listdir(sub_path)
            traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        except:
            continue
        if len(traget_files)!=len(data_files):
            if len(data_files)%len(traget_files)==0:
                os.rename(sub_path, sub_path.replace("SE0", "SE33"))#0 3
                os.rename(sub_path.replace("SE0", "SE1"), sub_path)#1 0
                os.rename(sub_path.replace("SE0", "SE33"), sub_path.replace("SE0", "SE1"))#3 1

            for j in range(len(traget_files)):
                file_path = os.path.join(sub_path, traget_files[j])  #
                flag=os.path.exists(file_path)#只要平扫没有的则删除
                if not flag:
                    os.remove(file_path.replace("SE0", "SE1"))
        data_files = os.listdir(sub_path)
        traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        if len(traget_files) != len(data_files):
            # os.remove(sub_path.split("/ST0")[0])
            shutil.rmtree(sub_path.split("/ST0")[0])
            print(sub_path+':len(data_files) != len(traget_files)')
            continue

        for j in range(len(data_files)):
            #注释部分用于挑选特定部分数据，如头部数据
            # a = int(data_files[j].split("M")[1])
            # if a < 40 or a > 70:  # 只挑选头部数据
            #     continue
            try:
                file_path = os.path.join(sub_path, data_files[j])  # 直接将文件中按行读到list里，效果与方法2一样
                dsA = pydicom.dcmread(file_path, force=True)  # 读取头文件
                dsB = pydicom.dcmread(file_path.replace("SE0", "SE1"), force=True)  # 读取头文件
                if dsA.PatientID not in id:
                    id.append(dsA.PatientID)
                elif id[-1]!=dsA.PatientID:
                    shutil.rmtree(sub_path.split("/ST0")[0])
                    # print("repeat id:", sub_path)
                    break
                if dsA.RescaleIntercept != -1024 or dsB.RescaleIntercept != -1024:  # check
                    print("dsA.RescaleIntercept!=-1024:", file_path)
                    continue
                if dsA.AccessionNumber != dsB.AccessionNumber:
                    print("dsA.SliceLocation != dsB.SliceLocation:", file_path)
                    continue
                if dsA.SliceLocation != dsB.SliceLocation:
                    print("dsA.SliceLocation != dsB.SliceLocation:", file_path)
                    continue
                if "C+" not in dsA.SeriesDescription and "C+" in dsB.SeriesDescription:
                    a = 1
                elif "C+" not in dsB.SeriesDescription and "C+" in dsA.SeriesDescription:
                    file_pathb = file_path
                    file_patha = file_path.replace("SE0", "SE1")
                    catch_path = os.path.join(catch, file_patha.split("/")[-1])
                    shutil.move(file_patha, catch_path)
                    shutil.move(file_pathb, file_patha)
                    shutil.move(catch_path, file_pathb)
                    print("exchange path:", file_patha)
            except:
                continue
                # else:
                #     print("CTA not in dsA/dsB.SeriesDescription path:", file_path)
                #     continue
            if i <= int(len(path_list) * 0.6):#1+563
                f1.writelines(file_path + "\n")
            elif i>int(len(path_list) * 0.6) and int(len(path_list) * 0.8):
                f2.writelines(file_path + "\n")
            else:
                f3.writelines(file_path + "\n")
    f1.close()
    f1.close()
    f2.close()
    print('id numbers:',len(id))

#get specified number of case
def get_necktest_list():
    path="../../../data/disease/neck/"#neckk
    catch="../../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)
    path_list=[]
    for root, dirs, files in os.walk(path, topdown=False):
        if"SE0" in root :
            path_list.append(root)
    random.shuffle(path_list)
    file_path_list=[]
    # f = open("error.txt", "w")
    f1 = open("dtest1.txt", "w")#564
    i=0
    id=[]
    for sub_path in path_list:
        i=i+1
        try:
            aa=os.path.join(sub_path.split('SE0')[0], 'SE2')
            if os.path.isdir(aa):
                shutil.rmtree(aa)
                # os.remove(aa)
            data_files = os.listdir(sub_path)
            traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        except:
            continue
        if len(traget_files)!=len(data_files):
            if len(data_files)%len(traget_files)==0:
                os.rename(sub_path, sub_path.replace("SE0", "SE33"))#0 3
                os.rename(sub_path.replace("SE0", "SE1"), sub_path)#1 0
                os.rename(sub_path.replace("SE0", "SE33"), sub_path.replace("SE0", "SE1"))#3 1
            for j in range(len(traget_files)):
                file_path = os.path.join(sub_path, traget_files[j])  #
                flag=os.path.exists(file_path)#只要平扫没有的则删除
                if not flag:
                    os.remove(file_path.replace("SE0", "SE1"))
        data_files = os.listdir(sub_path)
        traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        if len(data_files)==0:
            shutil.rmtree(sub_path.split("/ST0")[0])
        if len(traget_files) != len(data_files):
            # os.remove(sub_path.split("/ST0")[0])
            shutil.rmtree(sub_path.split("/ST0")[0])
            print(sub_path+':len(data_files) != len(traget_files)')
            continue

        for j in range(len(data_files)):
            #注释部分用于挑选特定部分数据，如头部数据
            # a = int(data_files[j].split("M")[1])
            # if a > int(len(data_files)/2+75) and (a+1)<len(data_files):  # 只挑选头部数据
            #     continue
            try:
                file_path = os.path.join(sub_path, data_files[j])  # 直接将文件中按行读到list里，效果与方法2一样
                dsA = pydicom.dcmread(file_path, force=True)  # 读取头文件
                dsB = pydicom.dcmread(file_path.replace("SE0", "SE1"), force=True)  # 读取头文件
                if dsA.PatientID not in id:
                    id.append(dsA.PatientID)
                elif id[-1]!=dsA.PatientID:
                    shutil.rmtree(sub_path.split("/ST0")[0])
                    # print("repeat id:", sub_path)
                    break
                # if 'Head' in dsA.ProtocolName:  # check ProtocolName BodyPartExamined
                #     print("dsA.BodyPartExamined==head:", file_path)
                if dsA.RescaleIntercept != -1024 or dsB.RescaleIntercept != -1024:  # check
                    print("dsA.RescaleIntercept!=-1024:", file_path)
                    continue
                if dsA.AccessionNumber != dsB.AccessionNumber:
                    print("dsA.AccessionNumber != dsB.AccessionNumber:", file_path)
                    continue
                if dsA.SliceLocation != dsB.SliceLocation:
                    print("dsA.SliceLocation != dsB.SliceLocation:", file_path)
                    continue
                if "C+" not in dsA.SeriesDescription and "C+" in dsB.SeriesDescription:
                    a = 1
                elif "C+" not in dsB.SeriesDescription and "C+" in dsA.SeriesDescription:
                    file_pathb = file_path
                    file_patha = file_path.replace("SE0", "SE1")
                    catch_path = os.path.join(catch, file_patha.split("/")[-1])
                    shutil.move(file_patha, catch_path)
                    shutil.move(file_pathb, file_patha)
                    shutil.move(catch_path, file_pathb)
                    print("exchange path:", file_patha)
                else:
                    print("CTA not in dsA/dsB.SeriesDescription path:", file_path)
                    continue
            except:
                continue

            f1.writelines(file_path + "\n")
    f1.close()
    print('id numbers:',len(id))

def get_abdtest_list():
    path="../../../data/disease/abd/"#neckk
    catch="../../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)
    path_list=[]
    for root, dirs, files in os.walk(path, topdown=False):
        if"SE0" in root :
            path_list.append(root)
    random.shuffle(path_list)

    f1 = open("dtest2.txt", "w")#564
    i=0
    id=[]
    for sub_path in path_list:
        i=i+1
        try:
            aa=os.path.join(sub_path.split('SE0')[0], 'SE2')
            if os.path.isdir(aa):
                shutil.rmtree(aa)
                # os.remove(aa)
            data_files = os.listdir(sub_path)
            traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        except:
            continue
        if len(traget_files)!=len(data_files):
            if len(data_files)%len(traget_files)==0:
                os.rename(sub_path, sub_path.replace("SE0", "SE33"))#0 3
                os.rename(sub_path.replace("SE0", "SE1"), sub_path)#1 0
                os.rename(sub_path.replace("SE0", "SE33"), sub_path.replace("SE0", "SE1"))#3 1

            for j in range(len(traget_files)):
                file_path = os.path.join(sub_path, traget_files[j])  #
                flag=os.path.exists(file_path)#只要平扫没有的则删除
                if not flag:
                    os.remove(file_path.replace("SE0", "SE1"))
        data_files = os.listdir(sub_path)
        traget_files = os.listdir(sub_path.replace("SE0", "SE1"))
        if len(traget_files) != len(data_files):
            # os.remove(sub_path.split("/ST0")[0])
            shutil.rmtree(sub_path.split("/ST0")[0])
            print(sub_path+':len(data_files) != len(traget_files)')
            continue

        for j in range(len(data_files)):
            try:
                file_path = os.path.join(sub_path, data_files[j])  # 直接将文件中按行读到list里，效果与方法2一样
                dsA = pydicom.dcmread(file_path, force=True)  # 读取头文件
                dsB = pydicom.dcmread(file_path.replace("SE0", "SE1"), force=True)  # 读取头文件
                if dsA.PatientID not in id:
                    id.append(dsA.PatientID)
                elif id[-1]!=dsA.PatientID:
                    shutil.rmtree(sub_path.split("/ST0")[0])
                    # print("repeat id:", sub_path)
                    break
                if dsA.RescaleIntercept != -1024 or dsB.RescaleIntercept != -1024:  # check
                    print("dsA.RescaleIntercept!=-1024:", file_path)
                    continue
                if dsA.AccessionNumber != dsB.AccessionNumber:
                    print("dsA.SliceLocation != dsB.SliceLocation:", file_path)
                    continue
                if dsA.SliceLocation != dsB.SliceLocation:
                    print("dsA.SliceLocation != dsB.SliceLocation:", file_path)
                    continue
                if "C+" not in dsA.SeriesDescription and "C+" in dsB.SeriesDescription:
                    a = 1
                elif "C+" not in dsB.SeriesDescription and "C+" in dsA.SeriesDescription:
                    file_pathb = file_path
                    file_patha = file_path.replace("SE0", "SE1")
                    catch_path = os.path.join(catch, file_patha.split("/")[-1])
                    shutil.move(file_patha, catch_path)
                    shutil.move(file_pathb, file_patha)
                    shutil.move(catch_path, file_pathb)
                    print("exchange path:", file_patha)
            except:
                continue
                # else:
                #     print("CTA not in dsA/dsB.SeriesDescription path:", file_path)
                #     continue
            f1.writelines(file_path + "\n")

    f1.close()
    print('id numbers:',len(id))

#对列表文件进行check
def statistic():
    catch="../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)
    f = open("traind.txt")  # test train！！  list  filter
    # fd1 = open("testd2.txt")  # test train！！  list  filter
    # fd2 = open("traind2.txt")  # test train！！  list  filter
    f1 = open("statistic.txt", "w")
    path_list = []
    for line in f.readlines():
        path_list.append(line)
    # for line in fd1.readlines():
    #     path_list.append(line)
    # for line in fd2.readlines():
    #     path_list.append(line)
    path_list.sort()
    path_list.sort(key=lambda x: (x.split('IM')[0],int(x.split('IM')[1])))
    gender={'male':0,'female':0}
    equip={}
    age=[]
    path='f'
    for line in path_list:
        try:
            file_path = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            if path==line.split('IM')[0]:
                # list.append(line)
                continue
            else:
                # list=[]
                path = line.split('IM')[0]
                dsA = pydicom.dcmread(file_path, force=True)  # 读取头文件
                ag = int(dsA.PatientAge[0:3])  # PatientAge
                age.append(ag)
                # dsA.BodyPartExamined
                if dsA.PatientSex == 'M':
                    gender['male'] += 1
                else:
                    gender['female'] += 1
                aa = dsA.Manufacturer.split(' ')[0]
                if aa not in equip:
                    # 'GE MEDICAL SYSTEMS'
                    equip.setdefault(aa.split(' ')[0], 1)  # 'GE MEDICAL SYSTEMS'
                else:
                    equip[aa] += 1
                # dsA.InstitutionName
        except:
            continue
            # newimg=np.random.rand(512,512)*4095
            # if dsA[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
            #     newimg = newimg.astype(np.uint16)  # newimg 是图像矩阵 ds是dcm
            # elif dsA[0x0028, 0x0100].value == 8:
            #     newimg = newimg.astype(np.uint8)
            # else:
            #     raise Exception("unknow Bits Allocated value in dicom header")
            # dsA.PixelData = newimg.tobytes()    # 替换矩阵
            # pydicom.dcmwrite('a', dsA)
            # dsA.PatientPosition

    agee=np.array(age)
    max=np.max(agee)
    min=np.min(agee)
    f1.writelines('slices numbers:'+str(len(path_list))+"\n")
    f1.writelines('average age:' + str(min+(max-min)/2) +'  difference age:'+str((max-min)/2) +"\n")
    f1.writelines('male :'+str(gender['male']) +'female :'+str(gender['female'])+"\n")
    for eq in equip:
        f1.writelines(eq+':'+str(equip[eq]) + "\n")
    f1.close()#{'GE': 54900, 'SIEMENS': 18705, 'Philips': 1491}
    print("finished!")

#按照不同设备的存放list清单
def make_equip_split():#
    catch = "../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)
    f = open("test_abd.txt")  # test train！！  list  filter    test_neck  test_abd
    ff = open("test2.txt", "w")  # test train！！  list  filter
    f1 = open("GE1.txt", "w")
    f2 = open("siemens1.txt", "w")
    f3 = open("philip1.txt", "w")
    path_list = []
    for line in f.readlines():
        path_list.append(line)
    path_list.sort()
    path_list.sort(key=lambda x: (x.split('IM')[0],int(x.split('IM')[1])))
    equips={}
    path=path_list[0].split('IM')[1]
    i=0
    for line in path_list:
        file_path = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
        dsA = pydicom.dcmread(file_path, force=True)  # 读取头文件
        equip = dsA.Manufacturer.split(' ')[0]
        if 'GE' in equip:
            f1.writelines(line)
        elif 'SIEMENS' in equip:
            f2.writelines(line)
        elif 'Philips' in equip:
            f3.writelines(line)

        if path!=line.split('IM')[0]:
            i=i+1
            if i>100:  #neck 100,abd 98
                break
            # path = path_list[0].split('IM')[0]#slices
            path = line.split('IM')[0]#patients
            list=os.listdir(path)
            if len(list)<30:# just exclude the os.listdir list,can't exclude the existing list
                continue
            if equip not in equips:
                # 'GE MEDICAL SYSTEMS'
                equips.setdefault(equip.split(' ')[0], 1)  # 'GE MEDICAL SYSTEMS'
            else:
                equips[equip] += 1
        ff.writelines(line)
    for eq in equips:
        print(eq+':'+str(equips[eq]) + "\n")
    f.close()
    ff.close()

#dicom to nii
def dcm2nii_sitk(path_read, path_save):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, path_save)

#image aligement using ANTS
def aligement():
    catch="../../../data/catch"
    f = open("dataa.txt")  # test train！！  list  filter
    path_list = []
    for line in f.readlines():
        path_list.append(line)
    path_list.sort()
    path_list.sort(key=lambda x: (x.split('IM')[0], int(x.split('IM')[1])))
    path = 'f'
    j=0
    for line in path_list:
        j=j+1
        # if j<30000:
        #     continue
        if j%1000==0:
            print('processed:',j)

        file_path = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
        if path==line.split('IM')[0]:
            dsa = pydicom.dcmread(file_path.replace('SE0', 'SE1'), force=True)  # 读取头文件
            se0newimg = f_img_arr[:, :, index - i]
            if dsa[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
                se0newimg = se0newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm uint16
            elif dsa[0x0028, 0x0100].value == 8:
                se0newimg = se0newimg.astype(np.int8)
            else:
                raise Exception("unknow Bits Allocated value in dicom header")
            # ds.dtype=int16
            dsa.PixelData = se0newimg.tobytes()  # 替换矩阵
            pydicom.dcmwrite(file_path, dsa)

            ds = pydicom.dcmread(file_path.replace('SE0', 'SE1'), force=True)  # 读取头文件
            newimg = warped_img_arr[:, :, index - i]
            if (index - i)==0:
                newimg = m_img[:, :, index - i]
            if ds[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
                newimg = newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm uint16
            elif ds[0x0028, 0x0100].value == 8:
                newimg = newimg.astype(np.int8)
            else:
                raise Exception("unknow Bits Allocated value in dicom header")
            # ds.dtype=int16
            ds.PixelData = newimg.tobytes()  # 替换矩阵
            pydicom.dcmwrite(file_path.replace('SE0', 'SE1'), ds)
            i=i+1
        else:
            # list=[]
            i=0
            file_path = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            path = line.split('IM')[0]
            if not os.path.isdir(catch):
                os.makedirs(catch)

            if len(os.listdir(path))<5:
                path='f'
                continue
            # dicom2nifti.convert_directory(path, catch, compression=True, reorient=True)
            # dicom2nifti.convert_directory(path.replace('SE0', 'SE1'), catch, compression=True, reorient=True)
            se0output = os.path.join(catch, '1.nii.gz')
            dcm2nii_sitk(path, se0output)
            f_img = ants.image_read(se0output)

            se1output = os.path.join(catch, '2.nii.gz')
            dcm2nii_sitk(path.replace('SE0', 'SE1'), se1output)
            m_img = ants.image_read(se1output)

            shutil.rmtree(catch)
            '''
            ants.registration()函数的返回值是一个字典：
                warpedmovout: 配准到fixed图像后的moving图像
                warpedfixout: 配准到moving图像后的fixed图像
                fwdtransforms: 从moving到fixed的形变场
                invtransforms: 从fixed到moving的形变场

            type_of_transform参数的取值可以为：
                Rigid：刚体
                Affine：仿射配准，即刚体+缩放
                ElasticSyN：仿射配准+可变形配准，以MI为优化准则，以elastic为正则项
                SyN：仿射配准+可变形配准，以MI为优化准则
                SyNCC：仿射配准+可变形配准，以CC为优化准则
            '''
            f_img_arr = f_img.numpy(single_components=False).astype(np.int16)+1024
            # warped_img_arr.transpose((1,0,2))
            f_img_arr = np.rot90(f_img_arr, -1)
            f_img_arr = np.flip(f_img_arr, axis=1)
            dsa = pydicom.dcmread(file_path, force=True)  # 读取头文件
            index=f_img_arr.shape[2]-1
            se0newimg=f_img_arr[:,:,index-i]
            if dsa[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
                se0newimg = se0newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm uint16
            elif dsa[0x0028, 0x0100].value == 8:
                se0newimg = se0newimg.astype(np.int8)
            else:
                raise Exception("unknow Bits Allocated value in dicom header")
            # ds.dtype=int16
            dsa.PixelData = se0newimg.tobytes()  # 替换矩阵
            pydicom.dcmwrite(file_path, dsa)

            # 图像配准
            mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN')
            # 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
            warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                               interpolator="linear")
            # 将配准后图像的direction/origin/spacing和原图保持一致
            warped_img.set_direction(f_img.direction)
            warped_img.set_origin(f_img.origin)
            warped_img.set_spacing(f_img.spacing)

            # 将antsimage转化为numpy数组
            m_img = m_img.numpy(single_components=False).astype(np.int16)+1024
            m_img = np.rot90(m_img, -1)
            m_img = np.flip(m_img, axis=1)
            warped_img_arr = warped_img.numpy(single_components=False).astype(np.int16)+1024
            warped_img_arr = np.rot90(warped_img_arr, -1)
            ds = pydicom.dcmread(file_path.replace('SE0', 'SE1'), force=True)  # 读取头文件
            index=warped_img_arr.shape[2]-1
            newimg=warped_img_arr[:,:,index-i]
            if ds[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
                newimg = newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm uint16
            elif ds[0x0028, 0x0100].value == 8:
                newimg = newimg.astype(np.int8)
            else:
                raise Exception("unknow Bits Allocated value in dicom header")
            # ds.dtype=int16
            ds.PixelData = newimg.tobytes()  # 替换矩阵
            pydicom.dcmwrite(file_path.replace('SE0', 'SE1'), ds)
            i=i+1
            # 从numpy数组得到antsimage
            # img = ants.from_numpy(warped_img_arr, origin=None, spacing=None, direction=None, has_components=False,is_rgb=False)
            # img_name = os.path.join(catch, 'warped_img.nii.gz')
            # ants.image_write(warped_img, img_name)  # 图像的保存

#对列表排序
def sort():
    f = open("dtest2.txt")  # test train！！  list  filter
    f1 = open("dtest222.txt", 'w')  # test train！！  list  filter
    path_list=[]
    for line in f.readlines():
        path_list.append(line)
    path_list.sort()
    path_list.sort(key=lambda x: (x.split('IM')[0],int(x.split('IM')[1])))
    for name in path_list:
        try:
            dsA = pydicom.dcmread(name.strip('\n'), force=True)  # 读取头文件
            f1.writelines(name)
        except:
            a=1

#图像翻转，主要针对配准后的结果进行处理
def invert():
    catch="../../../data/catch"
    f = open("test.txt")  # test train！！  list  filter
    path_list = []
    for line in f.readlines():
        path_list.append(line)
    path_list.sort()
    path_list.sort(key=lambda x: (x.split('IM')[0], int(x.split('IM')[1])))
    path = 'f'
    j=0
    for line in path_list:
        line=line.strip('\n')
        dsa = pydicom.dcmread(line, force=True)  # 读取头文件
        data=(dsa.pixel_array)
        se0newimg =np.flip(data,axis=1)
        if dsa[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
            se0newimg = se0newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm uint16
        elif dsa[0x0028, 0x0100].value == 8:
            se0newimg = se0newimg.astype(np.int8)
        else:
            raise Exception("unknow Bits Allocated value in dicom header")
        # ds.dtype=int16
        dsa.PixelData = se0newimg.tobytes()  # 替换矩阵
        pydicom.dcmwrite(line, dsa)

        dsb = pydicom.dcmread(line.replace('SE0','SE1'), force=True)  # 读取头文件
        datab=(dsb.pixel_array)
        se1newimg =np.flip(datab,axis=1)
        if dsb[0x0028, 0x0100].value == 16:  # 如果dicom文件矩阵是16位格式
            se1newimg = se1newimg.astype(np.int16)  # newimg 是图像矩阵 ds是dcm uint16
        elif dsb[0x0028, 0x0100].value == 8:
            se1newimg = se1newimg.astype(np.int8)
        else:
            raise Exception("unknow Bits Allocated value in dicom header")
        # ds.dtype=int16
        dsb.PixelData = se1newimg.tobytes()  # 替换矩阵
        pydicom.dcmwrite(line.replace('SE0','SE1'), dsb)
        j=j+1
        if j%1000==0:
            print('processed:',j)

if __name__ == '__main__':
    get_neck_list()#
    get_abd_list()
    get_necktest_list()
    get_abdtest_list()
    # sort()
    # aligement()
    # filter1()
    # filter2()
    # sort()
    # invert()
    statistic()
    # get_test_list()
