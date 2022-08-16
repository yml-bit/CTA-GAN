import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
import functools
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torchvision.transforms.functional as tf

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        assert torch.cuda.is_available(), "Currently, we only support CUDA version"
        device = (f'cuda:{"0"}' if torch.cuda.is_available() else 'cpu')
        self.device =device# opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        # x=F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        # print(x.size())
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 4))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))

                # x=model(res[-1])
                # x=F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
                # print(x.size())
                # res.append(x)
            return res[1:]
        else:
            # print('yml')
            return self.model(input)
            # x=self.model(input)
            # return  F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class Discriminator_m(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False),
                 use_sigmoid=False, num_D=1, getIntermFeat=True):#False True
        super(Discriminator_m, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:  # 各层输出
            result = [input]
            for i in range(len(model)):
                x = model[i](result[-1])
                result.append(x)
            return result[1:]
        else:
            x = model(input)
            return [x]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            s = input_downsampled.size()[2]
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            x = self.singleD_forward(model, input_downsampled)
            result.append(x)
            if i != (num_D - 1):
                # input_downsampled = self.downsample(input_downsampled)
                input_downsampled = tf.center_crop(input_downsampled, int(s/2))#int(s/2)
        # print(len(result[0]))
        # plt.subplot(2, 2, 2)
        # plt.imshow(np.squeeze(input_downsampled)* 255, cmap='gray')  # ,vmin=0,vmax=255
        # plt.show()
        return result

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.Tensor):#torch.cuda.FloatTensor
        super(GANLoss, self).__init__()
        self.target_real = Variable(tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(tensor(1, 1).fill_(0.0), requires_grad=False)
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):#input[0]
            loss = 0
            #print(len(input))
            w=[1.8,0.2]
            # w = [1, 1]
            i=0
            for input_i in input:
                #print(len(input))
                x = input_i[-1]
                pred= F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
                if target_is_real:
                    loss += self.loss(pred, self.target_real)*w[i]
                else:
                    loss += self.loss(pred, self.target_fake)*w[i]
                i=i+1
            return loss
        else:
            x=input[-1]
            pred = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
            if target_is_real:
                loss= self.loss(pred, self.target_real)
            else:
                loss= self.loss(pred, self.target_fake)
            return loss

if  __name__=='__main__':
    input1=torch.Tensor(np.random.rand(1,1,512,512))#2,192,192,4
    input2=torch.Tensor(np.random.rand(1,1,512,512))#2,192,192,4
    # a=torch.where((input1>0)&(input1<0.4),2,1)
    # b=1
    # model = Generator_x(1, 1)
    # model(input1)
    # model = Generator_xx(1, 1)
    # model(input1)
    # model(input1,input2)
    model = Discriminator_m(1)
    model.forward(input1)
    model.summary()
    a=1
