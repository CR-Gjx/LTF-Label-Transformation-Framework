import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import numpy as np
import model_resnet32
from spectral_norm import SpectralNorm
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self,nz, ngf, nc):
        super(generator, self).__init__()
        self.embed = (nn.Embedding(10, nz))
        # self.fc1 = nn.Linear(10, nz)
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 1, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf * 1)

        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.fc1 = (nn.Linear(2*nz,256))
        self.fc2 = (nn.Linear(256,512))
        self.fc3 = (nn.Linear(512,1024))
        self.fc4 = (nn.Linear(1024,1024))
        # self.fc5 = (nn.Linear(2048,1024))
        self.batchnorm1 = (nn.BatchNorm1d(256))
        self.batchnorm2 = (nn.BatchNorm1d(512))
        self.batchnorm3 = (nn.BatchNorm1d(1024))
        self.batchnorm4 = (nn.BatchNorm1d(1024))
        self.__initialize_weights()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        if len(label.size()) == 2:
            label_emb = label.matmul(self.embed.weight)
        else:
            label_emb = self.embed(label)
        input = torch.cat([label_emb,input],dim=1)
        x = self.fc1(input)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = torch.tanh(x)
        output = output.view(-1, 1, 32,32)
        return output

        # return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class generator_fm(nn.Module):
    # initializers
    def __init__(self,nz, ngf, nc):
        super(generator_fm, self).__init__()
        self.embed = (nn.Embedding(10, nz))
        # self.fc1 = nn.Linear(10, nz)
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 1, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf * 1)

        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.__initialize_weights()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        if len(label.size()) == 2:
            # print("sd")
            label_emb = label.matmul(self.embed.weight)
        else:
            # print("dsa")
            label_emb = self.embed(label)
        # label_emb = label_emb.type(torch.cuda.LongTensor)
        input = torch.mul(input,label_emb)
        # input = torch.cat([label_emb,input],dim=1)
        x = input.view(input.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv5(x)
        output = torch.tanh(x)
        output = output.view(-1, 1, 32,32)
        return output

        # return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class discriminator_source(nn.Module):
    # initializers
    def __init__(self,  ndf, nc, num_classes=10):
        super(discriminator_source, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)

        self.conv3 = nn.Conv2d(ndf , ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.sigmoid = nn.Sigmoid()

        self.fc1 = SpectralNorm(nn.Linear(32*32,512))
        self.fc2 = SpectralNorm(nn.Linear(512,128))
        self.fc3 = SpectralNorm(nn.Linear(128,32))
        # self.fc4 = (nn.Linear(64,32))

        self.c = SpectralNorm(nn.Linear(32,10))
        self.mi = SpectralNorm(nn.Linear(32,10))
        self.fc4 = SpectralNorm(nn.Linear(32,1))
        self.batch_norm1 = (nn.BatchNorm1d(512))
        self.batch_norm2 = (nn.BatchNorm1d(128))
        self.batch_norm3 = (nn.BatchNorm1d(32))
        self.__initialize_weights()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        input = input.type(torch.cuda.FloatTensor)
        input = input.view(-1, 1024)
        x = self.fc1(input)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = self.batch_norm3(x)
        x = F.relu(x)

        c = self.c(x)
        mi = self.mi(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x,c,mi#c.squeeze(1),x

        # return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class discriminator_source_fm(nn.Module):
    # initializers
    def __init__(self,  ndf, nc, num_classes=10):
        super(discriminator_source_fm, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1))

        self.conv3 = SpectralNorm(nn.Conv2d(ndf , ndf * 4, 4, 2, 1))
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1))
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = SpectralNorm(nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0))
        self.gan_linear = SpectralNorm(nn.Linear(ndf * 1, 1))
        self.aux_linear = SpectralNorm(nn.Linear(ndf * 1, num_classes))
        self.mi_linear = SpectralNorm(nn.Linear(ndf * 1, num_classes))

        self.sigmoid = nn.Sigmoid()
        # self.__initialize_weights()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        input = input.type(torch.cuda.FloatTensor)
        x = self.conv1(input)
        x = self.lrelu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.lrelu(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)
        mi = self.mi_linear(x)
        s = self.gan_linear(x)
        s = self.sigmoid(s)
        return s.squeeze(1),c,mi#c.squeeze(1),x

        # return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
class feature_extractor_fm(nn.Module):
    # initializers
    def __init__(self,  ndf, nc, num_classes=10,feature_num=10):
        super(feature_extractor_fm, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)

        self.conv3 = nn.Conv2d(ndf , ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, feature_num)
        self.mi_linear = nn.Linear(ndf * 1, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        input = input.type(torch.cuda.FloatTensor)
        x = self.conv1(input)
        x = self.lrelu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.lrelu(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)
        return c#c.squeeze(1),x

        # return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class feature_extractor(nn.Module):
    # initializers
    def __init__(self,  ndf, nc, num_classes=10):
        super(feature_extractor, self).__init__()
        self.ndf = ndf
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

        self.D_in = 1024
        self.H = ndf
        self.D_out = 10
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H),
            torch.nn.ReLU(),
        )
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = input.view(-1, self.D_in)
        x = self.model(x)

        return x

        # return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class discriminator_target(nn.Module):
    # initializers
    def __init__(self, d=128,img_shape=[1,1,1]):
        super(discriminator_target, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(int(d/2), d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # print(x)
        x = F.sigmoid(self.conv4(x))
        return x

class D_target_classifier(nn.Module):
    def __init__(self,d,num_classes,args):
        super(D_target_classifier, self).__init__()
        self.fc1 = nn.Linear(d, num_classes)
        # self.fc2 = nn.Linear(128,num_classes)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self,input):
        # x = F.leaky_relu(self.fc1(input),0.2)
        # x = F.leaky_relu(self.fc2(x),0.2)
        x = self.fc1(input)
        return x

class D_target_distribution(nn.Module):
    def __init__(self,d,num_class=10,drop_out = 0.0):
        super(D_target_distribution, self).__init__()
        self.fc1 = nn.Linear(d, num_class)
        self.fc3 = nn.Linear(num_class,1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self,input):
        x = F.relu((self.fc1(input)))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

class Transform_prob(nn.Module):
    def __init__(self,d,num_class=10,drop_out = 0.0):
        super(Transform_prob, self).__init__()
        # self.drop_out = drop_out
        self.fc1 = nn.Linear(d, num_class)
        self.batch_norm1 = nn.BatchNorm1d(num_class)
        self.fc3 = nn.Linear(num_class,1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self,input):
        x = F.relu((self.fc1(input)))

        x = self.fc3(x)
        x = F.relu(x)
        # print(x)
        return x

class Transform_reinforce(nn.Module):
    def __init__(self,d,num_class):
        super(Transform_reinforce, self).__init__()
        self.fc1 = nn.Linear(d, num_class)
        self.fc2 = nn.Linear(num_class,num_class)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self,input):
        x = F.relu(self.fc1(input))
        x = F.softmax(x,dim=1)
        # print(x)
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.aux_linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)
        self.gan_linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 1)

        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)

        s = self.gan_linear(out)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)


def DPN26():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)


# def test():
#     net = DPN92()
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y)

# test()
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,feature_num = 10, in_channels = 3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.aux_linear = nn.Linear(512*block.expansion, feature_num)
        self.gan_linear = nn.Linear(512*block.expansion, 1)

        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feature = self.aux_linear(out)
        # c = self.aux_linear(out)
        # s = self.gan_linear(out)
        # s = self.sigmoid(s)
        # return s.squeeze(1), feature#c.squeeze(1),out
        return feature#c.squeeze(1),out


def ResNet18(feature_num = 10, in_channels = 3):
    return ResNet(BasicBlock, [2,2,2,2],feature_num=feature_num,in_channels=in_channels)

def ResNet34(feature_num = 10, in_channels = 3):
    return ResNet(BasicBlock, [3,4,6,3],feature_num=feature_num,in_channels=in_channels)

def ResNet50(feature_num = 10, in_channels = 3):
    return ResNet(Bottleneck, [3,4,6,3],feature_num=feature_num,in_channels=in_channels)

def ResNet101(feature_num = 10, in_channels = 3):
    return ResNet(Bottleneck, [3,4,23,3],feature_num=feature_num,in_channels=in_channels)

def ResNet152(feature_num = 10, in_channels = 3):
    return ResNet(Bottleneck, [3,8,36,3],feature_num=feature_num,in_channels=in_channels)
