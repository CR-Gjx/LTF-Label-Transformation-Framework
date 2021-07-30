import os
import torch.utils.data as data
import torch
from scipy.io import loadmat
from os.path import join
import numpy as np
from model import y_to_x,random_shift,make_moons
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off
class Dataset_simul(data.Dataset):
    def __init__(self, specs,g_func):
        super(Dataset_simul, self).__init__()
        self.root = specs['root']
        self.id = specs['id']
        self.num_class = specs['num_class']
        self.sample_size = specs['num_train']
        self.dim = specs['dim']
        self.ratio = specs['ratio']
        self.train = specs['train']
        self.y_dim = specs['y_dim']
        self.x_dim = specs['x_dim']
        self.r = specs['r']
        self.rn = specs['rn']
        self.y2x = g_func

        self.shift_func = random_shift(input_size=self.y_dim+self.x_dim)
        full_path = join(self.root, str(self.x_dim)+str(self.y_dim)+str(self.id)+str(self.rn))

        if os.path.isfile(full_path + '.npz') and not self.train:
            npzfile = np.load(full_path + '.npz',allow_pickle=True)
            x = npzfile['x']
            y = npzfile['y']
            xt = npzfile['xt']
            yt = npzfile['yt']
            self.num = len(xt)
            self.data = xt
            self.labels = yt
        else:
            # if data not exist, generate data
            if self.id == 1:
                mean2 = 0.707*self.r
            elif self.id == 2:
                mean2 = -0.707*self.r
            mean1 = 0
            cov = 1
            # source domain
            # y = np.random.normal(mean1,cov,self.sample_size).reshape((-1,1))
            y = np.linspace(-1*self.r, 1*self.r, num=int(self.sample_size)).reshape((-1,1))

            # y = np.linspace(-15, 15, num=int(self.sample_size )).reshape((-1, 1))
            # y_max = 10  # exclude extreme point
            # y_min = -10  # exclude extreme point
            # y = np.linspace(y_min, y_max, num=int(self.sample_size )).reshape((-1, 1))
            if self.id ==3:
                w = [0.5,0.5]
                yt = []
                for i in range(int(self.sample_size/2)):
                    if np.random.choice(np.arange(2), p=w) == 0:
                        yt.append(np.random.normal(0.707*self.r,cov/self.ratio,1))
                    else:
                        yt.append(np.random.normal(-0.707*self.r, cov / self.ratio, 1))
                yt = np.concatenate(yt).reshape((-1,1))
            elif self.id == 4:
                y_max = 1*self.r  # exclude extreme point
                y_min = -1*self.r  # exclude extreme point
                yt = torch.randn((int(self.sample_size/2), 1))
                noise_T = torch.randn((yt.shape[0], 1))
                yt = self.shift_func(yt,noise_T)
                yt = (y_max-y_min)*yt + y_min
                yt = yt.detach().numpy()
                # xt = self.y2x(yt)
            else:
                yt = np.random.normal(mean2,cov/self.ratio,int(self.sample_size/2)).reshape((-1,1))
            x = self.y2x(y,noise=0.05*self.r,r = self.r)
            xt = self.y2x(yt,noise=0.05*self.r,r =self.r)
            np.savez(full_path, x=x, y=y, xt=xt, yt=yt)
            npzfile = np.load(full_path + '.npz')
            x = npzfile['x']
            y = npzfile['y']
            xt = npzfile['xt']
            yt = npzfile['yt']
            if self.train:
                self.num = len(x)
                self.data = x
                self.labels = y
                # print(self.data.shape)
            else:
                self.num = len(xt)
                self.data = xt
                self.labels = yt

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        # label = torch.LongTensor([np.int64(labels)])
        return img, labels

    def __len__(self):
        return self.num
