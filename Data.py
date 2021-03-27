import numpy as np
import utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def generate_randomlabel(ep,num,sp=0):
    random_list = []
    while len(random_list) < num:
        x = np.random.randint(sp,ep,1)
        if x not in random_list:
            random_list.append(x)
    return random_list

def tweak_dist(X, y, num_labels, n, Py):
    shape = (n, *X.shape[1:])
    Xshift = np.zeros(shape)
    yshift = np.zeros(n, dtype=np.int8)

    # get indices for each label
    indices_by_label = Get_indices_by_label(y,num_labels)

    labels = np.argmax(
        np.random.multinomial(1, Py, n), axis=1)

    for i in range(n):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[labels[i]])
        Xshift[i] = X[idx]
        yshift[i] = y[idx]

    return Xshift, yshift

def Get_indices_by_label(y, num_labels):
    indices_by_label = [(y == k).nonzero()[0] for k in range(num_labels)]

    return indices_by_label

def Sample_batch(X, y, n, Py, indices_by_label):
    shape = (n, *X.shape[1:])
    X_batch = np.zeros(shape)
    y_batch = np.zeros(n, dtype=np.int8)
    labels = np.argmax(
        np.random.multinomial(1, Py, n), axis=1)
    for i in range(n):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[labels[i]])
        X_batch[i] = X[idx]
        y_batch[i] = y[idx]
    return X_batch, y_batch


def tweak_one(num_labels, knockout_label, p):
    # create Py
    # call down to tweak_dist
    Py = np.full(num_labels, (1. - p) / (num_labels - 1))
    Py[knockout_label] = p
    print(Py)
    return Py

def tweak_monority(num_labels, num_knock_lable, p = 0.001):
    # create Py
    # call down to tweak_dist
    knockout_labels = generate_randomlabel(num_labels, num_knock_lable)
    # l = len(knockout_labels)
    # print(l)
    Py = np.full(num_labels, (1. - p*num_knock_lable) / (num_labels - num_knock_lable))
    for knockout_label in knockout_labels:
        Py[knockout_label] = p
    print(Py)
    return Py

def preProcessData(p_Q,args,p_P = [.1, .1, .1, .1 ,.1 ,.1, .1, .1, .1, .1],alpha = 0.01,img_size=32,
                   batch_size = 128,num_train_samples=30000,num_test_samples = 10000,tweak_train= False,tweak_type = 0):
    # data_loader
    if args.dataset == "mnist":
        args.img_size = 32
        # img_shape= [1,args.img_size,args.img_size]

        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_loader = torch.utils.data.DataLoader(
            datasets.QMNIST('data', what='train', download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.QMNIST('data', what='test', download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            # transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == "cifar100":
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            # transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif args.dataset == "f-m":
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            # transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    for i, (x_, y_) in enumerate(train_loader):
        if i == 0:
            x_all, y_all = x_, y_
        else:
            x_all = np.concatenate((x_all, x_), axis=0)
            y_all = np.concatenate((y_all, y_), axis=0)
    for i, (x_, y_) in enumerate(test_loader):
        if i == 0:
            x_test, y_test = x_, y_
        else:
            x_test = np.concatenate((x_test, x_), axis=0)
            y_test = np.concatenate((y_test, y_), axis=0)
    num = 2
    n = x_all.shape[0]
    x_train, y_train = x_all, y_all
    x_val, y_val = x_all, y_all
    if tweak_train:
        x_train, y_train = tweak_dist(x_train, y_train, args.num_class, num_train_samples, p_P)
        x_val, y_val = tweak_dist(x_val, y_val, args.num_class, num_train_samples, p_P)

    else:
        x_test, y_test = tweak_dist(x_test, y_test, args.num_class, num_test_samples, p_Q)
    return x_train, y_train, x_val, y_val, x_test, y_test

class Dataset(Dataset):
    def __init__(self, data, label):
        self.images = data
        self.target = label

    def __getitem__(self, index):
        img = self.images[index]
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)
