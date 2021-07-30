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
import model
import utils
import Data
import numpy as np
import math
import train_test
# import model_resnet32
from biggan_model import Generator,Discriminator
from utils_BBSE import *
from utils_RLLS import *
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,confusion_matrix

# SEED = np.random.randint(1, 10000)

# args.num_class = 10

parser = argparse.ArgumentParser(description='Direct label Transform')
parser.add_argument('--preTrain',  type=int, default=0,
                        help='whether the model needs to pre-train a generator or not')
parser.add_argument('--cuda',  type=int, default=1,help=' CUDA training')
parser.add_argument('--tweak',  type=int, default=0,help=' 0:dirichlet   1:tweak one  2: minority class' )
parser.add_argument('--load_base',  type=int, default=0,help=' 0:dirichlet   1:tweak one  2: minority class' )
parser.add_argument('--batch_size',  type=int, default=128,help=' batch size')
parser.add_argument('--SEED',  type=int, default=42,help=' Random seed')
parser.add_argument('--alpha',  type=float, default=1,help=' batch size')
parser.add_argument('--img_size',  type=int, default=32,help=' image size')
parser.add_argument('--num_class',  type=int, default=10,help=' image size')
parser.add_argument('--g_epochs',  type=int, default=200,help='pretrain epochs')
parser.add_argument('--c_epochs',  type=int, default=20,help='pretrain epochs')
parser.add_argument('--num_D_steps',  type=int, default=4,help='num_D_steps')
parser.add_argument('--dataset',  type=str, default='mnist',
                        help='mnist or cifar10')
parser.add_argument('--nc', type=int,default=1, help='channel of input image; gray:1, RGB:3')
parser.add_argument('--nz', type=int, default=64, help='length of noize.')
parser.add_argument('--ndf', type=int, default=64, help='number of filters.')
parser.add_argument('--ngf', type=int, default=64, help='number of filters.')
parser.add_argument('--feature_num', type=int, default=10, help='number of feature.')
parser.add_argument('--savingroot', default='./result', help='path to saving.')
parser.add_argument('--big_model', default=False, help='network structure, True: self attention gan, False: plain model. Only specific for training on Cifar10 dataset')
args = parser.parse_args()
f = open("./output"+args.dataset+str(args.tweak)+str(args.feature_num)+".txt", 'w+')
print("Random Seed: ",args.SEED,file=f)
print("Random Seed: ",args.SEED)
#np.random.seed(args.SEED)
#torch.manual_seed(args.SEED)
use_cuda = args.cuda and torch.cuda.is_available()
# print(args.preTrain)
# results save folder
root = 'MNIST_cDCGAN_results/'
model_name = 'MNIST_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
save_name = root + model_name
path_G = save_name + 'generator_param.pkl'
path_D_s = save_name + 'discriminator_param.pkl'
if not os.path.exists(os.path.join(args.savingroot, args.dataset)):
        os.makedirs(os.path.join(args.savingroot, args.dataset))
os.makedirs(os.path.join(args.savingroot,args.dataset,'images'), exist_ok=True)
os.makedirs(os.path.join(args.savingroot,args.dataset,'chkpts'), exist_ok=True)
# torch.set_default_tensor_type('torch.cuda.LongTensor')
# fixed noise & label
device = torch.device("cuda" if use_cuda else "cpu")
print(torch.backends.cudnn.enabled)
temp_z_ = torch.randn(10, 100)

# training parameters
lr = 0.0002
# g_epochs = 20


# network
if args.dataset == 'mnist':
    netd_g = model.discriminator_source(args.ndf, args.nc, num_classes=10).to(device)
    net_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
    net_featureExtrator_mi = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
    netg = model.generator(args.nz, args.ngf, args.nc).to(device)
    netg.weight_init(mean=0, std=0.02)
    netd_g.weight_init(mean=0, std=0.02)
    D_classifier = model.D_target_classifier(d=args.feature_num , num_classes=10,args=args).to(device)
    D_classifier_mi = model.D_target_classifier(d=args.feature_num , num_classes=10,args=args).to(device)
    D_distribution = model.D_target_distribution(d=args.feature_num).to(device)
    T = model.Transform_reinforce(int(args.num_class / 1),int(args.num_class )).to(device)
elif args.dataset == 'cifar10':
        netd_g = Discriminator().to(device)
        netg = Generator().to(device)
        net_featureExtrator = model.ResNet18().to(device)
        net_featureExtrator_mi = model.ResNet18().to(device)
        D_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10,args=args).to(device)
        D_classifier_mi = model.D_target_classifier(d=args.feature_num, num_classes=10,args=args).to(device)
        D_distribution = model.D_target_distribution(d=args.feature_num).to(device)


elif args.dataset == 'f-m':
        netd_g = model.discriminator_source_fm(args.ndf, args.nc, num_classes=10).to(device)
        netg = model.generator_fm(args.nz, args.ngf, args.nc).to(device)
        net_featureExtrator =  model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)
        net_featureExtrator_mi = model.ResNet18(in_channels=1,feature_num=args.feature_num).to(device)
        D_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10,args=args).to(device)
        D_classifier_mi = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
        D_distribution = model.D_target_distribution(d=args.feature_num,num_class=args.num_class).to(device)
        T = model.Transform_reinforce(int(args.num_class/1),int(args.num_class )).to(device)
# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

def embed_z(opt):
    fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).to(device)
    return fixed



def train_gan(opt,loader,netd_g,net_featureExtrator,netg,test_loader,D_classifier,net_featureExtrator_mi=None,D_classifier_mi=None):

    # os.makedirs(os.path.join(opt.savingroot,args.dataset,'images'), exist_ok=True)
    # os.makedirs(os.path.join(opt.savingroot,args.dataset,'chkpts'), exist_ok=True)

    #Build networ

    optd_c = optim.Adam([{'params': net_featureExtrator.parameters()},
                            {'params': D_classifier.parameters()}], lr=2e-4,
                      betas=(0.5, 0.999))
    optd_mi = optim.Adam([{'params': net_featureExtrator_mi.parameters()},
                            {'params': D_classifier_mi.parameters()}], lr=2e-4,
                      betas=(0.5, 0.999))
    optd_g = optim.Adam(netd_g.parameters(), lr=2e-4,
                      betas=(0.0, 0.999))
    optg = optim.Adam(netg.parameters(), lr=2e-4,
                      betas=(0.0, 0.999))

    print('training_start',file=f)
    print('training_start')
    step = 0
    acc = []

    fixed = embed_z(opt)
    #
    if  opt.dataset == 'mnist' or 'f-m':
        for epoch in range(opt.c_epochs):
            print('Epoch ',epoch,file=f)
            print('Epoch ',epoch)


    step = 0
    if opt.dataset == 'cifar10':
        netg.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                 'chkpts/g_' + str(1000) + 'alpha' + str(args.alpha) + str(
                                                     args.dataset) + '.pth')))
        train_test.test(netg, fixed, -1, opt)

    else:
        for epoch in range(opt.g_epochs):
                print('Epoch:',epoch,file=f)
                print('Epoch:',epoch)
                if opt.dataset == 'mnist' or 'f-m':

                       step = train_test.train_g(netd_g,net_featureExtrator,netg,optd_g,optg,loader,epoch,step,opt,D_classifier,
                                              net_featureExtrator_mi,D_classifier_mi,optd_c,optd_mi)
                       train_test.test(netg, fixed, epoch, opt)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

if args.preTrain:
    p_train = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
    np.save( os.path.join(args.savingroot, args.dataset,"p_train"+str(args.alpha)+str(args.dataset)+".npy"), p_train)
    print("alpha:", args.alpha,file=f)
    print("alpha:", args.alpha)
    p_Q = np.random.dirichlet([args.alpha] * args.num_class)
    np.save( os.path.join(args.savingroot, args.dataset,"p_Q"+str(args.alpha)+str(args.dataset)+".npy"), p_Q)
    print("The test labels distribution is ", p_Q,file=f)
    print("The test labels distribution is ", p_Q)
    print(sum(p_Q),file=f)
    print(sum(p_Q))

    x_train, y_train,x_val,y_val, x_test, y_test = Data.preProcessData(p_Q=p_Q, args=args,p_P=p_train, alpha=args.alpha, img_size=args.img_size,
                                                           batch_size=args.batch_size)
    np.save( os.path.join(args.savingroot, args.dataset,"x_train" + str(args.alpha) + str(args.dataset) + ".npy"), x_train)
    np.save( os.path.join(args.savingroot, args.dataset,"y_train" + str(args.alpha) + str(args.dataset) + ".npy"), y_train)
    np.save( os.path.join(args.savingroot, args.dataset,"x_val" + str(args.alpha) + str(args.dataset) + ".npy"), x_val)
    np.save( os.path.join(args.savingroot, args.dataset,"y_val" + str(args.alpha) + str(args.dataset) + ".npy"), y_val)
    np.save( os.path.join(args.savingroot, args.dataset,"x_test" + str(args.alpha) + str(args.dataset) + ".npy"), x_test)
    np.save( os.path.join(args.savingroot, args.dataset,"y_test" + str(args.alpha) + str(args.dataset) + ".npy"), y_test)
    train_dataset = Data.Dataset(x_train, y_train)
    val_dataset = Data.Dataset(x_val, y_val)
    test_dataset = Data.Dataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), shuffle=True)
    print('training G start!',file=f)
    print('training G start!')

    train_gan(args,train_loader,netd_g,net_featureExtrator,netg,test_loader,D_classifier,net_featureExtrator_mi,D_classifier_mi)

else:
    p_train = np.load(os.path.join(args.savingroot, args.dataset,"p_train"+str(args.alpha)+str(args.dataset)+".npy"))
    print("alpha:", args.alpha,file=f)
    print("alpha:", args.alpha)
    p_Q = np.load(os.path.join(args.savingroot, args.dataset,"p_Q"+str(args.alpha)+str(args.dataset)+".npy"))


    x_train = np.load(os.path.join(args.savingroot, args.dataset,"x_train"+str(args.alpha)+str(args.dataset)+".npy"))
    y_train = np.load(os.path.join(args.savingroot, args.dataset,"y_train"+str(args.alpha)+str(args.dataset)+".npy"))
    x_val = np.load(os.path.join(args.savingroot, args.dataset,"x_val" + str(args.alpha) + str(args.dataset) + ".npy"))
    y_val = np.load(os.path.join(args.savingroot, args.dataset,"y_val" + str(args.alpha) + str(args.dataset) + ".npy"))
    x_test = np.load(os.path.join(args.savingroot, args.dataset,"x_test"+str(args.alpha)+str(args.dataset)+".npy"))
    y_test = np.load(os.path.join(args.savingroot, args.dataset,"y_test"+str(args.alpha)+str(args.dataset)+".npy"))
    train_dataset = Data.Dataset(x_train, y_train)
    val_dataset = Data.Dataset(x_val, y_val)
    test_dataset = Data.Dataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), shuffle=True)

    if args.dataset == 'cifar10':
        netg.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                 'chkpts/g_' + str(1000) + 'alpha' + str(args.alpha) + str(
                                                     args.dataset) + '.pth')))
    else:
        netg.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                     'chkpts/g_' + str(args.g_epochs - 1) + 'alpha' + str(
                                                         args.alpha) + str(args.dataset) + '.pth')))
        netd_g.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                       'chkpts/d_g' + str(args.g_epochs - 1) + 'alpha' + str(
                                                           args.alpha) + str(args.dataset) + '.pth')))


def generate_batch_images(Trans_func,G_func,batch_size = 128):  ## Generate one batch images according to the transformed label distribution
    z_ = torch.randn((batch_size, args.nz)).view(-1, args.nz)
    z_T = torch.randn((batch_size, int(args.num_class/1))).view(-1, int(args.num_class/1))
    if use_cuda:
        z_T = Variable(z_T.to(device))
        z_ = Variable(z_.to(device))
    else:
        z_T = Variable(z_T)
        z_ = Variable(z_)
    Trans_func.eval()
    G_func.eval()

    T_label_ = Trans_func(z_T)
    sample_T = torch.multinomial(T_label_, 1)
    one_hot_T = sample_T.to('cpu')
    one_hot_T = torch.zeros(batch_size, args.num_class).scatter_(1, one_hot_T.type(torch.LongTensor).view(-1, 1),1).to(device)
    sample_T = sample_T.view(-1).type(torch.LongTensor).to(device)
    # Generate a batch of images
    fake_imgs = G_func(z_, sample_T)
    return fake_imgs,one_hot_T

def Test_dis(Trans_func,py_base,wt_true):  ## Calculate transformed label distribution
    Trans_func.eval()
    label_test = np.zeros(args.num_class)
    for i in range(1000):
        z_T = torch.randn((1000, int(args.num_class/1))).view(-1, int(args.num_class/1))
        if use_cuda:
            z_T = Variable(z_T.to(device))
        else:
            z_T = Variable(z_T)
        T_label_test = Trans_func(z_T)
        sample_T = torch.multinomial(T_label_test, 1)
        sample_T = sample_T.to('cpu')

        T_label_test = sample_T.cpu().numpy().squeeze()
    # print(T_label_test)
        for j in range(T_label_test.shape[0]):
            label_test[T_label_test[j]] += 1
    label_dis = label_test / sum(label_test)
    w_model = np.array(label_dis)/py_base
    print("The learnt labels distribution is ",label_dis,file=f)
    print("The learnt labels distribution is ",label_dis)
    print("The real labels distribution is ", wt_true*py_base,file=f)
    print("The real labels distribution is ", wt_true*py_base)
    # error = math.sqrt(sum((label_dis-p_Q)**2))/len(label_dis)
    error = np.sum(np.square((w_model-wt_true)))/len(label_dis)
    print("The distance is", error,file=f)
    print("The distance is", error)
    return error,label_dis

def train( model1,model2, device, train_loader, optimizer, epoch,log_interval = 100):  ##train a classifier
    model2.train()
    model1.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        # print(target)
        data, target = data.to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model2(model1(data))
        # print(data.shape)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()),file=f)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def Calculte_Mean_Feature( model1,model2, device, train_loader):
    model2.eval()
    model1.eval()
    mean_feature = np.zeros(shape=(args.num_class,args.feature_num))
    y_number = np.zeros(shape=(args.num_class,1))
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        # print(target)
        data = data.to(device)
        features = model1(data)
        features = features.detach().cpu().numpy()
        for id, i in enumerate(target):
            y_number[i] += 1
            mean_feature[i] += features[id]
    mean_feature = mean_feature/y_number
    return mean_feature

def Calculte_Mean_Feature_Test( model1,model2, device, train_loader):
    model2.eval()
    model1.eval()
    mean_feature = np.zeros(shape=(args.feature_num))
    number = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        number += data.size()[0]
        # print(target)
        data = data.to(device)
        features = model1(data)
        features = features.detach().cpu().numpy()
        mean_feature += np.sum(features,axis=0)
    mean_feature = mean_feature / number
    return mean_feature

def weighted_train( model1,model2, device, train_loader, optimizer, epoch,weight= None, log_interval = 100):   ##  train a classifier using ERM method
    model2.train()
    model1.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        # print(target)
        data, target = data.to(device), target.type(torch.LongTensor).to(device)
        if weight is not None:
            w = torch.tensor(weight.squeeze()).float().to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=w)

        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        output = model2(model1(data))

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()



    print('Weighted Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()), file=f)
    print('Weighted Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(model1, model2, device, test_loader):  ## Test the classifier
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    real_label = []
    pred_label = []
    with torch.no_grad():
        for data, target in test_loader:
            real_label.append(target.int())
            data = data.float()
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            output = model2(model1(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_label.append(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    f1 = f1_score(np.concatenate(real_label), np.concatenate(pred_label), average='macro')
    recall = recall_score(np.concatenate(real_label), np.concatenate(pred_label), average='macro')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), file=f)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), f1, recall


def test_classifier(model1, model2, device, train_loader, test_loader, balanced_test_loader, opt, alpha,
                    epochs=1 + args.c_epochs, classifier='model', weight=None,rn=0):   ## Test the classifier
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    real_label = []
    pred_label = []
    acc_list = []
    balanced_acc_list = []
    f1_list = []
    balanced_f1_list = []
    recall_list = []
    balanced_recall_list = []
    for epoch in range(1 + args.c_epochs):
        if classifier == 'baseline':

            break
        else:
            weighted_train(model1, model2, device, train_loader, opt, epoch, weight=weight)
    acc, f1, recall = test(model1, model2, device, test_loader)
    balanced_acc, balanced_f1, balanced_recall = test(model1, model2, device,
                                                      balanced_test_loader)
    acc_list.append(acc)
    balanced_acc_list.append(balanced_acc)
    f1_list.append(f1)
    balanced_f1_list.append(balanced_f1)
    recall_list.append(recall)
    balanced_recall_list.append(balanced_recall)

    np.save( os.path.join(args.savingroot, args.dataset, str(args.tweak),"balanced"+classifier + "Accuracy" + 'Alpha' + str(alpha) + str(args.dataset)+str(rn) + ".npy"), balanced_acc_list)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"balanced"+classifier + "f1" + 'Alpha' + str(alpha) + str(args.dataset)+str(rn) + ".npy"), balanced_f1_list)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"balanced"+classifier + "recall" + 'Alpha' + str(alpha) + str(args.dataset)+str(rn) + ".npy"), balanced_recall_list)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),classifier + "Accuracy" + 'Alpha' + str(alpha) + str(args.dataset)+str(rn) + ".npy"), acc_list)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),classifier + "f1" + 'Alpha' + str(alpha) + str(args.dataset) +str(rn)+ ".npy"), f1_list)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),classifier + "recall" + 'Alpha' + str(alpha) + str(args.dataset) +str(rn)+ ".npy"), recall_list)
    return acc_list, f1_list, recall_list, balanced_acc_list, balanced_f1_list, balanced_recall_list

def save_result(acc,f1,recall,alpha,classifier='model',balanced=''):
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),balanced+classifier+"Accuracy" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"), acc)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),balanced+classifier+"f1" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"), f1)
    np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),balanced+classifier+"recall" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"), recall)

repeat_nums = 10
for rn in range(repeat_nums):
    if args.dataset == "mnist":
        args.img_size = 32
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        balanced_test_loader = torch.utils.data.DataLoader(
            datasets.QMNIST('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            # transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        balanced_test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "f-m":
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            # transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5,)),
        ])
        balanced_test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)
    if args.dataset == 'mnist':
        net_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
        D_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
    elif args.dataset == 'cifar10':
        net_featureExtrator = model.ResNet18().to(device)

        D_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
    elif args.dataset == 'f-m':
        net_featureExtrator =  model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)
        D_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
    if use_cuda:
        net_featureExtrator.to(device)
        D_classifier.to(device)

    if args.dataset == 'mnist':
        optd_c = optim.SGD([{'params': net_featureExtrator.parameters()},
                            {'params': D_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif args.dataset ==  'f-m':
        optd_c = optim.SGD([{'params': net_featureExtrator.parameters()},
                            {'params': D_classifier.parameters()}], lr=0.01, momentum=0.9)
    else:
        optd_c = optim.SGD([{'params': net_featureExtrator.parameters()},
                        {'params': D_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)

    if args.load_base == 0:
        for epoch in range(1 + args.c_epochs):
            train(net_featureExtrator, D_classifier, device, train_loader,optd_c, epoch)
            print("Epochs acc on standard test set: ",test(net_featureExtrator, D_classifier, device,test_loader))
        torch.save(net_featureExtrator.state_dict(), os.path.join(args.savingroot, args.dataset,
                                                           'chkpts/net_featureExtrator' + str(args.dataset) + '.pth'))
        torch.save(D_classifier.state_dict(), os.path.join(args.savingroot, args.dataset, 'chkpts/D_classifier'  + str(args.dataset) + '.pth'))
    if args.tweak == 0:
        alphas_list = [0.01,0.1,1,10]
    elif args.tweak==1:
        alphas_list = [0.9,0.8,0.7,0.6,0.5]
    else:
        alphas_list = [0.5, 0.4,0.3,0.2]
    netg.eval()
    train_test.toggle_grad(netg,False)
    for alpha in alphas_list:
        if not os.path.exists(os.path.join(args.savingroot, args.dataset, str(args.tweak))):
            os.makedirs(os.path.join(args.savingroot, args.dataset, str(args.tweak)))
        if not os.path.exists(os.path.join(args.savingroot, args.dataset, str(args.tweak),'chkpts')):
            os.makedirs(os.path.join(args.savingroot, args.dataset, str(args.tweak),'chkpts'))
        if not os.path.exists(os.path.join(args.savingroot, args.dataset, str(args.tweak),'data')):
            os.makedirs(os.path.join(args.savingroot, args.dataset, str(args.tweak),'data'))
        model_accuracies = []
        balanced_model_accuracies = []
        model_f1s = []
        balanced_model_f1s = []
        model_recalls = []
        balanced_model_recalls = []
        baseline_accuracies = []
        balanced_baseline_accuracies = []
        baseline_f1s = []
        balanced_baseline_f1s = []
        baseline_recalls = []
        balanced_baseline_recalls = []
        BBSE_accuracies = []
        balanced_BBSE_accuracies = []
        BBSE_f1s = []
        balanced_BBSE_f1s = []
        BBSE_recalls = []
        balanced_BBSE_recalls = []
        DLT_accuracies = []
        balanced_DLT_accuracies = []
        DLT_f1s = []
        balanced_DLT_f1s = []
        DLT_recalls = []
        balanced_DLT_recalls = []
        RLLS_accuracies = []
        balanced_RLLS_accuracies = []
        RLLS_f1s = []
        balanced_RLLS_f1s = []
        RLLS_recalls = []
        balanced_RLLS_recalls = []
        # balanced_acc = []

        torch.cuda.empty_cache()
        model_accuracy = []
        balanced_model_accuracy = []
        model_f1 = []
        balanced_model_f1 = []
        model_recall = []
        balanced_model_recall = []
        if args.tweak == 0:
            p_Q = np.random.dirichlet([alpha] * args.num_class)
        elif args.tweak==1:
            p_Q = Data.tweak_one(args.num_class, np.random.randint(0,args.num_class,1),alpha)
        else:
            p_Q = Data.tweak_monority(args.num_class,int(alpha*args.num_class))
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"p_Q" + str(alpha) + str(args.dataset) + str(rn)+".npy"), p_Q)
        print("The test labels distribution is ", p_Q,file=f)
        print("The test labels distribution is ", p_Q)
        print(sum(p_Q),file=f)
        print(sum(p_Q))
        _, _, _, _, x_test, y_test = Data.preProcessData(p_Q=p_Q, args=args, p_P=p_train,
                                                                             alpha=alpha, img_size=args.img_size,
                                                                             batch_size=args.batch_size)

        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"data","x_test" + str(alpha) + str(args.dataset) +str(rn)+ ".npy"), x_test)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"data","y_test" + str(alpha) + str(args.dataset) + str(rn)+".npy"), y_test)
        test_dataset = Data.Dataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), shuffle=True)
        if args.dataset == 'mnist':
            D_distribution = model.D_target_distribution(d=args.feature_num).to(device)
            T = model.Transform_reinforce(int(args.num_class / 1),int(args.num_class )).to(device)
        elif args.dataset == 'cifar10':
            D_distribution = model.D_target_distribution(d=args.feature_num).to(device)
            T = model.Transform_reinforce(int(args.num_class / 1),int(args.num_class )).to(device)

        elif args.dataset == 'f-m':
            D_distribution = model.D_target_distribution(d=args.feature_num,num_class=args.num_class).to(device)
            T = model.Transform_reinforce(int(args.num_class ),int(args.num_class )).to(device)
        if use_cuda:
            T.to(device)
            D_distribution.to(device)
        net_featureExtrator.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                   'chkpts/net_featureExtrator' + str(args.dataset) + '.pth')))
        train_test.toggle_grad(net_featureExtrator, False)
        D_classifier.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                   'chkpts/D_classifier' + str(args.dataset) + '.pth')))

        print("###########################  The original classifier ###################")
        print("###########################  The original classifier ###################",file=f)
        test_classifier(net_featureExtrator, D_classifier, device, train_loader, test_loader, balanced_test_loader,
                        optd_c, alpha, classifier='baseline',rn = rn)
        Py_true = calculate_marginal(y_test.squeeze(), args.num_class)
        Py_base = calculate_marginal(y_val.squeeze(), args.num_class)
        # print(Py_base)
        wt_true = Py_true / Py_base

        optimizer_classifier = optim.SGD(D_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        optimizer_T_D = torch.optim.Adam(D_distribution.parameters(), lr=8e-5, betas=(0.5,0.999))

        optimizer_T = torch.optim.Adam(T.parameters(), lr=8e-5, betas=(0.5,0.999))



        train_t_epoch = args.g_epochs
        train_classifier_epoch = len(test_loader)
        epochs = 0
        print('training target start!',file=f)
        print('training target start!')
        start_time = time.time()
        error = 1
        errors = []

        while epochs < 1000:  ## the training epochs for label transformation module
            D_t_losses = []
            G_t_losses = []
            # Test_dis(T)
            # # learning rate decay

            epoch_start_time = time.time()

            for x_, y_ in test_loader:
                    # train discriminator Dß
                for i in range(4):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    D_distribution.train()
                    x_ = x_.float()
                    mini_batch = x_.size()[0]
                    optimizer_T_D.zero_grad()
                    y_real_ = torch.ones(mini_batch)
                    y_fake_ = torch.zeros(mini_batch)

                    x_ = Variable(x_.to(device))
                    y_real_, y_fake_ = Variable(y_real_.to(device)), Variable(y_fake_.to(device))

                    fake_imgs,_ = generate_batch_images(T,netg,batch_size=mini_batch)
                    # Adversarial loss

                    real_loss = BCE_loss(D_distribution(net_featureExtrator((x_))), y_real_)
                    fake_loss = BCE_loss(D_distribution(net_featureExtrator((fake_imgs.detach()))), y_fake_)
                    loss_D = (real_loss + fake_loss)/2

                    loss_D.backward()
                    optimizer_T_D.step()

                    # Train the generator every n_critic iterations
                    # print("[Epoch %d/%d][D loss: %f] " % (epochs, train_t_epoch, loss_D.item()),file=f)
                    # print("[Epoch %d/%d][D loss: %f] " % (epochs, train_t_epoch, loss_D.item()))
                # if epochs % 1 == 0:
                        # -----------------
                        #  Train Generator
                        # -----------------
                for i in range(5):
                    D_distribution.eval()
                    optimizer_T.zero_grad()
                    y_real_ = torch.ones(mini_batch)

                    z_ = torch.randn((mini_batch, args.nz)).view(-1, args.nz)
                    z_T = torch.randn((mini_batch, int(args.num_class/1))).view(-1, int(args.num_class/1))
                    z_T = Variable(z_T.to(device))
                    z_ = Variable(z_.to(device))
                    y_real_= Variable(y_real_.to(device))
                    T.train()
                    T_label_ = T(z_T)
                    sample_T = torch.multinomial(T_label_,1)
                    sample_T = sample_T.view(-1).type(torch.LongTensor).to(device)
                    one_hot_T = torch.zeros(mini_batch, args.num_class).scatter_(1, sample_T.type(torch.LongTensor).view(-1,1), 1).to(device)

                    # Generate a batch of images
                    gen_imgs = netg(z_, sample_T)

                    rewards = D_distribution(net_featureExtrator(gen_imgs))
                    loss_G = -torch.mean(torch.log(torch.masked_select(T_label_,one_hot_T.bool()).view(mini_batch,-1))*(rewards-0.15))
                    loss_G.backward()
                    # print(loss_G.item())

                    optimizer_T.step()
                # print("[Epoch %d/%d] [G loss: %f]" % (epochs, train_t_epoch,loss_G.item()),file=f)
                # print("[Epoch %d/%d] [G loss: %f]" % (epochs, train_t_epoch,loss_G.item()))
            errors.append(error)
            one_hot_T = torch.zeros(mini_batch, args.num_class).scatter_(1, torch.LongTensor([8]*mini_batch).view(mini_batch,1),1)
            one_hot_T = torch.argmax(one_hot_T,dim=1)
            one_hot_T = one_hot_T.squeeze()
            # # Generate a batch of images
            if use_cuda:
                one_hot_T = Variable(one_hot_T.to(device))
            else:
                one_hot_T = Variable(one_hot_T)
            # Generate a batch of images
            gen_imgs = netg(z_, one_hot_T)
            # Adversarial loss
            rewards = D_distribution(net_featureExtrator(gen_imgs))
            # print(rewards,file=f)
            # print(rewards)
            epochs += 1
            error,_ = Test_dis(T,Py_base,wt_true)
            # Test_dis(T)
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
        torch.cuda.empty_cache()

        ## For BBSE and RLLS methods
        ypred_s, ypred_s_soft,y_true = predict_all(net_featureExtrator, D_classifier, device, val_loader)
        ypred_t, ypred_t_soft,_ = predict_all(net_featureExtrator, D_classifier, device, test_loader)
        # print(ypred_s.shape,file=f)
        # print(ypred_s.shape)

        train_features = Calculte_Mean_Feature(net_featureExtrator, D_classifier, device, train_loader)
        train_features = train_features.T
        test_mean_feature = Calculte_Mean_Feature_Test(net_featureExtrator, D_classifier, device, test_loader)


        wt_BBSE = estimate_labelshift_ratio(y_true.squeeze(), ypred_s.squeeze(), ypred_t.squeeze(),args.num_class)
        training_weights_BBSE=np.maximum(wt_BBSE, 0)
        wt_BBSE = np.array(training_weights_BBSE)
        Py_est_BBSE = estimate_target_dist(wt_BBSE.squeeze(), y_true.squeeze(),args.num_class)
# val_loader.dataset.target
#         print(np.concatenate((wt_BBSE.reshape(-1,1),wt_true.reshape(-1,1)),axis=1),file=f)
#         print(np.concatenate((wt_BBSE.reshape(-1,1),wt_true.reshape(-1,1)),axis=1))
#         print(np.concatenate((Py_est_BBSE.reshape(-1,1),Py_true.reshape(-1,1)),axis=1),file=f)
#         print(np.concatenate((Py_est_BBSE.reshape(-1,1),Py_true.reshape(-1,1)),axis=1))
        weightfunc_BBSE = lambda x,y: wt_BBSE[y.astype(int)]

        mu_t = calculate_marginal(ypred_t.squeeze(), args.num_class)
        # print(mu_t)
        mu_train_hat = calculate_marginal(ypred_s.squeeze(), args.num_class)
        # print(mu_train_hat)
        rho = compute_3deltaC(args.num_class, n_train=len(train_loader.dataset), delta=0.05)
        # print(rho)

        alpha_rlls = 0.0001
        C_yy = confusion_matrix(y_true,ypred_s,labels=np.array(range(args.num_class)))/len(ypred_s)
        wt_RLLS = compute_w_opt(C_yy.T, mu_t.squeeze(), mu_train_hat.squeeze(), alpha_rlls * rho)

        estimate_Y = compute_y_feature(train_features, test_mean_feature)
        # estimate_G_Y = compute_y_feature(train_G_mean_feature, test_mean_feature)

        # print("estimate_Y: ", estimate_Y)
        # print("estimate_Y: ", estimate_Y, file=f)
        wt_feature = estimate_Y / Py_base
        # wt_G_feature = estimate_G_Y/Py_base
        if not os.path.exists(os.path.join(args.savingroot, args.dataset, str(args.tweak))):
            os.makedirs(os.path.join(args.savingroot, args.dataset, str(args.tweak)))
        np.save(
            os.path.join(args.savingroot, args.dataset, str(args.tweak),
                         "RLLS_featureerror" + 'Alpha' + str(alpha) + str(args.dataset) + str(rn)+ ".npy"),
            np.array(np.sum(np.square((wt_feature - wt_true))) / args.num_class))

        print("###########################  Label distribution error for RLLS and BBSE ###################")
        print("###########################  Label distribution error for RLLS and BBSE ###################",file=f)
        print("mse rlls:",np.sum(np.square((wt_RLLS-wt_true)))/args.num_class,file=f)
        print("mse rlls:",np.sum(np.square((wt_RLLS-wt_true)))/args.num_class)
        print("mse BBSE:",np.sum(np.square((wt_BBSE-wt_true)))/args.num_class,file=f)
        print("mse BBSE:",np.sum(np.square((wt_BBSE-wt_true)))/args.num_class)
        # network
        if args.dataset == 'mnist':
            BBSE_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
            BBSE_classifier = model.D_target_classifier(d=args.feature_num , num_classes=10,args=args).to(device)
            optimizer_BBSE = optim.SGD([{'params': BBSE_featureExtrator.parameters()},
                                        {'params': BBSE_classifier.parameters()}], lr=0.01, momentum=0.9,
                                       weight_decay=5e-4)
        elif args.dataset == 'cifar10':
                BBSE_featureExtrator = model.ResNet18().to(device)
                BBSE_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
                optimizer_BBSE = optim.SGD([{'params': BBSE_featureExtrator.parameters()},
                                    {'params': BBSE_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)

        elif args.dataset == 'f-m':

                BBSE_featureExtrator =  model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)

                BBSE_classifier = model.D_target_classifier(d=args.feature_num,num_classes=10,args=args).to(device)
                optimizer_BBSE = optim.SGD([{'params': BBSE_featureExtrator.parameters()},
                                    {'params': BBSE_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)


        print("###########################  Test Classifier trained by BBSE ###################")
        print("###########################  Test Classifier trained by BBSE ###################",file=f)
        test_classifier(BBSE_featureExtrator, BBSE_classifier, device, train_loader, test_loader, balanced_test_loader,optimizer_BBSE,
                        alpha, classifier='BBSE',weight=wt_BBSE.squeeze(),rn = rn)

        torch.save(BBSE_featureExtrator.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                   'chkpts/BBSE_featureExtrator' + 'alpha' + str(alpha) + str(args.dataset) + str(rn)+'.pth'))
        torch.save(BBSE_classifier.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                     'chkpts/BBSE_classifier'  + 'alpha' + str(alpha) + str(args.dataset) +str(rn)+ '.pth'))
        torch.cuda.empty_cache()



        if args.dataset == 'mnist':
            RLLS_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
            RLLS_classifier = model.D_target_classifier(d=args.feature_num , num_classes=10,args=args).to(device)
            optimizer_RLLS = optim.SGD([{'params': RLLS_featureExtrator.parameters()},
                                        {'params': RLLS_classifier.parameters()}], lr=0.01, momentum=0.9,
                                       weight_decay=5e-4)
        elif args.dataset == 'cifar10':

                RLLS_featureExtrator = model.ResNet18().to(device)
                RLLS_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
                optimizer_RLLS = optim.SGD([{'params': RLLS_featureExtrator.parameters()},
                                    {'params': RLLS_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)

        elif args.dataset == 'f-m':
                RLLS_featureExtrator =  model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)

                RLLS_classifier = model.D_target_classifier(d=args.feature_num,num_classes=10,args=args).to(device)
                optimizer_RLLS = optim.SGD([{'params': RLLS_featureExtrator.parameters()},
                                    {'params': RLLS_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)

        print("###########################  Test Classifier trained by RLLS ###################")
        print("###########################  Test Classifier trained by RLLS ###################", file=f)
        test_classifier(RLLS_featureExtrator, RLLS_classifier, device, train_loader, test_loader,
                        balanced_test_loader,optimizer_RLLS,
                        alpha, classifier='RLLS',weight=wt_RLLS.squeeze(),rn = rn)

        torch.save(RLLS_featureExtrator.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                                   'chkpts/RLLS_featureExtrator' + 'alpha' + str(
                                                                       alpha) + str(args.dataset) + str(rn)+'.pth'))
        torch.save(RLLS_classifier.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                              'chkpts/RLLS_classifier' + 'alpha' + str(alpha) + str(
                                                                  args.dataset) + str(rn)+'.pth'))
        torch.cuda.empty_cache()
        # network

        print("###########################  Test Classifier trained by the ground truth label distribution using ERM ###################")
        print("###########################  Test Classifier trained by the ground truth label distribution using ERM ###################", file=f)
        if args.dataset == 'mnist':
            BEST_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
            BEST_classifier = model.D_target_classifier(d=args.feature_num , num_classes=10,args=args).to(device)
            optimizer_BEST = optim.SGD([{'params': BEST_featureExtrator.parameters()},
                                        {'params': BEST_classifier.parameters()}], lr=0.01, momentum=0.9,
                                       weight_decay=5e-4)
        elif args.dataset == 'cifar10':

                BEST_featureExtrator = model.ResNet18().to(device)
                BEST_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
                optimizer_BEST = optim.SGD([{'params': BEST_featureExtrator.parameters()},
                                    {'params': BEST_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)

        elif args.dataset == 'f-m':
                BEST_featureExtrator =  model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)
                BEST_classifier = model.D_target_classifier(d=args.feature_num,num_classes=10,args=args).to(device)
                optimizer_BEST = optim.SGD([{'params': BEST_featureExtrator.parameters()},
                                    {'params': BEST_classifier.parameters()}], lr=0.01, momentum=0.9, weight_decay=5e-4)



        test_classifier(BEST_featureExtrator, BEST_classifier, device, train_loader, test_loader,
                        balanced_test_loader,optimizer_BEST,
                        alpha, classifier='BEST',weight=wt_true.squeeze(),rn = rn)
        torch.save(BEST_featureExtrator.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                                   'chkpts/BEST_featureExtrator' + 'alpha' + str(
                                                                       alpha) + str(args.dataset) + str(rn)+'.pth'))
        torch.save(BEST_classifier.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                              'chkpts/BEST_classifier' + 'alpha' + str(alpha) + str(
                                                                  args.dataset) + str(rn)+'.pth'))
        torch.cuda.empty_cache()

        print(
            "###########################  Test Classifier trained by LTF using ERM ###################")
        print(
            "###########################  Test Classifier trained by LTF using ERM ###################",
            file=f)
        _,DLT_label_dis = Test_dis(T,Py_base,wt_true)
        DLT_label_dis = np.array(DLT_label_dis)
        wt_DLT = DLT_label_dis/Py_base
        training_weights_DLT=np.maximum(wt_DLT, 0)
        wt_DLT = np.array(training_weights_DLT)
        weightfunc_DLT = lambda x,y: wt_DLT[y.astype(int)]
        # network

        if args.dataset == 'mnist':
            DLT_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
            DLT_classifier = model.D_target_classifier(d=args.feature_num , num_classes=10,args=args).to(device)
            optimizer_weighted = optim.SGD([{'params': DLT_featureExtrator.parameters()},
                                            {'params': DLT_classifier.parameters()}], lr=0.01, momentum=0.9,
                                           weight_decay=5e-4)
        elif args.dataset == 'cifar10':
                DLT_featureExtrator = model.ResNet18().to(device)
                DLT_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
                optimizer_weighted = optim.SGD([{'params': DLT_featureExtrator.parameters()},
                                        {'params': DLT_classifier.parameters()}], lr=0.01, momentum=0.9,
                                       weight_decay=5e-4)

        elif args.dataset == 'f-m':
                DLT_featureExtrator =   model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)
                DLT_classifier = model.D_target_classifier(d=args.feature_num,num_classes=10,args=args).to(device)
                optimizer_weighted = optim.SGD([{'params': DLT_featureExtrator.parameters()},
                                        {'params': DLT_classifier.parameters()}], lr=0.01, momentum=0.9,
                                       weight_decay=5e-4)

        test_classifier(DLT_featureExtrator, DLT_classifier, device, train_loader, test_loader, balanced_test_loader,optimizer_weighted,
                        alpha,classifier='DLT',weight=wt_DLT.squeeze(),rn = rn)
        torch.save(DLT_featureExtrator.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                                   'chkpts/DLT_featureExtrator' + 'alpha' + str(
                                                                       alpha) + str(args.dataset) +str(rn)+ '.pth'))
        torch.save(DLT_classifier.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                              'chkpts/DLT_classifier' + 'alpha' + str(alpha) + str(
                                                                  args.dataset) + str(rn)+'.pth'))

        print(
            "###########################  Test Classifier trained by LTF using fine-tune ###################")
        print(
            "###########################  Test Classifier trained by LTF using fine-tune ###################",
            file=f)

        if args.dataset == 'mnist':
            model_featureExtrator = model.feature_extractor(args.feature_num, args.nc, num_classes=10).to(device)
            model_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
            optimizer_classifier = optim.SGD(model_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif args.dataset == 'cifar10':
            model_featureExtrator = model.ResNet18().to(device)
            model_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
            optimizer_classifier = optim.SGD(model_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        elif args.dataset == 'f-m':

            model_featureExtrator = model.feature_extractor_fm(args.ndf, args.nc, num_classes=10).to(device)

            model_classifier = model.D_target_classifier(d=args.feature_num, num_classes=10, args=args).to(device)
            optimizer_classifier = optim.SGD(model_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        model_featureExtrator.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                                    'chkpts/net_featureExtrator' + str(
                                                                        args.dataset) + '.pth')))
        model_classifier.load_state_dict(torch.load(os.path.join(args.savingroot, args.dataset,
                                                             'chkpts/D_classifier' + str(args.dataset) + '.pth')))
        for epoch in range(10):
            for i in range(train_classifier_epoch):
                imgs, labels = generate_batch_images(T,netg)
                target = torch.argmax(labels,dim=1).squeeze()
                # print(target)
                optimizer_classifier.zero_grad()
                output = model_classifier(model_featureExtrator(imgs))
                # print(data.shape)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer_classifier.step()
            print('Train Loss: {:.6f}'.format(loss.item()),file=f)
            print('Train Loss: {:.6f}'.format(loss.item()))
            acc,f1,recall = test(model_featureExtrator, model_classifier, device, test_loader)
            balanced_acc,balanced_f1,balanced_recall = test(model_featureExtrator, model_classifier, device, balanced_test_loader)
            model_accuracy.append(acc)
            balanced_model_accuracy.append(balanced_acc)
            model_f1.append(f1)
            balanced_model_f1.append(balanced_f1)
            model_recall.append(recall)
            balanced_model_recall.append(balanced_recall)
        torch.save(model_featureExtrator.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                                   'chkpts/model_featureExtrator' + 'alpha' + str(
                                                                       alpha) + str(args.dataset) + str(rn)+'.pth'))
        torch.save(model_classifier.state_dict(), os.path.join(args.savingroot, args.dataset,str(args.tweak),
                                                              'chkpts/model_classifier' + 'alpha' + str(alpha) + str(
                                                                  args.dataset) + str(rn)+'.pth'))
        print("models' accuracy ",model_accuracy,file=f)
        print("models' accuracy ",model_accuracy)
        print("models' f1 ",model_f1,file=f)
        print("models' f1 ",model_f1)
        print("models' recall ",model_recall,file=f)
        print("models' recall ",model_recall)
        print("models'balanced_ accuracy ",balanced_model_accuracy,file=f)
        print("models'balanced_ accuracy ",balanced_model_accuracy)
        print("models'balanced_ f1 ",balanced_model_f1,file=f)
        print("models'balanced_ f1 ",balanced_model_f1)
        print("models'balanced_ recall ",balanced_model_recall,file=f)
        print("models'balanced_ recall ",balanced_model_recall)
        print(errors)
        print(errors,file=f)
        print("BBSE weight：",np.concatenate((wt_BBSE.reshape(-1,1),wt_true.reshape(-1,1)),axis=1),file=f)
        print("BBSE weight：",np.concatenate((wt_BBSE.reshape(-1,1),wt_true.reshape(-1,1)),axis=1))
        print("BBSE p_Y：", np.concatenate((Py_est_BBSE.reshape(-1,1),Py_true.reshape(-1,1)),axis=1),file=f)
        print("BBSE p_Y：", np.concatenate((Py_est_BBSE.reshape(-1,1),Py_true.reshape(-1,1)),axis=1))
        print("DLT weight：" ,np.concatenate((wt_DLT.reshape(-1,1),wt_true.reshape(-1,1)),axis=1),file=f)
        print("DLT weight：" ,np.concatenate((wt_DLT.reshape(-1,1),wt_true.reshape(-1,1)),axis=1))
        print("DLT p_Y：", np.concatenate((DLT_label_dis.reshape(-1,1),Py_true.reshape(-1,1)),axis=1),file=f)
        print("DLT p_Y：", np.concatenate((DLT_label_dis.reshape(-1,1),Py_true.reshape(-1,1)),axis=1))
        print("RLLS weight：" ,np.concatenate((wt_RLLS.reshape(-1,1),wt_true.reshape(-1,1)),axis=1),file=f)
        print("RLLS weight：" ,np.concatenate((wt_RLLS.reshape(-1,1),wt_true.reshape(-1,1)),axis=1))
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"modelAccuracy"+ 'Alpha'+str(alpha)+str(args.dataset)+ str(rn)+".npy"),model_accuracy)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"modelf1"+ 'Alpha'+str(alpha)+str(args.dataset)+str(rn)+".npy"),model_f1)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"modelrecall"+ 'Alpha'+str(alpha)+str(args.dataset)+str(rn)+".npy"),model_recall)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"balancedmodelAccuracy"+ 'Alpha'+str(alpha)+str(args.dataset)+str(rn)+".npy"),balanced_model_accuracy)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"balancedmodelf1"+ 'Alpha'+str(alpha)+str(args.dataset)+str(rn)+".npy"),balanced_model_f1)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"balancedmodelrecall"+ 'Alpha'+str(alpha)+str(args.dataset)+str(rn)+".npy"),balanced_model_recall)

        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),
                             "BBSEerror" + 'Alpha' + str(alpha) + str(args.dataset) + str(rn)+".npy"), np.sum(np.square((wt_BBSE-wt_true)))/args.num_class)
        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),
                             "RLLSerror" + 'Alpha' + str(alpha) + str(args.dataset) + str(rn)+ ".npy"), np.sum(np.square((wt_RLLS-wt_true)))/args.num_class)

        np.save(os.path.join(args.savingroot, args.dataset,str(args.tweak),"errors"+ 'Alpha'+str(alpha)+str(args.dataset)+ str(rn)+".npy"),errors)

    for alpha in alphas_list:

        classifier_list=['model','baseline','BBSE','RLLS','DLT','BEST']
        for name in classifier_list:

            model_accuracies = np.load(os.path.join(args.savingroot, args.dataset,str(args.tweak),name+"Accuracy" + 'Alpha' + str(alpha) + str(args.dataset) +str(rn)+ ".npy"))
            model_f1s = np.load(os.path.join(args.savingroot, args.dataset,str(args.tweak),name+"f1" + 'Alpha' + str(alpha) + str(args.dataset) +str(rn)+ ".npy"))
            model_recalls = np.load(os.path.join(args.savingroot, args.dataset,str(args.tweak),name+"recall" + 'Alpha' + str(alpha) + str(args.dataset) + str(rn)+".npy"))
            print(name+" accuracy  Alpha" + str(alpha) + "  ", model_accuracies,file=f)
            print(name+" accuracy  Alpha" + str(alpha) + "  ", model_accuracies)
            print(name+ "  f1" + str(alpha) + "  ", model_f1s,file=f)
            print(name+ "  f1" + str(alpha) + "  ", model_f1s)
            print(name+ "  recall" + str(alpha) + "  ", model_recalls,file=f)
            print(name+ "  recall" + str(alpha) + "  ", model_recalls)