# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca)
# The code has been modified from pytorch example vae code and inspired by the origianl tensorflow implementation of gumble-softmax by Eric Jang.

from __future__ import print_function
import argparse
import numpy as np
# import kun_tars
import torch
from torch import nn, optim
from torch.nn import functional as F
from dataset_simul import Dataset_simul
from torch.autograd import Variable
from mmd import mix_rbf_mmd2_joint, mix_rbf_mmd2
import model as models
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
from model import y_to_x,make_moons
parser = argparse.ArgumentParser(description='label shift gumbel example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--x_dim', type=int, default=1, metavar='N',
                    help='number of x_dim')
parser.add_argument('--run_kmm', type=int, default=0, metavar='N',
                    help='whether to run kmm training')
parser.add_argument('--y_dim', type=int, default=1, metavar='N',
                    help='number of y_dim')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--id', type=int, default=3,
                    help='shift type 1,2 Gaussian 3 Mix-Gaussian 4 Random')
parser.add_argument('--rn', type=int, default=1,
                    help='the number of experiment')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
# device = "cpu"
args.cuda =False
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

x_dim = args.x_dim
y_dim = args.y_dim # one-of-K vector
noise_dim = 1
# id_list = [1','2','3','4']
BCE_loss = nn.BCELoss()
specs = {'root': './data', 'r':10,'id':args.id, 'num_class': 2, 'num_train': 1000, 'dim': 1, 'ratio': 2, 'train': True,'x_dim':x_dim,'y_dim':y_dim,'rn':args.rn}
kwargs = {'num_workers': 5, 'pin_memory': True} if args.cuda else {}
G_func = y_to_x(input_size=specs['y_dim'],output_size=specs['x_dim'])
train_loader = torch.utils.data.DataLoader(Dataset_simul(specs,make_moons), shuffle=True, batch_size=args.batch_size, **kwargs)
specs['train'] = False
test_loader = torch.utils.data.DataLoader(Dataset_simul(specs,make_moons), shuffle=True, batch_size=args.batch_size, **kwargs)

temp_min = 0.5
ANNEAL_RATE = 0.00003

f = open("./output" + str(args.x_dim) + str(args.id)+str(args.rn)+".txt", 'w+')
# G = model.Generator()
G_func = G_func.to(device)
GAN_model = models.GAN_Transform(args.temp, x_dim, y_dim,G_func).to(device) # Label Transformation model

# Regression models
regress = models.regressor(input_size=x_dim,output_size=y_dim).to(device)  # Original
weighted_regress = models.regressor(input_size=x_dim,output_size=y_dim).to(device) # KMM
weighted_regress_feature = models.regressor(input_size=x_dim,output_size=y_dim).to(device)  # KMM_feature


G_D = models.Discriminator(input_size=x_dim,output_size=y_dim).to(device)
T_D = models.T_D(input_size=10).to(device)

optimizer = optim.Adam(GAN_model.G.parameters(), lr=1e-3)

optimizer1 = optim.Adam(GAN_model.T.parameters(), lr=1e-3)
optimizerR = optim.Adam(regress.parameters(), lr=1e-3)
optimizerWeighted = optim.Adam(weighted_regress.parameters(), lr=1e-3)
optimizerWeighted_feature = optim.Adam(weighted_regress_feature.parameters(), lr=1e-3)
optimizerT_D = optim.Adam(T_D.parameters(), lr=1e-3)
optimizerG_D = optim.Adam(G_D.parameters(), lr=1e-3)

if x_dim == 1 :
    D_steps = 16
    G_steps = 1


errors_baseline = []
errors_weighted = []
errors_weighted_feature = []
errors_fine = []
errors_newreg = []

if x_dim == 1:
    optimizerFine = optim.Adam(regress.fc2.parameters(), lr=1e-3)
else:
    optimizerFine = optim.Adam(regress.fc2.parameters(), lr=1e-3)
# train conditional gan on source domain

def loss_hinge_dis(dis_real, dis_fake):

  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake

def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

def train_cgan(epoch):
    GAN_model.train()
    train_loss = 0
    mse_losses = 0
    test_losses = 0
    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.float().to(device)
        y = y.float().to(device)
        for i in range(20):
            optimizerG_D.zero_grad()
            noise = torch.randn((x.shape[0], noise_dim)).to(device)
            fake_y = torch.Tensor(x.shape[0], y_dim).uniform_(-10,10).to(device)
            # print(fake_y)
            fx = GAN_model(fake_y,noise)
            real_uncond, real_cond, _ = G_D(x,noise)
            fake_uncond, _, fake_cond_mi= G_D(fx.detach(),noise)
            real_loss, fake_loss = loss_hinge_dis(real_uncond,fake_uncond)
            # real_loss = F.mse_loss(fx,x)
            real_cond_loss = F.mse_loss(real_cond,y)
            fake_cond_mi_loss = F.mse_loss(fake_cond_mi,fake_y)
            D_loss = real_loss+real_cond_loss+fake_cond_mi_loss +fake_loss  # Please refer to TAC-GAN (Twin auxiliary classifier conditional GAN)
            D_loss.backward()
            optimizerG_D.step()

        for j in range(1):
            optimizer.zero_grad()
            noise = torch.randn((x.shape[0], noise_dim)).to(device)
            fake_y = torch.Tensor(x.shape[0], y_dim).uniform_(-10,10).to(device)
            fx = GAN_model(fake_y, noise)
            fake_uncond,fake_cond,fake_cond_mi = G_D(fx, noise)
            G_uncond_loss  = loss_hinge_gen(fake_uncond)
            G_cond_loss = F.mse_loss(fake_cond,fake_y)
            G_cond_mi_loss = F.mse_loss(fake_cond_mi,fake_y)
            G_loss = G_uncond_loss + G_cond_loss  - G_cond_mi_loss
            G_loss.backward()
            optimizer.step()

    for test_num in range(100):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            noise = torch.randn((x.shape[0], noise_dim)).to(device)
            # fake_y = torch.Tensor(x.shape[0], y_dim).uniform_(-10,10).to(device)
            fx = GAN_model(y, noise)
            x_loss = F.mse_loss(fx,x,reduction='sum')
            test_losses += x_loss.item()
    # print(len(train_loader.dataset))
    print('====> Epoch: {} Average loss: {:.4f}\t Mse loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), test_losses/ len(train_loader.dataset) / 100))
    return test_losses/ len(train_loader.dataset) / 100


# given trained generator in source domain, optimize p_y in target domain by adversarial training
def train_gan(epoch):
    GAN_model.train()
    train_loss = 0
    temp = args.temp

    median_samples = 1000
    sub = lambda feats, n: feats[np.random.choice(
        feats.shape[0], min(feats.shape[0], n), replace=False)]
    from sklearn.metrics.pairwise import euclidean_distances
    Z = np.r_[sub(train_loader.dataset.data, median_samples // 2), sub(test_loader.dataset.data, median_samples // 2)]
    D2 = euclidean_distances(Z, squared=True)
    upper = D2[np.triu_indices_from(D2, k=1)]
    kernel_width = np.median(upper, overwrite_input=True) / math.sqrt(2)
    sigma = np.sqrt(kernel_width / 2)  # The sigma trick from KMM method
    # print("sigma:",sigma)
    sum_loss = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        mini_batch = x.shape[0]
        for i in range(G_steps):
            train_idx = np.array(np.random.randint(0, len(train_loader.dataset), mini_batch))
            # print(train_idx)
            train_idx = train_idx.astype(int)
            train_batch = np.array(train_loader.dataset.labels)[train_idx]
            # print(train_batch.shape)
            train_batch = train_batch[:mini_batch, :]
            train_batch = torch.from_numpy(train_batch).float().to(device)  ## sample y from training dataset
            # print(y)
            x = x.float().to(device)
            optimizer1.zero_grad()
            noise = torch.randn((x.shape[0], noise_dim)).to(device)
            noise_T = torch.empty(x.shape[0], noise_dim).uniform_(-1, 1).to(device)
            fx, _ = GAN_model.forward_gumbel(train_batch, noise, noise_T)
            sigma_list = [sigma]
            loss = mix_rbf_mmd2(regress.feature_extractor(x), regress.feature_extractor(fx), sigma_list=sigma_list)

            loss.backward()
            optimizer1.step()
            # sum_loss += loss.item()
            # print("loss_g",loss.item())

        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

    g_x = []
    for batch_idx, (x, y) in enumerate(test_loader):
        mini_batch = x.shape[0]
        for i in range(G_steps):
            train_idx = np.array(np.random.randint(0, len(train_loader.dataset), mini_batch))
            train_idx = train_idx.astype(int)
            train_batch = np.array(train_loader.dataset.labels)[train_idx]
            train_batch = train_batch[:mini_batch, :]
            train_batch = torch.from_numpy(train_batch).float().to(device)  ## sample y from training dataset
            x = x.float().to(device)
            noise = torch.randn((x.shape[0], noise_dim)).to(device)
            noise_T = torch.randn((x.shape[0], noise_dim)).to(device)
            noise_T = torch.empty(x.shape[0], noise_dim).uniform_(-1, 1).to(device)
            fx, _ = GAN_model.forward_gumbel(train_batch, noise, noise_T)
            g_x.append(fx.detach().cpu().numpy())
    g_x = torch.from_numpy(np.concatenate(g_x)).reshape(-1,1)
    sigma_list = [sigma]
    loss = mix_rbf_mmd2(regress.feature_extractor(torch.from_numpy(test_loader.dataset.data).float().to(device)), regress.feature_extractor(g_x.float().to(device)), sigma_list=sigma_list)
    sum_loss += loss.item()
    if epoch % 100 == 0:
        mini_batch = 10000
        train_idx = np.array(np.random.randint(0, train_loader.dataset.labels.shape[0], mini_batch))
        print(train_idx)
        train_idx = train_idx.astype(int)
        train_batch = np.array(train_loader.dataset.labels)[train_idx]
        # print(train_batch.shape)
        train_batch = train_batch[:mini_batch, :]
        train_batch = torch.from_numpy(train_batch).float().to(device)
        # if args.cuda:
        noise = torch.randn((train_batch.shape[0], noise_dim)).to(device)
        noise_T = torch.randn((train_batch.shape[0], noise_dim)).to(device)
        noise_T = torch.empty(train_batch.shape[0], noise_dim).uniform_(-1, 1).to(device)
        fxt, yt = GAN_model.forward_gumbel(train_batch, noise, noise_T)
        yt = yt.detach().cpu().numpy()  # pay attention, the y ouput is onehot, need to convert to ordinal label
        fxt = fxt.detach().cpu().numpy()
        num_bins = 100
        # the histogram of the data
        # sns.distplot
        # f, ax = plt.subplots(figsize=(6, 6))
        # n, bins, patches = plt.hist(yt, num_bins, normed=1, facecolor='blue', alpha=0.5)
        ax = sns.distplot(yt, num_bins, hist=False,norm_hist=True,color='green',label = "Density", axlabel = "Generate Label Distribution")
        plt.savefig("T" + str(specs['id']) + str(x_dim) + str(y_dim) + str(epoch) + ".png")
        plt.close()
        # plt.plot(bins[0:-1], n, 'r--')
        # plt.savefig("T" + str(specs['id']) + str(x_dim) + str(y_dim) + str(epoch) + ".png")
        # plt.close()
    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return sum_loss

def train_regression(epoch):
    regress.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.float().to(device)
        y = y.float().to(device)
        optimizerR.zero_grad()
        fy = regress(x)
        loss = F.mse_loss(fy,y)
        loss.backward()
        train_loss += loss.item()
        optimizerR.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def finetune(epoch):
    regress.train()
    train_loss = 0
    for i in range(int(len(train_loader.dataset)/args.batch_size)):
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        mini_batch = args.batch_size
        train_idx = np.array(np.random.randint(0, len(train_loader.dataset), mini_batch))
        # print(train_idx)
        train_idx = train_idx.astype(int)
        train_batch = np.array(train_loader.dataset.labels)[train_idx]
        # print(train_batch.shape)
        train_batch = train_batch[:mini_batch, :]
        train_batch = torch.from_numpy(train_batch).float().to(device)
        # if args.cuda:
        #     x = x.cuda()
        #     y = y.cuda()
        # x = Variable(x)
        # y = Variable(y)
        optimizerFine.zero_grad()
        noise = torch.randn((train_batch.shape[0], noise_dim)).to(device)
        noise_T = torch.randn((train_batch.shape[0], noise_dim)).to(device)
        noise_T = torch.empty(train_batch.shape[0], noise_dim).uniform_(-1, 1).to(device)
        x, y = GAN_model.forward_gumbel(train_batch,noise,noise_T)
        # x = G_func(y, noise)
        fy = regress(x)
        loss = F.mse_loss(fy, y)
        loss.backward()
        train_loss += loss.item()
        optimizerFine.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(x), len(train_loader.dataset),
                       100. * i / len(train_loader),
                       loss.item() / len(x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def weighted_train_regress(epoch,weight_func,train_regressor = weighted_regress,opt = optimizerWeighted):
    weighted_regress.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        mini_batch = x.shape[0]
        weight_batch = []
        for i in range(mini_batch):
            weight_batch.append(weight_func(x[i],y[i].item()))
        weight_batch = np.array(weight_batch)
        weight_batch = torch.from_numpy(weight_batch).view(-1,1).float().to(device)
        # if args.cuda:
        x = x.float().to(device)
        y = y.float().to(device)


        opt.zero_grad()
        # noise = torch.randn((x.shape[0], x_dim)).cuda()
        fy = train_regressor(x)
        loss = F.mse_loss(fy, y,reduce=False)
        # print(weight_batch.size())
        loss = loss*weight_batch
        # print(loss)
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        opt.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(x)))

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def Test(Test_model = regress):
    Test_model.eval()
    test_loss = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.float().to(device)
        y = y.float().to(device)
        fy = Test_model(x)
        loss = F.mse_loss(fy, y)
        test_loss += loss.item()
    print('====>  Average Test loss: {:.6f}'.format( test_loss / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset)

def run():
    best_err =100000
    gen_filename = 'results/0' + str(x_dim) + str(y_dim) + '_best.pkl' # The best generator file name
    if not os.path.exists(gen_filename):
        for epoch in range(1,  20000 + 1):
            cur_error = train_cgan(epoch)  # Train the generator P(X|Y)
            if cur_error <= best_err:
                best_err = cur_error
                gen_filename = 'results/0' +  str(x_dim) + str(y_dim) + '_best.pkl'
                torch.save(GAN_model.G.state_dict(), gen_filename)
            elif epoch % 100 == 0 :
                gen_filename = 'results/0' + str(x_dim) + str(y_dim) + str(epoch)+'.pkl'
                torch.save(GAN_model.G.state_dict(), gen_filename)
            print('best value:', best_err)
        np.save("best_err"  + str(args.x_dim) + str(y_dim) + ".txt", best_err)
    gen_filename = 'results/0' + str(x_dim) + str(y_dim) + '_best.pkl'
    GAN_model.G.load_state_dict(torch.load(gen_filename))

    for epoch in range(1, args.epochs + 1):  # Train the original regressor
            train_regression(epoch)
            errors_baseline.append(Test())

    best_loss_g = 10000
    for epoch in range(1, 10000 + 1):
        cur_loss = train_gan(epoch)  # Train the label transformation module T
        if cur_loss <= best_loss_g:
            best_loss_g = cur_loss
            gen_filename = 'results/transformation_model'+str(x_dim)+str(y_dim)+ str(specs['id']) + '_best.pkl'
            torch.save(GAN_model.state_dict(), gen_filename)
    gen_filename = 'results/transformation_model' + str(x_dim) + str(y_dim) + str(specs['id']) + '_best.pkl'
    GAN_model.load_state_dict(torch.load(gen_filename))

if __name__ == '__main__':

    # train
    run()  # Main function

    # save model
    gen_filename = 'results/transformation_model'+str(x_dim)+str(y_dim)+ str(specs['id'])+str(args.rn)+'.pkl'
    reg_filename = 'results/reg'+ str(specs['id'])+str(x_dim)+str(y_dim)+'.pkl'
    torch.save(GAN_model.state_dict(), gen_filename)
    torch.save(regress.state_dict(), reg_filename)
    colors = ['b', 'r', 'k', 'g', 'c']  # domain
    markers = ['o', 'x', '<', '+']  # class
    if args.run_kmm:
        beta_kun = kun_tars.py_betaKMM_targetshift(np.array(train_loader.dataset.data),np.array(train_loader.dataset.labels),
                                           np.array(test_loader.dataset.data))

        fig, ax = plt.subplots(1, 1)
        np.save("beta_kun"  + str(args.x_dim) + str(args.id) + str(args.rn)+".npy", beta_kun.squeeze())
        colors = ['b', 'r', 'k', 'g', 'c']  # domain
        markers = ['o', 'x','<']  # class
        ax.scatter(np.array(train_loader.dataset.labels).squeeze(), beta_kun.squeeze(), c=colors[0], marker=markers[1],
                   cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig("kun_zhang"+str(specs['id'])+str(x_dim)+str(y_dim)+".png")

        beta_kun_feature = kun_tars.py_betaKMM_targetshift(
            np.array(
                regress.feature_extractor(torch.from_numpy(train_loader.dataset.data).float().to(device)).detach().cpu().numpy()),
            np.array(train_loader.dataset.labels),
            np.array(
                regress.feature_extractor(torch.from_numpy(test_loader.dataset.data).float().to(device)).detach().cpu().numpy()))#,sigma=0.04)
        fig, ax = plt.subplots(1, 1)
        #
        ax.scatter(np.array(train_loader.dataset.labels).squeeze(), beta_kun_feature.squeeze(), c=colors[0], marker=markers[1],
                   cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig("kun_zhang_feature"+str(specs['id'])+str(x_dim)+str(y_dim)+".png")

        beta_kun = beta_kun.squeeze()
        labels = train_loader.dataset.labels.squeeze()
        weight_dict = {}
        for i in range(beta_kun.shape[0]):
            weight_dict[labels[i].item()] = beta_kun[i]
        weight_func = lambda x,y: weight_dict[y]
        beta_kun_feature = beta_kun_feature.squeeze()
        np.save("beta_kun_feature" + str(args.x_dim) + str(args.id) +  str(args.rn)+".npy", beta_kun_feature)
        labels = train_loader.dataset.labels.squeeze()
        weight_dict_feature = {}
        for i in range(beta_kun_feature.shape[0]):
            weight_dict[labels[i].item()] = beta_kun_feature[i]
        weight_func_feature = lambda x,y: weight_dict[y]
        print(weight_dict)
        for epoch in range(1, args.epochs + 1):
            weighted_train_regress(epoch,weight_func)
            errors_weighted.append(Test(weighted_regress))
        reg_filename = 'results/weighted_regress' + str(D_steps) + str(specs['id']) + str(x_dim) + str(y_dim)+ str(args.rn) + '.pkl'
        torch.save(regress.state_dict(), reg_filename)
        for epoch in range(1, args.epochs + 1):
            weighted_train_regress(epoch,weight_func_feature,weighted_regress_feature,optimizerWeighted_feature)
            errors_weighted_feature.append(Test(weighted_regress_feature))
        reg_filename = 'results/weighted_regress_feature' + str(D_steps) + str(specs['id']) + str(x_dim) + str(y_dim) + str(args.rn)+ '.pkl'
        torch.save(regress.state_dict(), reg_filename)
    for epoch in range(1, args.epochs + 1):
        finetune(epoch)
        errors_fine.append(Test())
    reg_filename = 'results/finetune_reg' + str(D_steps) + str(specs['id']) + str(x_dim) + str(y_dim) + str(args.rn)+ '.pkl'
    torch.save(regress.state_dict(), reg_filename)

    fig, ax = plt.subplots(1, 1)
    if args.run_kmm:
        Label_Com = ['baseline', 'weight_kmm','weight_kmm_feature', 'finetune']
        ax.scatter(np.array(range(args.epochs)).squeeze(), errors_baseline, c=colors[0], marker=markers[0],
                   label=Label_Com[0],
                   cmap=matplotlib.colors.ListedColormap(colors))
        ax.scatter(np.array(range(args.epochs)).squeeze(), errors_weighted, c=colors[1], marker=markers[1],
                   label=Label_Com[1],
                   cmap=matplotlib.colors.ListedColormap(colors))
        ax.scatter(np.array(range(args.epochs)).squeeze(), errors_weighted_feature, c=colors[2], marker=markers[2],
                   label=Label_Com[2],
                   cmap=matplotlib.colors.ListedColormap(colors))
        ax.scatter(np.array(args.epochs + np.array(range(args.epochs))).squeeze(), errors_fine, c=colors[3],
                   marker=markers[3], label=Label_Com[3],
                   cmap=matplotlib.colors.ListedColormap(colors))
        ax.legend()
        plt.savefig("erros" + str(specs['id']) + str(x_dim) + str(y_dim) + ".png")
    else:
        Label_Com = ['baseline', 'finetune']
        ax.scatter(np.array(range(args.epochs)).squeeze(), errors_baseline, c=colors[0], marker=markers[0],
                   label=Label_Com[0],
                   cmap=matplotlib.colors.ListedColormap(colors))

        ax.scatter(np.array(args.epochs + np.array(range(args.epochs))).squeeze(), errors_fine, c=colors[1],
                   marker=markers[3], label=Label_Com[1],
                   cmap=matplotlib.colors.ListedColormap(colors))
        ax.legend()
        plt.savefig("erros" + str(specs['id']) + str(x_dim) + str(y_dim) + ".png")

    np.save("errors_baseline" + str(D_steps)+ str(args.x_dim) + str(args.id) + str(args.rn)+ ".npy", errors_baseline)
    if args.run_kmm:
        np.save("errors_weighted" + str(D_steps)+ str(args.x_dim) + str(args.id) + str(args.rn)+ ".npy",
                errors_weighted)
        np.save("errors_weighted_feature"+ str(D_steps) + str(args.x_dim) + str(args.id) + str(args.rn)+ ".npy",
                errors_weighted_feature)
    np.save("errors_fine" + str(D_steps)+ str(args.x_dim) + str(args.id) + str(args.rn)+ ".npy",
            errors_fine)

    print("x_dim",str(args.x_dim) + str(args.id),file=f)
    print("errors_baseline: ",errors_baseline[-1],file=f)
    if args.run_kmm:
        print("errors_weighted: ",errors_weighted[-1],file=f)
        print("errors_weighted_feature: ",errors_weighted_feature[-1],file=f)
    print("errors_fine: ",errors_fine[-1],file=f)

