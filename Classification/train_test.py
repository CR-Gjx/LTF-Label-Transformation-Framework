
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from biggan_model import G_D
from tqdm import tqdm
import losses

def denorm(x):
    return (x +1)/2
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off
def train_c(epoch,netd_c,optd_c,loader,step,opt,D_classifier):


    netd_c.train()
    D_classifier.train()
    for _, (image_c, label) in enumerate(loader):

        # plt.imshow(np.transpose((image[0].cpu().numpy()+1)/2,[1,2,0]))
        # plt.show()
        # print(image.max(),image.min())


        real_label = label.type(torch.LongTensor).cuda()
        image_c = image_c.float()
        real_input_c = Variable(image_c).cuda()

        real_cls = D_classifier(netd_c(real_input_c))
        real_loss_c = F.cross_entropy(real_cls, real_label)  #

        optd_c.zero_grad()
        real_loss_c.backward()
        optd_c.step()

    if not os.path.exists(os.path.join(opt.savingroot, opt.dataset)):
        os.makedirs(os.path.join(opt.savingroot, opt.dataset))


    # torch.save(netd_c.state_dict(), os.path.join(opt.savingroot, opt.dataset, 'chkpts/d_'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.pth'))
    torch.save(netd_c.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                 'chkpts/d_c' + str(epoch) + 'alpha' + str(opt.alpha) + str(
                                                     opt.dataset) + '.pth'))
    torch.save(D_classifier.state_dict(), os.path.join(opt.savingroot,opt.dataset,'chkpts/D_classifier'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.pth'))
    step += 1
    return step

def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off

class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float32)


    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_ = y_.to(device, torch.long)
    return z_, y_

def GAN_training_function(G, D, GD, z_, y_, config):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config.batch_size)

        y = torch.split(y, config.batch_size)
        counter = 0

        # Optionally toggle D and G's "require_grad"

        toggle_grad(D, True)
        toggle_grad(G, False)

        for step_index in range(config.num_D_steps):
            z_.sample_()
            y_.sample_()
            D_fake, D_real = GD(z_[:config.batch_size], y_[:config.batch_size],
                                x[counter], y[counter], train_G=False)

            D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
            D_loss = (D_loss_real + D_loss_fake)
            D_loss.backward()
            # counter += 1
            D.optim.step()

        # Optionally toggle "requires_grad"
        toggle_grad(D, False)
        toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        z_.sample_()
        y_.sample_()
        D_fake = GD(z_, y_, train_G=True)
        G_loss = losses.generator_loss(D_fake)

        G_loss.backward()
        G.optim.step()

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train

def train_g_cifar(netd_g, netg, loader, epoch, step, opt):
    noise, fake_label = prepare_z_y(G_batch_size=opt.batch_size, dim_z=opt.nz, nclasses=10)

    G_D_net = G_D(G=netg, D=netd_g)
    train = GAN_training_function(G=netg, D=netd_g, GD=G_D_net, z_=noise, y_=fake_label, config=opt)

    # pbar = tqdm(enumerate(loader), dynamic_ncols=True)
    #
    # for _, (image_c, label) in pbar:
    for _, (image_c, label) in enumerate(loader):
        netg.train()
        netd_g.train()
        image_c = image_c.cuda()
        label = label.cuda()

        metrics = train(image_c, label)

        step = step + 1

        G_loss = metrics['G_loss']
        D_loss_real = metrics['D_loss_real']
        D_loss_fake = metrics['D_loss_fake']


        # pbar.set_description(
        #     (f'{step}; G_loss: {G_loss:.5f};' f' D_loss_real: {D_loss_real:.5f} 'f' D_loss_fake: {D_loss_fake:.5f}')
        # )

        print('G_loss', G_loss, step)
        print('D_loss_real', D_loss_real, step)
        print('D_loss_fake', D_loss_fake, step)

    #######################
    # save image pre epoch
    #######################
    # torch.save(netg.state_dict(), os.path.join(opt.savingroot, opt.dataset, f'chkpts/g_{epoch:03d}.pth'))
    # torch.save(netg.state_dict(), os.path.join(opt.savingroot, opt.dataset,
    #                                            'chkpts/g_' + str(epoch) + 'alpha' + str(opt.alpha) + str(
    #                                                opt.dataset) + '.pth'))
    if epoch % 10 == 0 or epoch == opt.g_epochs-1:
        torch.save(netg.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                   'chkpts/g_' + str(epoch) + 'alpha' + str(opt.alpha) + str(
                                                       opt.dataset) + '.pth'))
        torch.save(netd_g.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                     'chkpts/d_g' + str(epoch) + 'alpha' + str(opt.alpha) + str(
                                                         opt.dataset) + '.pth'))

    return step

def train_g(netd_g,netd_c,netg,optd_g,optg,loader,epoch,step,opt,D_classifier,netd_c_mi,D_classifier_mi,optd_c,optd_mi):

    g_rate = 1
    netg.train()
    netd_g.train()
    # netd_c.train()
    if opt.dataset == 'mnist':
        d_steps = 128
        g_steps = 1
    else:
        d_steps = 4
        g_steps = 1
    for _, (image_g, target) in enumerate(loader):

        # #######################
        # # real input and label
        # #######################
        image_g = image_g.float()
        # print(image_g.size())
        if opt.dataset=='mnist':
            real_input_g = image_g.view(-1,32*32).cuda()
        else:
            real_input_g = image_g.cuda()

        target = target.type(torch.LongTensor).cuda()
        real_ = torch.ones(real_input_g.size()[0]).cuda()
        real_ = Variable(real_.cuda())
        fake_ = torch.zeros(real_input_g.size()[0]).cuda()
        fake_ = Variable(fake_.cuda())
        #
        #
        #######################
        # # fake input and label
        # #######################
        for i in range(d_steps):
            netd_g.train()
            netg.eval()
            noise = torch.Tensor(real_input_g.size()[0], opt.nz).normal_(0, 1).cuda()
            fake_label =torch.LongTensor(real_input_g.size()[0]).random_(10).cuda()
            fake_label_one = torch.zeros(real_input_g.size()[0],10).scatter_(1, fake_label.type(torch.LongTensor).view(real_input_g.size()[0],-1), 1).cuda()
            #
            # #######################
            # # update net d
            # #######################
            fake_label_one = fake_label_one.type(torch.LongTensor)
            fake_label_one = Variable(fake_label_one.cuda())
            fake_input = netg(noise,fake_label)
            real_pred,c_real,_ = netd_g(real_input_g)
            fake_pred,c_fake,mi_fake = netd_g(fake_input.detach())

            real_loss_c = F.cross_entropy(c_real, target)  #
            loss_mi = F.cross_entropy(mi_fake, fake_label)
            # if opt.dataset == 'mnist':
            real_loss_g = F.binary_cross_entropy(real_pred, real_)
                # fake_loss = (F.binary_cross_entropy(fake_pred, fake_)*g_rate + F.cross_entropy(fake_cls, fake_label)) / 2 #
            fake_loss = F.binary_cross_entropy(fake_pred, fake_)
            # else:
            #     real_loss_g, fake_loss = losses.discriminator_loss(fake_pred, real_pred)
            print("Epochs : ", epoch,"real loss",real_loss_g)
            print("Epochs : ", epoch,"fake loss",fake_loss)
            real_loss_g = (real_loss_g + fake_loss + real_loss_c + loss_mi)
            # real_loss_g = (real_loss_g + fake_loss)
            optd_g.zero_grad()

            real_loss_g.backward()
            optd_g.step()



        d_loss = real_loss_g

        ######################
        # update net g
        ######################

        for g_iter in range(g_steps):

            optg.zero_grad()
            netd_g.eval()
            netg.train()
            D_classifier.eval()
            netd_c.eval()
            fake_label = torch.LongTensor(real_input_g.size()[0]).random_(10).cuda()
            noise = Variable(torch.Tensor(real_input_g.size()[0], opt.nz).normal_(0, 1)).cuda()

            fake_input = netg(noise, fake_label)

            fake_pred,c_fake,mi_fake  = netd_g(fake_input)

            # c_fake = D_classifier(netd_c(fake_input))
            loss_c = F.cross_entropy(c_fake, fake_label)  #
            # mi_fake = D_classifier_mi(netd_c_mi(fake_input))
            loss_mi = F.cross_entropy(mi_fake, fake_label)
            # if opt.dataset == 'mnist':
            g_loss = F.binary_cross_entropy(fake_pred, real_)
            # else:
            #     g_loss = losses.generator_loss(fake_pred)
            g_loss  = g_loss + loss_c - loss_mi

            g_loss.backward()
            optg.step()

        step = step + 1

    #######################
    # save image pre epoch
    #######################
    if not os.path.exists(os.path.join(opt.savingroot,opt.dataset,'images')):
        os.makedirs(os.path.join(opt.savingroot,opt.dataset,'images'))
    torchvision.utils.save_image(denorm(fake_input.data), os.path.join(opt.savingroot,opt.dataset,'images/fake_'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.jpg'))
    # utils.save_image(denorm(real_input.data), f'images/real_{epoch:03d}.jpg')

    #######################
    # save model pre epoch
    #######################
    if not os.path.exists(os.path.join(opt.savingroot,opt.dataset,'chkpts')):
        os.makedirs(os.path.join(opt.savingroot,opt.dataset,'chkpts'))
    if epoch % 10 == 0 or epoch == opt.g_epochs - 1:
        torch.save(netg.state_dict(), os.path.join(opt.savingroot,opt.dataset,'chkpts/g_'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.pth'))
        torch.save(netd_g.state_dict(), os.path.join(opt.savingroot,opt.dataset,'chkpts/d_g'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.pth'))
    # torch.save(netd_c.state_dict(), os.path.join(opt.savingroot,opt.dataset,'chkpts/d_c'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.pth'))
    # torch.save(D_classifier.state_dict(), os.path.join(opt.savingroot,opt.dataset,'chkpts/D_classifier'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.pth'))

    return step


def test(netg,fixed,epoch,opt):
    netg.eval()
    #
    # fixed = Variable(torch.Tensor(10*opt.num_class, opt.nz).normal_(0, 1)).cuda()
    # label = Variable(torch.LongTensor([range(10)] * opt.num_class)).view(-1).cuda()
    # label_one = torch.zeros(10*opt.num_class, opt.num_class).scatter_(1, label.type(torch.LongTensor).view(10*opt.num_class,-1), 1).cuda()
    # label_one = torch.argmax(label_one,dim=1).view(-1)
    # label_one = Variable(label_one.cuda())
    # fixed_input = netg(fixed,label_one)
    toggle_grad(netg, False)
    if not os.path.exists(os.path.join(opt.savingroot,opt.dataset,'images')):
        os.makedirs(os.path.join(opt.savingroot,opt.dataset,'images'))
    for i in range(opt.num_class):
        fixed = torch.randn(10, opt.nz).cuda()
        label = torch.ones(10).long().cuda() * i
        if i == 0:
            fixed_input = netg(fixed, label).cpu()
        else:
            fixed_input = torch.cat([fixed_input, netg(fixed, label).cpu()], dim=0)
    # fixed_input = fixed_input.view(-1,1,32,32)
    torchvision.utils.save_image(denorm(fixed_input.data), os.path.join(opt.savingroot,opt.dataset,'images/fixed_'+str(epoch)+'alpha'+str(opt.alpha)+str(opt.dataset)+'.jpg'), nrow=10)
    toggle_grad(netg, True)
def test_acc(model,D_classifier, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.float().cuda(), target.type(torch.LongTensor).cuda()
            # plt.imshow(np.transpose((data[0].cpu().numpy()+1)/2,[1,2,0]))
            # plt.show()
            output = D_classifier(model(data))
            test_loss += F.nll_loss(output, target).sum().item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)*1.0))

    return correct / len(test_loader.dataset)*1.0