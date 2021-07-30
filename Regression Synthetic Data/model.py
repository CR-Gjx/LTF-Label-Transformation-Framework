import torch
from torch import nn, optim
from torch.nn import functional as F
from spectral_norm import SpectralNorm
import numpy as np
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils import check_random_state
prrelu = nn.PReLU(num_parameters=1)
def xavier_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
class Transform(nn.Module):
    def __init__(self,d=10,input_size = 2,output_size = 1):
        super(Transform,self).__init__()
        self.fc1 = (nn.Linear(input_size,d))
        self.fc2 = (nn.Linear(d,d))
        self.fc3 = (nn.Linear(d,output_size))
        self.__initialize_weights()
    def forward(self, input,noise):
        input = torch.cat((input,noise),dim=1)
        x = self.fc1(input)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc2(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc3(x)
        x = torch.sigmoid(x)*20-10
        return x
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.fill_(0)

class random_shift(nn.Module):
    def __init__(self,input_size =1,d = 5):
        super(random_shift,self).__init__()
        self.fc1 = nn.Linear(input_size,d)
        self.fc2 = nn.Linear(d,d)
        self.fc3 = nn.Linear(d,1)
        self.__initialize_weights()
    def forward(self, input,noise):
        input = torch.cat((input, noise), dim=1)
        x = self.fc1(input)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc2(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)

class y_to_x(nn.Module):
    def __init__(self, d=10,input_size = 1,output_size = 1):
        super(y_to_x, self).__init__()
        self.fc1 = nn.Linear(input_size+output_size, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, output_size)
        self.output_size = output_size
        self.__initialize_weights()
    def forward(self, input,noise):
        input = torch.cat((input,noise),dim=1)
        x = self.fc1(input)
        # x = F.leaky_rrelu(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        # if self.training:
        x = self.fc3(x) #+torch.Tensor(input.shape[0], self.output_size).normal_(0, 1)
        # else:
            # x = self.fc3(x) + torch.Tensor(input.shape[0], self.output_size).normal_(0, 0.1)
        return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)

def project_90(x):
    return np.pi - x

def make_moons(input_y, shuffle=True, r = 1,noise=None, random_state=None,prob = 0.6):
    """Make two interleaving half circles
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    # n_samples_out = n_samples // 2
    # n_samples_in = n_samples - n_samples_out

    generator = check_random_state(random_state)

    # generate_x = np.zeros(input_y.shape)
    generate_x = np.arcsin(input_y/r)
    generate_x[np.where(input_y >= 0)] = (1 - np.cos(generate_x[np.where(input_y >= 0)]))*r
    generate_x[np.where(input_y < 0)] = (-1 + np.cos(generate_x[np.where(input_y < 0)]))*r
    X , y = generate_x, input_y
    if noise is not None:
        X += generator.normal(scale=noise, size=input_y.shape)
    print(X.shape)
    # if shuffle:
    #     X, y = util_shuffle(X , y, random_state=generator)


    return X

class regressor(nn.Module):
    def __init__(self, d=10,input_size = 1,output_size = 1):
        super(regressor, self).__init__()
        self.fc1 = nn.Linear(input_size , d)
        self.fc2 = nn.Linear(d, output_size)
        self.fc3 = nn.Linear(d, d)
        self.__initialize_weights()
    def feature_extractor(self,input):
        x = self.fc1(input)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        return x

    def forward(self, input):
        x = self.feature_extractor(input)
        x = self.fc2(x)
        return x
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, d=10,input_size = 2,output_size = 1):
        super(Generator, self).__init__()
        self.fc1 = (nn.Linear(input_size, d))
        self.fc2 = (nn.Linear(d, d))
        self.fc3 = (nn.Linear(d, output_size))

        self.batch_norm1 = (nn.BatchNorm1d(d))
        self.__initialize_weights()










    def forward(self, input,noise):
        input = torch.cat((input, noise), dim=1)
        x = self.fc1(input)
        # x = self.batch_norm1(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        # x = F.relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        # x = F.relu(x)
        x = self.fc3(x)
        # x = torch.sigmoid(x) * 20 - 10
        return x
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1)
                m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self, d=10,input_size = 1,output_size = 1):
        super(Discriminator, self).__init__()
        self.fc1 = SpectralNorm(nn.Linear(input_size, d))
        self.fc3 = SpectralNorm(nn.Linear(d, d))
        self.fc2 = SpectralNorm(nn.Linear(d, output_size))
        self.reg = SpectralNorm(nn.Linear(d, output_size))
        self.reg_mi = SpectralNorm(nn.Linear(d, output_size))
        self.batch_norm1 = (nn.BatchNorm1d(d))
        self.batch_norm2 = (nn.BatchNorm1d(d))
        self.__initialize_weights()
    def forward(self, input,noise):
        # input = torch.cat((input, predict), dim=1)
        x = self.fc1(input)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc3(x)
        x = self.batch_norm2(x)
        feature = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc2(feature)
        # feature = torch.cat((feature, noise), dim=1)
        reg_out = self.reg(feature)
        reg_out_mi = self.reg_mi(feature)
        return x,reg_out,reg_out_mi
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight_u.data.normal_(0.0, 1)
                m.bias.data.fill_(0)

class T_D(nn.Module):
    def __init__(self,input_size = 2):
        super(T_D,self).__init__()
        d = input_size
        self.fc1 = SpectralNorm(nn.Linear(input_size,d))
        self.fc2 = SpectralNorm(nn.Linear(d,1))
        self.batch_norm1 = (nn.BatchNorm1d(d))
        self.__initialize_weights()
    def forward(self, input):
        # input = torch.cat((x,y),dim=1)
        x = self.fc1(input)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x,negative_slope=0.1)
        x = self.fc2(x)
        return x
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.0, 1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight_u.data.normal_(0.0, 1)
                m.bias.data.fill_(0)

class GAN_Transform(nn.Module):
    def __init__(self, temp, x_dim, y_dim,realG):
        super(GAN_Transform, self).__init__()

        self.G = Generator(input_size=y_dim+x_dim,output_size=x_dim)
        self.real_G = realG
        # MLP network for generation of x
        self.T = Transform(input_size=y_dim*2,output_size=y_dim)

    def forward(self, y,noise):  # conditional GAN forward function
        return self.G(y,noise)

    def forward_gumbel(self, y,noise,noise_T):
        q_y = self.T(y,noise_T)
        return self.G(q_y,noise), q_y
    def forward_gumbel_realG(self, y,noise,noise_T):
        q_y = self.T(y,noise_T)
        return self.real_G(q_y,noise), q_y