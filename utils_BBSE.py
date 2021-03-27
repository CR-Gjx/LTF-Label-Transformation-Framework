import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
def predict_all(model1,model2, device,test_loader):
    '''
    :param X: an ndarray containing the data. The first axis is over examples
    :param net: trained model
    :param dfeat: the dimensionality of the vectorized feature
    :param batchsize: batchsize used in iterators. default is 64.
    :return: Two ndarrays containing the soft and hard predictions of the classifier.
    '''

    ypred_soft=[]
    ypred=[]
    y_true = []
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float()
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            output = model2(model1(data))
            # test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            output = F.softmax(output,dim=1)
            # print(pred)
            ypred_soft.append(output.cpu().numpy())
            ypred.append(pred.cpu().numpy().squeeze())
            y_true.append(target.cpu().numpy().squeeze())

    ypred_soft_all = np.concatenate(ypred_soft, axis=0)
    ypred_all = np.concatenate(ypred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    return ypred_all, ypred_soft_all,y_true

def idx2onehot(a,k):
    a=a.astype(int)
    b = np.zeros((a.size, k))
    b[np.arange(a.size), a] = 1
    return b



def confusion_matrix_probabilistic(ytrue, ypred,k):
    # Input is probabilistic classifiers in forms of n by k matrices
    n,d = np.shape(ypred)
    C = np.dot(ypred.T, idx2onehot(ytrue,k))
    return C/n


def calculate_marginal(y,k):
    mu = np.zeros(shape=(k))
    for i in range(k):
        mu[i] = np.count_nonzero(y == i)
    return mu/np.size(y)

def calculate_marginal_probabilistic(y,k):
    return np.mean(y,axis=0)

def estimate_labelshift_ratio(ytrue_s, ypred_s, ypred_t,k):
    if ypred_s.ndim == 2: # this indicates that it is probabilistic
        C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
        mu_t = calculate_marginal_probabilistic(ypred_t, k)
    else:
        C = confusion_matrix(ytrue_s, ypred_s,labels=np.array(range(k)))
        C = C.T/len(ypred_s)
        print(C)
        print("BBSE hard")
        mu_t = calculate_marginal(ypred_t, k)
    print(mu_t)
    lamb = (1/min(len(ypred_s),len(ypred_t)))
    wt = np.linalg.solve(np.dot(C.T, C)+lamb*np.eye(k), np.dot(C.T, mu_t))
    return wt

def estimate_target_dist(wt, ytrue_s,k):
    ''' Input:
    - wt:    This is the output of estimate_labelshift_ratio)
    - ytrue_s:      This is the list of true labels from validation set

    Output:
    - An estimation of the true marginal distribution of the target set.
    '''
    mu_t = calculate_marginal(ytrue_s,k)
    return wt*mu_t

# functions that convert beta to w and converge w to a corresponding weight function.
def beta_to_w(beta, y, k):
    w = []
    for i in range(k):
        w.append(np.mean(beta[y.astype(int) == i]))
    w = np.array(w)
    return w

# a function that converts w to beta.
def w_to_beta(w,y):
    return w[y.astype(int)]

def w_to_weightfunc(w):
    return lambda x, y: w[y.astype(int)]
