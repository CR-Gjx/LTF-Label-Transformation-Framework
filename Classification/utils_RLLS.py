from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import torchvision
import cvxpy as cp
# from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import os
import copy

def calculate_confusion(predictions,train_labels,n_class):
    m_train = predictions.shape[0]
    C_yy = np.zeros(shape=(n_class,n_class))
    for i in range(n_class):
        for j in range(n_class):
            C_yy[i, j] = float(len(np.where((predictions == i) & (train_labels == j))[0])) / m_train
    return C_yy

def calculate_mu_y(predictions,n_class,m_train):
    mu_y_train_hat = np.zeros(n_class)
    for i in range(n_class):
        mu_y_train_hat[i] = float(len(np.where(predictions == i)[0])) / m_train
    return mu_y_train_hat

def compute_w_inv(C_yy, mu_y):
    # compute weights

    try:
        w = np.matmul(np.linalg.inv(C_yy), mu_y)
        print('Estimated w is', w)
        # fix w < 0
        w[np.where(w < 0)[0]] = 0
        print('If there is negative w, fix with 0:', w)
        return w
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print('Cannot compute using matrix inverse due to singlar matrix, using psudo inverse')
            w = np.matmul(np.linalg.pinv(C_yy), mu_y)
            w[np.where(w < 0)[0]] = 0
            return w
        else:
            raise RuntimeError("Unknown error")

def compute_y_feature(C, feature_test):
    n = C.shape[1]
    theta = cp.Variable(n)
    print("theta: ",theta)
    objective = cp.Minimize(cp.pnorm(C * theta - feature_test))
    constraints = [theta >= 0 ,
                  cp.sum(theta) == 1]
    # constraints = [theta >= 0]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    # w = 1 + theta.value * labda
    y = theta.value
    print('Estimated y is', y)

    return y

def compute_w_feature(C, feature_test, feature_train, rho, labda=1):
    n = C.shape[1]
    theta = cp.Variable(n)
    print("theta: ",theta)
    b = feature_test - feature_train
    objective = cp.Minimize(cp.pnorm(C * theta - b) + rho * cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    w = 1 + theta.value * labda
   # y = theta.value
    print('Estimated w_feature is', w)

    return w

def compute_w_opt(C_yy, mu_y, mu_train_y, rho, labda=1):
    n = C_yy.shape[0]
    theta = cp.Variable(n)
    print("theta: ",theta)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy * theta - b) + rho * cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    w = 1 + theta.value * labda

    print('Estimated w is', w)

    return w

def compute_3deltaC(n_class, n_train, delta):
    rho = 3 * (2 * np.log(2 * n_class / delta) / (3 * n_train) + np.sqrt(2 * np.log(2 * n_class / delta) / n_train))
    return rho


def choose_alpha(n_class, C_yy, mu_y, mu_y_train, rho, true_w):
    alpha = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    w2 = np.zeros((len(alpha), n_class))
    for i in range(len(alpha)):
        w2[i, :] = compute_w_opt(C_yy, mu_y, mu_y_train, alpha[i] * rho)
    mse2 = np.sum(np.square(np.matlib.repmat(true_w, len(alpha), 1) - w2), 1) / n_class
    i = np.argmin(mse2)
    print("mse2, ", mse2)
    return alpha[i]


def compute_true_w(train_labels, test_labels, n_class, m_train, m_test):
    # compute the true w
    mu_y_train = np.zeros(n_class)
    for i in range(n_class):
        mu_y_train[i] = float(len(np.where(train_labels == i)[0])) / m_train
    mu_y_test = np.zeros(n_class)
    for i in range(n_class):
        mu_y_test[i] = float(len(np.where(test_labels == i)[0])) / m_test
    true_w = mu_y_test / mu_y_train
    print('True w is', true_w)
    return true_w


def compute_naive_w(train_labels, n_class, m_train):
    # compute naive label ratio just using the training labels
    mu_y_train = np.zeros(n_class)
    for i in range(n_class):
        mu_y_train[i] = float(len(np.where(train_labels == i)[0])) / m_train
    mu_y_test = 0.1 * np.ones(n_class)
    true_w = mu_y_test / mu_y_train
    print('Naive w is', true_w)
    return true_w


def compute_w_tls(C_yy, mu_y, mu_train_y):
    '''
    TLS
    '''
    n = C_yy.shape[0]
    b = mu_y

    obj = np.zeros((n, n + 1))
    obj[0:n, 0:n] = C_yy
    obj[0:n, -1] = b
    # SVD
    u, s, vh = np.linalg.svd(obj, full_matrices=True)
    # calculate theta
    vxx = vh[0:n, -1]
    vyy = vh[-1, -1]
    theta = -vxx / vyy

    w = theta

    w[w <= 0] = 0
    print('Estimated w is', w)
    return w