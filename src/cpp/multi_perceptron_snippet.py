"""
for l in range(1, L):
    N, M = curr.shape
           curr                              curr           prev             curr
    b[l][:M, :mB] = einsum('ij,jm->im', W[l][:M,:N], V[l-1][:N, :mB]) - Theta[l][:M]
    V[l][:M, :mB] = g(b[l][:M, :mB])
   dV[l][:M, :mB] = dg(V[l][:M, :mB])
"""  #

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint as ppr


def sigmoid(x):
    return x


def dsigmoid(x):
    return x


def init_weights(trials, M, N, targetType='0/1'):
    if trials == 0:
        shapeNN = (M, N)
    else:
        shapeNN = (trials, M, N,)

    # variance has to be 1/N, variance = stdv^2, so:
    var = 1. / N
    std = var ** 0.5
    initWji = (np.random.normal(loc=0, scale=std, size=shapeNN))

    if targetType == '-1/1':
        initWji *= 2.
        initWji -= 1.
    initWji = np.require(initWji, requirements='C')
    return initWji


def init_parameters(shapeNN, Theta, W):
    L = len(shapeNN)
    hL = L - 1

    inits = {}

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]

        Theta[l][:M] = np.zeros([M, 1], dtype=float)
        W[l][:M, :N] = init_weights(0, M, N, )
        print(M, N)

    inits['Theta'] = Theta.copy()
    inits['W'] = W.copy()

    return inits


def training_containers(shapeNN, mB):
    L = len(shapeNN)
    hL = L - 1

    b = {}
    W = {}
    V = {}
    dV = {}
    dW = {}
    Delta = {}
    Theta = {}
    dTheta = {}

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]

        b[l] = np.zeros([M, mB], dtype=float)
        W[l] = np.zeros([M, N], dtype=float)
        dW[l] = np.zeros([M, N], dtype=float)
        dV[l] = np.zeros([M, mB], dtype=float)
        V[l] = np.zeros([M, mB], dtype=float)
        Delta[l] = np.zeros([M, mB], dtype=float)
        Theta[l] = np.zeros([M, 1], dtype=float)
        dTheta[l] = np.zeros([M, 1], dtype=float)

    N = shapeNN[0]
    M = shapeNN[-1]
    V[0] = np.zeros([N, mB], dtype=float)
    V['Z'] = V[L - 1].copy()

    return W, Theta, dW, dTheta, V, dV, b, Delta


def a2s(a):
    return np.array2string(a, separator=', ').replace('[', "{").replace(']', '}')


def print_a2s_cpp(a, name):
    N, M = a.shape
    s = 'float %s[%d][%d] = ' % (name, N, M)
    s += a2s(a) + ';'
    print(s)
    return s


def train_perceptron(mB, epochs, iterations, g, dg, eta, shapeNN,
                     W, Theta, dW, dTheta, V, dV, b, Delta, ):
    L = len(shapeNN)
    HL = L - 1

    for epoch in range(epochs + 1):

        if epoch < epochs:
            for t in range(iterations):
                muIdx = 0

                N = shapeNN[0]
                M = shapeNN[-1]

                V[0][:N, :mB] = np.random.randn(N, 1, mB)[:N, muIdx]
                V['Z'][:M, :mB] = np.random.randn(M, 1, mB)[:M, muIdx]

                for l in range(1, L):
                    N = shapeNN[l - 1]
                    M = shapeNN[l]

                    print_a2s_cpp(W[l], 'W_%d'%l)
                    print_a2s_cpp(V[l - 1], 'V_%d'%(l-1))

                    b[l][:M, :mB] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :mB])# - Theta[l][:M]
                    print_a2s_cpp(b[l], 'b_%d'%l)
                    V[l][:M, :mB] = g(b[l][:M, :mB])
                    dV[l][:M, :mB] = dg(V[l][:M, :mB])

                Delta[HL][:M, :mB] = (V['Z'][:M, :mB] - V[HL][:M, :mB]) * dV[HL][:M, :mB]

                for l in reversed(range(1, HL)):
                    N = shapeNN[l]
                    M = shapeNN[l + 1]
                    Delta[l][:N, :mB] = np.einsum('im,ij->jm', Delta[l + 1][:M, :mB], W[l + 1][:M, :N]) * dV[l][:N, :mB]

                for l in range(1, L):
                    N = shapeNN[l - 1]
                    M = shapeNN[l]
                    dW[l][:M, :N] = np.einsum('im,jm->ij', Delta[l][:M, :mB], V[l - 1][:N, :mB])
                    dTheta[l][:M, 0] = np.sum(Delta[l][:M, :mB], axis=1)

                    W[l][:M, :N] += eta * dW[l][:M, :N]
                    Theta[l][:M] -= eta * dTheta[l][:M]


def FFR135_HW3_1(mB=2, epochs=1, eta=0.3, g=sigmoid, dg=dsigmoid):
    tasks = {
        # '3.1.1': [784, 10],
        '3.1.2': [4, 3, 2],
        # '3.1.3': [784, 100, 10],
        # '3.1.4': [784, 100, 100, 10],
        # '3.2': [784, 30, 30, 30, 30, 10],
    }

    np.random.seed(0)

    taskC = {}

    for task in tasks.keys():

        shapeNN = tasks[task]

        L = len(shapeNN)
        HL = L - 1

        W, Theta, dW, dTheta, V, dV, b, Delta = training_containers(shapeNN, mB)

        ppr(W)

        inits = init_parameters(shapeNN, Theta, W)

        for l in range(1, L):
            N = shapeNN[l - 1]
            M = shapeNN[l]
            W[l][:M, :N] = inits['W'][l] * 1.
            Theta[l][:M] = inits['Theta'][l] * 0.

        train_perceptron(mB, epochs, 1, g, dg, eta, shapeNN,
                         W, Theta, dW, dTheta, V, dV, b, Delta, )


FFR135_HW3_1()
