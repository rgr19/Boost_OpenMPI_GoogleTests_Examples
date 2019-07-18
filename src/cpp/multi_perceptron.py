###############################################################################
################################# common.py ###################################
###############################################################################


DEBUG = 1

print('working in file: ', __file__)
import matplotlib

# matplotlib.use('tkagg')
matplotlib.use('agg')

import time
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from pprint import pprint as pprint

np.random.seed(0)

import os

for dir in ['figs', 'logs', 'data']:
    try:
        os.mkdir(dir)
    except:
        pass


def best_parameters(W, Theta, HL):
    best = {}
    best['W'] = W.copy()
    best['Theta'] = Theta.copy()
    best['C'] = dict(Tr=1e6, Va=1e6, Te=1e6)
    return best


def init_C_H_U(epochs, L):
    C = {}
    H = {}
    U = {'Tr': {}, 'Va': {}, 'Te': {}, }
    C['Tr'] = np.zeros([epochs + 1])
    C['Va'] = np.zeros([epochs + 1])
    C['Te'] = np.zeros([epochs + 1])
    H['Tr'] = np.zeros([epochs + 1])
    H['Va'] = np.zeros([epochs + 1])
    H['Te'] = np.zeros([epochs + 1])
    for l in range(1, L):
        U['Tr'][l] = np.zeros([epochs + 1])
        U['Te'][l] = np.zeros([epochs + 1])
        U['Va'][l] = np.zeros([epochs + 1])
    return C, H, U


def init_C(epochs):
    C = {}
    C['Tr'] = np.zeros([epochs + 1])
    C['Va'] = np.zeros([epochs + 1])
    C['Te'] = np.zeros([epochs + 1])
    return C


def timefunc(name, func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    end = time.time()

    print('executing %s took: %f' % (name, end - start))
    return out


def ppr(x, lvl=-2):
    stack = traceback.extract_stack()
    filename, lineno, funcname, code = stack[lvl]
    print('%s' % code)
    pprint(x)


def pprsh(x):
    if type(x) == dict:
        x = {k: c.shape for k, c in x.items()}
    elif type(x) == list:
        x = [c.shape for c in x]
    elif type(x) == type(np.array(0)):
        x = x.shape
    ppr(x, lvl=-3)


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


def validation_containers(shapeNN, p):
    L = len(shapeNN)
    hL = L - 1

    b = {}
    V = {}

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]

        b[l] = np.zeros([M, p], dtype=float)
        V[l] = np.zeros([M, p], dtype=float)

    return b, V


def validation_full_containers(shapeNN, p):
    L = len(shapeNN)
    hL = L - 1

    b = {}
    V = {}
    dV = {}
    Delta = {}
    dTheta = {}

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]

        b[l] = np.zeros([M, p], dtype=float)
        V[l] = np.zeros([M, p], dtype=float)
        dV[l] = np.zeros([M, p], dtype=float)
        Delta[l] = np.zeros([M, p], dtype=float)
        dTheta[l] = np.zeros([M, 1], dtype=float)

    return b, V, dV, Delta, dTheta


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


def generate_mini_batches(epochs, iterations, mB, pTr):
    muIdx = np.zeros([epochs, iterations, mB], dtype=int)

    for epoch in range(epochs):
        muIdx[epoch].flat = np.random.permutation(np.arange(pTr))

    return muIdx


def sigmoid(b, out=None):
    if out is None:
        return 1. / (1. + np.exp(-1. * b))
    else:
        out[:] = 1. / (1. + np.exp(-1. * b))


def dsigmoid(sigmoid, out=None):
    if out is None:
        return sigmoid * (1.0 - sigmoid)
    else:
        out[:] = sigmoid * (1.0 - sigmoid)


def cerror(C, O, t, epoch, p, M):
    C[epoch] = 1. / float(2. * p) * np.sum(np.abs(t[:M, :p] - O[:M, :p]))


def energy(H, O, t, epoch, p, M):
    H[epoch] = 0.5 * np.sum(np.power(t[:M, :p] - O[:M, :p], 2.))


def accuracy(x):
    return (1. - x) * 100.


def learning_speed(U, dTheta, epoch, L):
    for l in range(1, L):
        U[l][epoch] = np.linalg.norm(dTheta[l])


def validate_full(g, dg, Zi, W, Theta, b, V, dV, Delta, dTheta, C, U, H, epoch, shapeNN, p):
    L = len(shapeNN)
    HL = L - 1

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]
        b[l][:M, :p] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :p]) - Theta[l][:M]
        V[l][:M, :p] = g(b[l][:M, :p])
        dV[l][:M, :p] = dg(V[l][:M, :p])

    M = shapeNN[-1]
    Delta[HL][:M, :p] = (Zi[:M, :p] - V[HL][:M, :p]) * dV[HL][:M, :p]

    for l in reversed(range(1, HL)):
        N = shapeNN[l]
        M = shapeNN[l + 1]
        Delta[l][:N, :p] = np.einsum('im,ij->jm', Delta[l + 1][:M, :p], W[l + 1][:M, :N]) * dV[l][:N, :p]

    for l in range(1, L):
        M = shapeNN[l]
        dTheta[l][:M, 0] = np.sum(Delta[l][:M, :p], axis=1)

    cerror(C, V[L - 1], Zi, epoch, p, shapeNN[-1])
    energy(H, V[L - 1], Zi, epoch, p, shapeNN[-1])
    learning_speed(U, dTheta, epoch, L)


# g, xTrainNorm, tTrain, W, Theta, bTr, VTr, C['Tr'], epoch, shapeNN, pTr
def validate(g, Zi, W, Theta, b, V, C, epoch, shapeNN, p):
    L = len(shapeNN)

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]
        b[l][:M, :p] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :p]) - Theta[l][:M]
        V[l][:M, :p] = g(b[l][:M, :p])

    cerror(C, V[L - 1], Zi, epoch, p, shapeNN[-1])


def plot_all_C(taskC, task):
    """
    Plot the classification error of the training set and of the validation set
     as a function of epoch for the four networks specified above.
    Use logarithmical scaling of the yy-axis for the plot.
    :return:
    """
    fig = plt.figure(figsize=(15, 10))
    plt.title('FFR135 HW %s : Classification error in logscale' % task)

    ax = plt.subplot(111)
    color = iter(cm.rainbow(np.linspace(0, 1, 2 * len(taskC.keys()))))
    maxC = 0
    minC = 1e6
    for shapeNN in taskC.keys():
        C = taskC[shapeNN]
        CTr = C['Tr']
        CVa = C['Va']
        c1 = next(color)
        c2 = next(color)
        ax.plot(CTr, ':', c=c1, zorder=1, lw=3, label='C(Tr) %s' % (shapeNN,))
        ax.plot(CVa, '-', c=c2, zorder=2, lw=2, label='C(Va) %s' % (shapeNN,))
        ax.plot(0, 0)
        ax.plot(30, 0)
        ax.plot(0, 50)
        maxC = max(np.max(CVa), max(np.max(CTr), maxC))
        minC = min(np.min(CVa), min(np.min(CTr), minC))

    # ax.plot(CTe, c='g', zorder=3, lw=1, label='CTe')
    plt.grid(True)
    plt.legend()
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200])

    ax.set_yscale('log')
    ax.set_ylim(minC, maxC)

    ax.set_xlabel('epoch [-]')
    ax.set_ylabel('log C [%]')

    plt.savefig("figs/FFR135_HW%s_C.png" % task)

    if ISIPYTHON:
        plt.show()


def plot_all_U(taskU, task):
    fig = plt.figure(figsize=(15, 10))
    plt.title('FFR135 HW %s : Learning speed U for l-layer' % task)
    ax = plt.subplot(111)

    nlines = 0
    for shapeNN in taskU.keys():
        layers = taskU[shapeNN].keys()
        nlines += len(layers)

    color = iter(cm.rainbow(np.linspace(0, 1, nlines)))
    maxUTr = 0
    minUTr = 1e6
    for shapeNN in taskU.keys():
        layers = taskU[shapeNN].keys()
        for l in layers:
            UTr = taskU[shapeNN][l]
            c1 = next(color)
            ax.plot(UTr, ':', c=c1, zorder=1, lw=3, label='U(Tr) L:%d, NN:%s' % (l, shapeNN,))
            maxUTr = max(np.max(UTr), maxUTr)
            minUTr = min(np.min(UTr), minUTr)
            ax.plot(0, maxUTr)

    plt.grid(True)
    plt.legend()
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, int(maxUTr)])

    ax.set_yscale('log')
    ax.set_ylim(minUTr, maxUTr * 1.05)

    ax.set_xlabel('epoch [-]')
    ax.set_ylabel('log U(l) [%]')

    plt.savefig("figs/FFR135_HW%s_U.png" % task)

    if ISIPYTHON:
        plt.show()


def plot_all_H(taskH, task):
    fig = plt.figure(figsize=(15, 10))
    plt.title('FFR135 HW %s : Energy Functions H' % task)

    ax = plt.subplot(111)
    color = iter(cm.rainbow(np.linspace(0, 1, len(taskH.keys()))))
    maxHTr = 0
    minHTr = 1e6
    for shapeNN in taskH.keys():
        H = taskH[shapeNN]
        HTr = H['Tr']
        maxHTr = max(np.max(HTr), maxHTr)
        minHTr = min(np.min(HTr), minHTr)
        c1 = next(color)
        ax.plot(HTr, ':', c=c1, zorder=1, lw=3, label='H(Tr) %s' % (shapeNN,))
        ax.plot(0, 0)
        ax.plot(30, 0)

    plt.grid(True)
    plt.legend()
    ax.set_ylim(minHTr, maxHTr * 1.05)

    ax.set_xlabel('epoch [-]')
    ax.set_ylabel('H')

    plt.savefig("figs/FFR135_HW%s_H.png" % task)

    if ISIPYTHON:
        plt.show()


def plot_activation(mB, t, epoch, task, b, V, dV, shapeNN, L):
    if t == 0 and epoch == 29:
        fig = plt.figure()
        plt.title('Task:%s, activation at epoch[%d]' % (task, epoch))

    for l in range(1, L):
        N = shapeNN[l - 1]
        M = shapeNN[l]

        if t == 0 and epoch == 29:
            plt.plot(np.mean(b[l][:M, :mB], axis=1), np.mean(V[l][:M, :mB], axis=1), '.',
                     label='V[l=%d][M=%d][task:%s]' % (l, M, task))
            plt.plot(np.mean(b[l][:M, :mB], axis=1), np.mean(dV[l][:M, :mB], axis=1), '.',
                     label='dV[l=%d][M=%d][task:%s]' % (l, M, task))

    if t == 0 and epoch == 29:
        plt.legend()
        plt.savefig('figs/ACTIVATION(EPOCH=%d)_HW%s.png' % (epoch, task))


###############################################################################
################################# loadMNIST.py ################################
###############################################################################

def ipython_info():
    import sys
    ip = False
    if 'ipykernel' in sys.modules:
        ip = True  # 'notebook'
    # elif 'IPython' in sys.modules:
    #    ip = 'terminal'
    return ip


import os
import gzip
import numpy as np

try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve


DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

LOG = open("logs/logLoadMNIST.txt", 'w')

np.random.seed(0)


def logprint(txt):
    LOG.write(txt + '\n')
    print(txt)


# Download and import the MNIST dataset from Yann LeCun's website.
# Reserve 10,000 examples from the training set for validation.
# Each image is an array of 784 (28x28) float values  from 0 (white) to 1 (black).
def load_data():
    xTrain = load_images('train-images-idx3-ubyte.gz', 'data/')
    tTrain = load_labels('train-labels-idx1-ubyte.gz', 'data/')
    xTest = load_images('t10k-images-idx3-ubyte.gz', 'data/')
    tTest = load_labels('t10k-labels-idx1-ubyte.gz', 'data/')

    pVa = int(1e4)
    pTr = xTrain.shape[0]

    idx = np.random.permutation(np.arange(pTr))
    idxValid = idx[:pVa]
    idxTrain = idx[pVa:]

    xValid = xTrain[idxValid]
    tValid = tTrain[idxValid]

    xTrain = xTrain[idxTrain]
    tTrain = tTrain[idxTrain]

    tTrain = to_one_hot(tTrain)
    tValid = to_one_hot(tValid)
    tTest = to_one_hot(tTest)

    return xTrain, xValid, xTest, tTrain, tValid, tTest


def load_images(filename, path=''):
    maybe_download(filename, path)
    with gzip.open(path + filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / np.float32(256)


def load_labels(filename, path=''):
    maybe_download(filename, path)
    with gzip.open(path + filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# Download the file, unless it's already here.
def maybe_download(filename, path=''):
    if not os.path.exists(path + filename):
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, path + filename)


# Convert class labels from scalars to one-hot vectors.
def to_one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def norm_data(xTrain, xValid, xTest, ):
    pTr = float(xTrain.shape[0])
    xTrainAvg = (1. / pTr) * np.sum(xTrain, axis=0)

    xTrainNorm = np.around(xTrain - xTrainAvg, 5)
    xValidNorm = np.around(xValid - xTrainAvg, 5)
    xTestNorm = np.around(xTest - xTrainAvg, 5)

    return xTrainNorm, xValidNorm, xTestNorm


def get_MNIST():
    xTrain, xValid, xTest, tTrain, tValid, tTest = load_data()
    xTrainNorm, xValidNorm, xTestNorm = norm_data(xTrain, xValid, xTest)

    MNIST = dict()

    MNIST['xTrain'] = xTrain.T
    MNIST['xValid'] = xValid.T
    MNIST['xTest'] = xTest.T

    MNIST['tTrain'] = tTrain.T
    MNIST['tValid'] = tValid.T
    MNIST['tTest'] = tTest.T

    MNIST['xTrainNorm'] = xTrainNorm.T
    MNIST['xValidNorm'] = xValidNorm.T
    MNIST['xTestNorm'] = xTestNorm.T

    MNIST['shape'] = dict(pTr=xTrain.shape[0],
                          pVa=xValid.shape[0],
                          pTe=xTest.shape[0],
                          N=xTrain.shape[1],
                          M=tTrain.shape[1]
                          )

    return MNIST


if __name__ == '__main__':

    xTrain, xValid, xTest, tTrain, tValid, tTest = load_data()
    xTrainNorm, xValidNorm, xTestNorm = norm_data(xTrain, xValid, xTest)

    print(xTrain.shape, xValid.shape, xTest.shape)
    print(tTrain.shape, tValid.shape, tTest.shape)
    print(xTrainNorm.shape, xValidNorm.shape, xTestNorm.shape)

    tmpStr = "RAW xTrain: var=%f, mean=%f, std=%f" % (
        np.var(xTrain), np.mean(xTrain), np.std(xTrain))
    tmpStr += "\nRAW xValid: var=%f, mean=%f, std=%f" % (
        np.var(xValid), np.mean(xValid), np.std(xValid))
    tmpStr += "\nRAW xTestNorm: var=%f, mean=%f, std=%f" % (
        np.var(xTest), np.mean(xTest), np.std(xTest))
    logprint(tmpStr)

    tmpStr = "NORMALIZED xTrain: var=%f, mean=%f, std=%f" % (
        np.var(xTrainNorm), np.mean(xTrainNorm), np.std(xTrainNorm))
    tmpStr += "\nNORMALIZED xValid: var=%f, mean=%f, std=%f" % (
        np.var(xValidNorm), np.mean(xValidNorm), np.std(xValidNorm))
    tmpStr += "\nNORMALIZED xTestNorm: var=%f, mean=%f, std=%f" % (
        np.var(xTestNorm), np.mean(xTestNorm), np.std(xTestNorm))
    logprint(tmpStr)

    if ISIPYTHON:
        fig = plt.figure()
        for i in range(100):
            fig.add_subplot(10, 10, i + 1)
            plt.imshow(xTrainNorm[i].reshape([28, 28]), cmap='gray')

        fig = plt.figure()
        for i in range(100):
            fig.add_subplot(10, 10, i + 1)
            plt.imshow(xValidNorm[i].reshape([28, 28]), cmap='gray')

        fig = plt.figure()
        for i in range(100):
            fig.add_subplot(10, 10, i + 1)
            plt.imshow(xTestNorm[i].reshape([28, 28]), cmap='gray')
        plt.show()


###############################################################################
################################# perceptron.py ###############################
###############################################################################


def ipython_info():
    import sys
    ip = False
    if 'ipykernel' in sys.modules:
        ip = True  # 'notebook'
    # elif 'IPython' in sys.modules:
    #    ip = 'terminal'
    return ip


ISIPYTHON = ipython_info()


def train_perceptron(logprint, task, mB, epochs, iterations, g, dg, eta, shapeNN, MNIST, C, bests,
                     MuIdx, W, Theta, dW, dTheta, V, dV, b, Delta,
                     bTr, VTr, bVa, VVa, bTe, VTe):
    xTrainNorm = MNIST['xTrainNorm']
    xValidNorm = MNIST['xValidNorm']
    xTestNorm = MNIST['xTestNorm']

    tTrain = MNIST['tTrain']
    tValid = MNIST['tValid']
    tTest = MNIST['tTest']

    pTr = MNIST['shape']['pTr']
    pTe = MNIST['shape']['pTe']
    pVa = MNIST['shape']['pVa']

    L = len(shapeNN)
    HL = L - 1

    VTr[0] = xTrainNorm
    VVa[0] = xValidNorm
    VTe[0] = xTestNorm

    for epoch in range(epochs + 1):
        validate(g, tTrain, W, Theta, bTr, VTr, C['Tr'], epoch, shapeNN, pTr)
        validate(g, tValid, W, Theta, bVa, VVa, C['Va'], epoch, shapeNN, pVa)
        validate(g, tTest, W, Theta, bTe, VTe, C['Te'], epoch, shapeNN, pTe)

        tmpStr = 'task:[%s] NN:%s [epoch:%d] : Ctr[%f%s], Cva[%f%s], Cte[%f%s], CteBest[%f%s]'

        if C['Te'][epoch] < bests['C']['Te']:
            bests['C']['Tr'] = C['Tr'][epoch]
            bests['C']['Te'] = C['Te'][epoch]
            bests['C']['Va'] = C['Va'][epoch]
            for l in range(1, L):
                bests['W'][l][:] = W[l]
                bests['Theta'][l][:] = Theta[l]

            tmpStr += ' <<< NEW BEST'

        logprint(
            tmpStr % (task, shapeNN, epoch,
                      100 * C['Tr'][epoch], "%", 100 * C['Va'][epoch], "%",
                      100 * C['Te'][epoch], '%', 100 * bests['C']['Te'], '%')
        )

        if epoch < epochs:
            for t in range(iterations):
                muIdx = MuIdx[epoch, t]

                N = shapeNN[0]
                M = shapeNN[-1]
                V[0][:N, :mB] = xTrainNorm[:N, muIdx]
                V['Z'][:M, :mB] = tTrain[:M, muIdx]

                for l in range(1, L):
                    N = shapeNN[l - 1]
                    M = shapeNN[l]
                    b[l][:M, :mB] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :mB]) - Theta[l][:M]
                    V[l][:M, :mB] = g(b[l][:M, :mB])
                    dV[l][:M, :mB] = dg(V[l][:M, :mB])

                if 0:
                    plot_activation(mb, t, epoch, task, b, V, dV)

                Delta[HL][:M, :mB] = (V['Z'][:M, :mB] - V[HL][:M, :mB]) * dV[HL][:M, :mB]

                print(muIdx)
                print(V['Z'][:M, :mB])
                print(V[HL][:M, :mB])
                print(np.argmax(V['Z'][:M, :mB], axis=0), np.argmax(V[HL][:M, :mB], axis=0))

                exit(0)

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


def perceptron_learning_speed(logprint, task, mB, epochs, iterations,
                              g, dg, eta, shapeNN, MNIST, C, U, H, bests,
                              MuIdx, W, Theta, dW, dTheta, V, dV, b, Delta,
                              bTr, VTr, dVTr, DeltaTr, dThetaTr, bVa, VVa, bTe, VTe, ):
    xTrainNorm = MNIST['xTrainNorm']
    xValidNorm = MNIST['xValidNorm']
    xTestNorm = MNIST['xTestNorm']
    tTrain = MNIST['tTrain']
    pTr = MNIST['shape']['pTr']

    L = len(shapeNN)
    HL = L - 1

    VTr[0] = xTrainNorm
    VVa[0] = xValidNorm
    VTe[0] = xTestNorm

    for epoch in range(epochs + 1):
        validate_full(g, dg, tTrain, W, Theta, bTr, VTr, dVTr, DeltaTr, dThetaTr,
                      C['Tr'], U['Tr'], H['Tr'], epoch, shapeNN, pTr)

        tmpStr = 'task:[%s] NN:%s [epoch:%d] : Ctr[%f%s], Htr[%f%s]'

        if C['Te'][epoch] < bests['C']['Te']:
            bests['C']['Tr'] = C['Tr'][epoch]
            bests['C']['Te'] = C['Te'][epoch]
            bests['C']['Va'] = C['Va'][epoch]
            for l in range(1, L):
                bests['W'][l][:] = W[l]
                bests['Theta'][l][:] = Theta[l]

            tmpStr += ' <<< NEW BEST'

        logprint(
            tmpStr % (task, shapeNN, epoch,
                      100 * C['Tr'][epoch], "%", 100 * H['Tr'][epoch], "%")
        )

        if epoch < epochs:
            for t in range(iterations):
                muIdx = MuIdx[epoch, t]

                N = shapeNN[0]
                M = shapeNN[-1]
                V[0][:N, :mB] = xTrainNorm[:N, muIdx]
                V['Z'][:M, :mB] = tTrain[:M, muIdx]

                for l in range(1, L):
                    N = shapeNN[l - 1]
                    M = shapeNN[l]
                    b[l][:M, :mB] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :mB]) - Theta[l][:M]
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


###############################################################################
################################# main.hw3.py ###############################
###############################################################################


def FFR135_HW3_1(taskId='3.1', mB=10, epochs=30, eta=0.3, g=sigmoid, dg=dsigmoid):
    tasks = {
        # '3.1.1': [784, 10],
        '3.1.2': [784, 30, 10],
        # '3.1.3': [784, 100, 10],
        # '3.1.4': [784, 100, 100, 10],
        # '3.2': [784, 30, 30, 30, 30, 10],
    }

    LOG = open("logs/logFFR135_HW%s.txt" % taskId, 'w')

    def logprint(txt):
        LOG.write(txt + '\n')
        print(txt)

    logprint("... get MNIST data from disk")
    MNIST = get_MNIST()

    shapeMNIST = MNIST['shape']
    pTr = shapeMNIST['pTr']
    pTe = shapeMNIST['pTe']
    pVa = shapeMNIST['pVa']

    logprint("...MNIST data shapeNN: %s" % shapeMNIST)

    iterations = shapeMNIST['pTr'] // mB

    logprint("...train for (%d) epochs and (%d) iterations at each epoch" % (epochs, iterations))

    MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

    trueMatches = np.zeros(pTr)
    falseMatches = np.zeros(pTr)

    taskC = {}

    for task in tasks.keys():

        C = init_C(epochs)

        shapeNN = tasks[task]

        L = len(shapeNN)
        HL = L - 1

        logprint("...TASK :%s NN shape: %s" % (task, shapeNN))

        bTr, VTr = validation_containers(shapeNN, pTr)
        bVa, VVa = validation_containers(shapeNN, pVa)
        bTe, VTe = validation_containers(shapeNN, pTe)

        W, Theta, dW, dTheta, V, dV, b, Delta = training_containers(shapeNN, mB)

        inits = init_parameters(shapeNN, Theta, W)

        bests = best_parameters(W, Theta, HL)

        logprint("initialize containers")

        for l in range(1, L):
            N = shapeNN[l - 1]
            M = shapeNN[l]
            W[l][:M, :N] = inits['W'][l] * 1.
            Theta[l][:M] = inits['Theta'][l] * 0.

        timefunc(task, train_perceptron, logprint, task, mB, epochs,
                 iterations, g, dg, eta, shapeNN, MNIST, C, bests,
                 MuIdx, W, Theta, dW, dTheta, V, dV, b, Delta,
                 bTr, VTr, bVa, VVa, bTe, VTe)

        taskC[tuple(shapeNN)] = C

        with open("data/FFR135_HW%s.json" % task, 'w') as fp:
            bests['bestC'] = bests['C']
            bests['C'] = C

            for key in bests.keys():
                if type(bests[key]) == np.ndarray:
                    bests[key] = bests[key].tolist()
                    continue
                for idx in bests[key]:
                    if type(bests[key][idx]) == np.ndarray:
                        bests[key][idx] = bests[key][idx].tolist()
            json.dump(bests, fp)
            fp.close()

    bests = best_parameters(W, Theta, HL)

    plot_all_C(taskC, taskId)

    df = pd.DataFrame(taskC)

    df.to_csv("data/FFR135_HW%s_C.csv" % taskId)


def FFR135_HW3_2_1(taskId='3.2.1', mB=10, epochs=300, eta=3 * 1e-3, g=sigmoid, dg=dsigmoid):
    tasks = {
        '3.2.1': [784, 30, 30, 30, 30, 10],
        # '3.2.1.2': [784, 100, 100, 100, 100, 10],
        # '3.2.1.3': [784, 392, 196, 100, 50, 10],
    }

    LOG = open("logs/logFFR135_HW%s.txt" % taskId, 'w')

    def logprint(txt):
        LOG.write(txt + '\n')
        print(txt)

    logprint("... get MNIST data from disk")
    MNIST = get_MNIST()

    shapeMNIST = MNIST['shape']
    pTr = shapeMNIST['pTr']
    pTe = shapeMNIST['pTe']
    pVa = shapeMNIST['pVa']

    logprint("...MNIST data shapeNN: %s" % shapeMNIST)

    iterations = shapeMNIST['pTr'] // mB

    logprint("...train for (%d) epochs and (%d) iterations at each epoch" % (epochs, iterations))

    MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

    taskC = {}

    for task in tasks.keys():

        C = init_C(epochs)

        shapeNN = tasks[task]

        L = len(shapeNN)
        HL = L - 1

        logprint("...TASK :%s NN shape: %s" % (task, shapeNN))

        bTr, VTr = validation_containers(shapeNN, pTr)
        bVa, VVa = validation_containers(shapeNN, pVa)
        bTe, VTe = validation_containers(shapeNN, pTe)

        W, Theta, dW, dTheta, V, dV, b, Delta = training_containers(shapeNN, mB)

        inits = init_parameters(shapeNN, Theta, W)

        bests = best_parameters(W, Theta, HL)

        logprint("initialize containers")

        for l in range(1, L):
            N = shapeNN[l - 1]
            M = shapeNN[l]
            W[l][:M, :N] = inits['W'][l] * 1.
            Theta[l][:M] = inits['Theta'][l] * 0.

        timefunc(task, train_perceptron,
                 logprint, task, mB, epochs, iterations, g, dg, eta, shapeNN, MNIST, C, bests,
                 MuIdx, W, Theta, dW, dTheta, V, dV, b, Delta,
                 bTr, VTr, bVa, VVa, bTe, VTe)

        taskC[tuple(shapeNN)] = C

        with open("data/FFR135_HW%s.json" % task, 'w') as fp:
            bests['bestC'] = bests['C']
            bests['C'] = C

            for key in bests.keys():
                if type(bests[key]) == np.ndarray:
                    bests[key] = bests[key].tolist()
                    continue
                for idx in bests[key]:
                    if type(bests[key][idx]) == np.ndarray:
                        bests[key][idx] = bests[key][idx].tolist()
            json.dump(bests, fp)
            fp.close()

    plot_all_C(taskC, taskId)

    df = pd.DataFrame(taskC)

    df.to_csv("data/FFR135_HW%s_C.csv" % taskId)


def FFR135_HW3_2_2(taskId='3.2.2', mB=10, epochs=50, eta=3 * 1e-3, g=sigmoid, dg=dsigmoid):
    tasks = {
        '3.2.2': [784, 30, 30, 30, 30, 10],
        # '3.2.2.2': [784, 100, 100, 100, 100, 10],
        # '3.2.2.3': [784, 392, 196, 100, 50, 10],
    }

    LOG = open("logs/logFFR135_HW%s.txt" % taskId, 'w')

    def logprint(txt):
        LOG.write(str(txt) + '\n')
        print(txt)

    logprint("...get MNIST data from disk")
    MNIST = get_MNIST()

    shapeMNIST = MNIST['shape']
    pTr = shapeMNIST['pTr']
    pTe = shapeMNIST['pTe']
    pVa = shapeMNIST['pVa']

    logprint("...MNIST data shapeNN: %s" % shapeMNIST)

    iterations = pTr // mB

    logprint("...train for (%d) epochs and (%d) iterations at each epoch" % (epochs, iterations))

    MuIdx = generate_mini_batches(epochs, iterations, mB, pTr)

    taskC = {}
    taskH = {}
    taskU = {}

    print(MuIdx)

    return

    for task in tasks.keys():

        shapeNN = tasks[task]

        L = len(shapeNN)
        HL = L - 1

        C, H, U = init_C_H_U(epochs, L)

        logprint("...NN shape: %s" % (shapeNN))

        bTr, VTr, dVTr, DeltaTr, dThetaTr = validation_full_containers(shapeNN, pTr)
        bVa, VVa, dVVa, DeltaVa, dThetaVa = validation_full_containers(shapeNN, pVa)
        bTe, VTe, dVTe, DeltaTe, dThetaTe = validation_full_containers(shapeNN, pTe)

        W, Theta, dW, dTheta, V, dV, b, Delta = training_containers(shapeNN, mB)

        inits = init_parameters(shapeNN, Theta, W)

        bests = best_parameters(W, Theta, HL)

        logprint("initialize containers")

        for l in range(1, L):
            N = shapeNN[l - 1]
            M = shapeNN[l]
            W[l][:M, :N] = inits['W'][l] * 1.
            Theta[l][:M] = inits['Theta'][l] * 0.

        timefunc(task, perceptron_learning_speed,
                 logprint, task, mB, epochs, iterations, g, dg, eta, shapeNN, MNIST, C, U, H, bests,
                 MuIdx, W, Theta, dW, dTheta, V, dV, b, Delta,
                 bTr, VTr, dVTr, DeltaTr, dThetaTr, bVa, VVa, bTe, VTe)

        taskC[tuple(shapeNN)] = C
        taskH[tuple(shapeNN)] = H
        taskU[tuple(shapeNN)] = U['Tr']

        with open("data/FFR135_HW%s.json" % task, 'w') as fp:
            bests['bestC'] = bests['C']
            bests['C'] = C
            bests['H'] = H
            bests['U'] = U['Tr']

            for key in bests.keys():
                if type(bests[key]) == np.ndarray:
                    bests[key] = bests[key].tolist()
                    continue
                for idx in bests[key]:
                    if type(bests[key][idx]) == np.ndarray:
                        bests[key][idx] = bests[key][idx].tolist()
                    elif type(bests[key][idx]) == dict:
                        for idy in bests[key][idx].keys():
                            if type(bests[key][idx][idy]) == np.ndarray:
                                bests[key][idx][idy] = bests[key][idx][idy].tolist()

            json.dump(bests, fp)
            fp.close()

    plot_all_U(taskU, taskId)
    plot_all_H(taskH, taskId)

    dfC = pd.DataFrame(taskC)
    dfH = pd.DataFrame(taskH)
    dfU = pd.Panel(taskU).to_frame()

    logprint(dfU)
    dfC.to_csv("data/FFR135_HW%s_C.csv" % taskId)
    dfH.to_csv("data/FFR135_HW%s_H.csv" % taskId)
    dfU.to_csv("data/FFR135_HW%s_U.csv" % taskId)


if __name__ == "__main__":

    from multiprocessing import Pool

    hwTasks = {
        FFR135_HW3_1: dict(taskId='3.1', mB=10, epochs=30, eta=0.3, g=sigmoid, dg=dsigmoid),
        # FFR135_HW3_2_1: dict(taskId='3.2.1', mB=10, epochs=50, eta=3 * 1e-3, g=sigmoid, dg=dsigmoid),
        # FFR135_HW3_2_2: dict(taskId='3.2.2', mB=10, epochs=50, eta=3 * 1e-3, g=sigmoid, dg=dsigmoid)
    }

    doMP = False

    if doMP:
        pool = Pool(processes=len(hwTasks))
        results = []

    for hwTask in hwTasks.keys():
        data = hwTasks[hwTask]
        print ('#' * 100)
        print ('#' * 20, 'RUN FFR135 HW %s : %s' % (hwTask, data))
        if doMP:
            results.append(pool.apply_async(hwTask, kwds=data))  # runs in *only* one process
        else:
            hwTask(**data)

    if doMP:
        [result.wait() for result in results]
        pool.join()
        pool.close()

    LOG.close()

    # screen -R -D
    # cd ~/W/FFR135*/H*3 && . activate py36i && python main.hw3.py
