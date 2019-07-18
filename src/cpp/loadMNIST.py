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

import matplotlib.pyplot as plt

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

#LOG = open("logs/logLoadMNIST.txt", 'w')

np.random.seed(0)


def logprint(txt):
    #LOG.write(txt + '\n')
    print(txt)


# Download and import the MNIST dataset from Yann LeCun's website.
# Reserve 10,000 examples from the training set for validation.
# Each image is an array of 784 (28x28) float values  from 0 (white) to 1 (black).
def load_data():
    xTrain = load_images('train-images-idx3-ubyte.gz', 'data/')
    tTrain = load_labels('train-labels-idx1-ubyte.gz', 'data/')
    xTest = load_images('t10k-images-idx3-ubyte.gz', 'data/')
    tTest = load_labels('t10k-labels-idx1-ubyte.gz', 'data/')

    print(np.sum(xTrain))

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
        for i,d in enumerate(data.flat):
            print(i, d)
            if i > 1000:
                break
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

    exit(0)

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
