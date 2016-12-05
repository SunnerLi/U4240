from q13 import *
import numpy as np
import sys

def returnLoad():
    trainX = np.ndarray([200, 3])
    trainY = np.ndarray([200, 1])
    testX = np.ndarray([1000, 3])
    testY = np.ndarray([1000, 1])
    with open('hw4_train.dat', 'r') as f:
        count = 0
        while True:
            string = f.readline().split()
            if string == []:
                break
            trainX[count][0] = 1.0
            for i in range(len(string) - 1):
                trainX[count][i + 1] = string[i]
            trainY[count][0] = string[-1]
            count += 1
    with open('hw4_test.dat', 'r') as f:
        count = 0
        while True:
            string = f.readline().split()
            if string == []:
                break
            testX[count][0] = 1.0
            for i in range(len(string) - 1):
                testX[count][i + 1] = string[i]
            testY[count][0] = string[-1]
            count += 1
    return trainX, trainY, testX, testY

def regressionWithData(_lambda, x, y):
    first = np.linalg.inv(np.matmul(np.transpose(x), x) + _lambda * np.eye(np.shape(x)[1]))
    return np.matmul(np.matmul(first, np.transpose(x)), y)

def Err_withData(w, x, y):
    errorCount = 0.0
    for i in range(np.shape(x)[0]):
        result = sign(np.matmul(np.transpose(w), x[i]))
        if not result == y[i]:
            errorCount += 1
    return errorCount / np.shape(x)[0]

if __name__ == "__main__":
    # Splite 120/80
    trainX, trainY, testX, testY = returnLoad()
    _trainX = trainX[:120]
    _valX = trainX[120:]
    _trainY = trainY[:120]
    _valY = trainY[120:]
    minAlpha = sys.maxint
    minEtrain = sys.maxint
    minEval = sys.maxint
    minEout = sys.maxint

    # Test for different lambda
    for i in range(-10, 3, 1):
        w = regressionWithData(pow(10, i), _trainX, _trainY)
        if Err_withData(w, _trainX, _trainY) <= minEtrain:
            minAlpha = i
            minEtrain = Err_withData(w, _trainX, _trainY)
            minEval = Err_withData(w, _valX, _valY)
            minEout = Err_withData(w, testX, testY)

    print "Min lambda   : ", minAlpha
    print "Min Etrain   : ", minEtrain
    print "Min Eval     : ", minEval
    print "Min Eout     : ", minEout