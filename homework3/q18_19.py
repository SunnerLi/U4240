import numpy as np
import math

"""
    This code solve the problem 18 and 19
    The revision is the origin gradient decent function
"""

# The activation function
THETA = lambda x: 1 / (1 + math.exp(-x))    # The sigmoid fundtion
SIGN  = lambda x: 1 if x >= 0 else -1       # The sign function

# Training constants
iteration = 2000
eta = 0.001

# Training data variable
trainN = 1000
trainX = np.ndarray([trainN, 20])
trainY = np.ndarray([trainN, 1])

# Testing data variable
testN = 3000
testX = np.ndarray([testN, 20])
testY = np.ndarray([testN, 1])

# The time period to show the loss information
showLossLimit = 500

# Logistic model
W = np.zeros([20])

def read():
    """
        Read the training data and testing data from file
    """
    # Read training data
    with open('hw3_train.dat', 'r') as f:
        for i in range(trainN):
            string = f.readline().split()
            trainX[i] = np.asarray(string[:-1])
            trainY[i] = string[-1]

    # Read testing data
    with open('hw3_test.dat', 'r') as f:
        for i in range(testN):
            string = f.readline().split()
            testX[i] = np.asarray(string[:-1])
            testY[i] = string[-1]

def gradient(_x, _y, _n):
    """
        Calculate the gradient for the cross entropy

        Arg:    _x  - The input array
                _y  - The tag array
                _n  - The size of the training set
        Ret:    The gradient vector that can be used to revise the model
    """
    _sum = np.zeros([20])
    for index in range(_n):
        reviseLength = THETA( -1 * _y[index][0] * np.dot(_x[index], W) )
        reviseDirect = -1 * ( _y[index] * _x[index] )
        _sum += (reviseLength * reviseDirect)
    return _sum / _n

def train():
    """
        Implement model training
    """
    global W
    W = np.zeros([20])
    for i in range(iteration):
        _grad = gradient(trainX, trainY, trainN)
        W = W - eta * _grad
        if i % showLossLimit == 0:
            #print "Iter: ", i, "loss: ", np.square(np.sum(_grad))
            pass

def test():
    """
        Testing by the testing data

        Ret:    The data-out error (Eout)
    """
    count = 0.0
    for index in range(testN):
        _y = SIGN( np.dot( testX[index], W ) ) 
        if not testY[index][0] == _y:
            count += 1
    return float(count) / testN

if __name__ == "__main__":
    print "----- Q18 -----"
    read()
    train()
    print "Eout: ", test()

    print "----- Q19 -----"
    eta = 0.01
    train()
    print "Eout: ", test()
