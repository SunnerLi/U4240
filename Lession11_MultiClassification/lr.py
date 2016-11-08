import matplotlib.pyplot as plt
import plotin1 as pi1
import numpy as np
import logging
import math

# Add log configuration
localLogger = None

# Sigmoid function
THETA = lambda x: 1 / (1 + math.exp(-x))

# Training Constant
iteration = 10000
eta = 0.001
lowerBound = 0.00000000001

def getGrad(x, y, w):
    """
        Compute the gradient of the logistic regression,
        which is correspond to the specific x, y and w

        Arg:    x, y and model weight
        Ret:    The gradient vector
    """
    global localLogger
    grad = np.zeros([np.shape(x)[1]])
    for i in range(np.shape(x)[0]):
        _x = x[i]
        _y = y[i]
        thetaIn = np.dot(_x, w) * _y
        
        try:
            thetaOut = THETA(thetaIn)
        except OverflowError, e:
            localLogger.error("overflow: " + str(thetaIn))
            print "overflow: " + str(thetaIn)
            thetaOut = 0
        grad = grad + THETA( -1 * thetaIn) * -1 * _y * _x
    localLogger.debug("loss: " + str(loss(grad)))
    return grad

def loss(grad):
    """
        Return the loss about the corresponding gradient

        Arg:    The gradient vector
        Ret:    The loss float value
    """
    return np.sum(np.square(grad))

def lr(x, y, lrLogger):
    """
        Implement the logistic regression

        Arg:    The training x and y
        Ret:    The trained weight
    """
    global localLogger
    localLogger = lrLogger

    w = np.ones([np.shape(x)[1]])
    for i in xrange(iteration):
        _grad = getGrad(x, y, w)
        w = w - eta * _grad
        if loss(_grad) < lowerBound:
            break
        elif i%1000 == 0:
            print "iter: ", i, '\tloss: ', loss(_grad)
    #val(x, w)
    return w

def val(x, w):
    """
        Testing function
    """
    _res = []
    for i in range(np.shape(x)[0]):
        _res.append(  np.dot(x[i], w)   )
    print _res