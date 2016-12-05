from q16 import *
import numpy as np
import sys

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
        if Err_withData(w, _valX, _valY) <= minEval:
            minAlpha = i
            minEtrain = Err_withData(w, _trainX, _trainY)
            minEval = Err_withData(w, _valX, _valY)
            minEout = Err_withData(w, testX, testY)

    print "Min lambda   : ", minAlpha
    print "Min Etrain   : ", minEtrain
    print "Min Eval     : ", minEval
    print "Min Eout     : ", minEout