from q16 import *
import numpy as np
import sys

if __name__ == "__main__":
    trainX, trainY, testX, testY = returnLoad()
    minAlpha = sys.maxint
    minEcv = sys.maxint
    
    # Test for folder 1
    for i in range(-10, 3, 1):
        _minEcv = 0
        for j in range(5):
            # Get the cross validation data
            _trainX = np.delete(trainX, range(j*40, j*40+40), 0)
            _trainY = np.delete(trainY, range(j*40, j*40+40), 0)
            _valX = trainX[j*40 : j*40+40]
            _valY = trainY[j*40 : j*40+40]

            # Do the regression and accumulate the Ecv
            w = regressionWithData(pow(10, i), _trainX, _trainY)
            _minEcv += Err_withData(w, _valX, _valY)
        _minEcv /= 5

        # Update the minumun value
        if _minEcv <= minEcv:
            minAlpha = i
            minEcv = _minEcv

    print "----- 19 -----"
    print "Min lambda : ", minAlpha
    print "Min Ecv    : ", minEcv
    print ""

    w = regressionWithData(pow(10, minAlpha), trainX, trainY)
    print "----- 20 -----"
    print "Ein  : ", Err_withData(w, trainX, trainY)
    print "Eout : ", Err_withData(w, testX, testY)