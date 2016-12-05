from q16 import *
import numpy as np
import sys

if __name__ == "__main__":
    # Splite 120/80
    trainX, trainY, testX, testY = returnLoad()
    minEin = sys.maxint
    minEout = sys.maxint

    # Test for different lambda
    w = regressionWithData(pow(10, 0), trainX, trainY)
    minEin = Err_withData(w, trainX, trainY)
    minEout = Err_withData(w, testX, testY)

    print "Min Ein   : ", minEin
    print "Min Eout  : ", minEout