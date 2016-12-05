from q13 import *
import sys

if __name__ == "__main__":
    # load and initialize the variable
    load()
    minAlpha = sys.maxint
    minEin = sys.maxint
    minEout = sys.maxint

    # Test for different lambda
    for i in range(-10, 3, 1):
        ridgeRegression(pow(10, i))
        if Ein() <= minEin:
            minAlpha = i
            minEin = Ein()
            minEout = Eout()

    print "Min lambda: ", minAlpha
    print "Min Ein   : ", minEin
    print "Min Eout  : ", minEout