import numpy as np
import os

# lambda
sign = lambda x: 1 if x > 0 else -1

# Constants
threshold = 0.5


def read(fileName="15_train.dat", threshold=0.5):
    """
        pass
    """
    # Variable
    D = []
    X = []
    Y = []

    # Get all
    f = open(fileName, 'r')
    while True:
        d = f.readline()
        if d == "":
            break
        d = d.split(' ')
        D.append(d)
    
    # Get X
    for i in range(len(D)):
        # Deal with the last dimension first
        last = len(D[i][3]) - 1
        while True:
            #print D[i][3][last]
            if not D[i][3][last] == '\t':
                last -= 1
            else:
                break
        Xi = [1, float(D[i][0]), float(D[i][1]), float(D[i][2]), float(D[i][3][:last])]
        X.append(Xi)

    # Get Y
    for i in range(len(D)):
        _tPos = len(D[i][3]) - 1
        while True:
            #print D[i][3][last]
            if not D[i][3][_tPos] == '\t':
                _tPos -= 1
            else:
                break
        #print i, "\t", D[i][3][_tPos+1:len(D[i][3])-1]
        Yi = int(D[i][3][_tPos+1:len(D[i][3])-1])
        Y.append(Yi)

    return D, X, Y

def pla(X, Y, eta=1, continue_if_wrong=False):
    """
        PLA algorithm
    """
    # Initialize the weight vector
    W = np.zeros(5)
    W[0] = threshold

    epoch = 0
    while True:
        #print "epoch: ", epoch
        noChange = True
        for i in range(len(X)):
            h_x = sign(np.dot(W, X[i]))
            if not h_x == Y[i]:
                noChange = False
                epoch += 1
                W += eta * Y[i] * np.array(X[i])
                if continue_if_wrong:
                    break
        if noChange:
            break
    return epoch+1