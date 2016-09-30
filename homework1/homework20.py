from homework15_IO import *
from homework16 import *
import pocket as p

# lambda
sign = lambda x: 1 if x > 0 else -1

# Constant
threshold = 1
BETTER = 1
WORST = 0

# Variable
trainingTimes = 20
updateTimes = 100
X = []
Y = []
DTest, XTest, YTest = read(fileName="18_test.dat")

def validate(W, W_):
    """
        Validate
    """
    # Count old
    old = 0
    for i in range(len(XTest)):
        if not sign(np.dot(X[i], W)) == Y[i]:
            old += 1 

    # Count new
    new = 0
    for i in range(len(XTest)):
        if not sign(np.dot(X[i], W_)) == Y[i]:
            new += 1 

    if old > new or old == new:
        return BETTER
    else:
        return WORST


def pocket18(X, Y, eta=0.5):
    """
        Pocket algorithm by revising the PLA algorithm

        Ret:    The error rate and the pocket weight
    """
    W = np.zeros(5)
    WPocket = W
    W[0] = threshold

    # Pocket algorithm
    updateTime = 0
    index = 0
    while updateTime < updateTimes:
        res = sign(np.dot(X[index], W))
        if not res == Y[index]:
            W = W + eta * np.array(X[index]) * Y[index]
            if validate(WPocket, W) == BETTER:
                WPocket = W
            updateTime += 1
        index = ( index + 1 ) % ( len(X)-1 )

    # Compute the error rate for the testing data
    minCount = 0
    for i in range(len(X)):
        if not sign(np.dot(X[i], WPocket)) == Y[i]:
            minCount += 1
    return float(minCount) / 500, WPocket

if __name__ == "__main__":
    # Variable
    errors = []
    WPocket = None

    D, X, Y = read(fileName="18_train.dat")
    for i in range(trainingTimes):
        X, Y = shuffle(X, Y)
        error, w_ = pocket18(X, Y, 1)
        print "Epoch: ", i, "\tError: ", error
        errors.append(error)
    print "Error Rate: ", float( np.sum(np.array(errors)) / trainingTimes ) 
    draw("Error Rate", errors)
    