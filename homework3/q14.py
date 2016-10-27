from q13 import *

getCoef = lambda a, b: (np.sum(np.corrcoef(np.transpose(a), b))-2)/2

time = 50

def highDimensionRegression():
    """
        Implement the high-dimensional linear regression
    """
    global W
    tagger = np.linalg.pinv( np.matmul( np.transpose(X), X) )
    pInv = np.matmul(tagger, np.transpose(X))
    W = np.matmul(pInv, Y) 

if __name__ == "__main__":
    WFinal = np.zeros([6, 1])
    for i in xrange(time):
        # Generate points
        X = (np.random.rand(N, 3) - 0.5) * 2
        Y = np.ndarray([N, 1])
        X[0][:] = 1
        W = None

        # Generate tags
        for i in range(N):
            Y[i][0] = f(X[i][1], X[i][2])
            if random.randint(0, 9) == 0:
                Y[i][0] = -Y[i][0]

        # Transform to the 7-dim space
        _X = np.ndarray([N, 6])

        _X[:, 0] = 1
        _X[:, 1] = X[:, 1]
        _X[:, 2] = X[:, 2]
        _X[:, 3] = np.matmul(X[:, 1], X[:, 2])
        _X[:, 4] = np.square(X[:, 1])
        _X[:, 5] = np.square(X[:, 2])
        X = _X

        highDimensionRegression()
        WFinal = np.add(WFinal, W)
    print "Final W^T: ", np.transpose(np.round(WFinal / time, 4))

    # Show the result
    choise1 = np.asarray([[-1, -1.5, 0.08, 0.13, 0.05, 0.05]])
    print "Correlation with choise1: ", getCoef(np.round(WFinal / time, 4), choise1)

    choise2 = np.asarray([-1, -1.5, 0.08, 0.13, 0.05, 1.5])
    print "Correlation with choise2: ", getCoef(np.round(WFinal / time, 4), choise2)

    choise3 = np.asarray([-1, -0.05, 0.08, 0.13, 1.5, 15])
    print "Correlation with choise3: ", getCoef(np.round(WFinal / time, 4), choise3)

    choise4 = np.asarray([-1, -0.05, 0.08, 0.13, 15, 1.5])
    print "Correlation with choise4: ", getCoef(np.round(WFinal / time, 4), choise4)

    choise5 = np.asarray([-1, -1.5, 0.08, 0.13, 1.5, 1.5])
    print "Correlation with choise5: ", getCoef(np.round(WFinal / time, 4), choise5)

    print "The answer is the choise which has highest correlation!"