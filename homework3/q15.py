from q13 import *

N = 1000
time = 1000

def linearRegression():
    """
        Implement the linear regression

        Ret:    The weight vector (Wlin)
    """
    tagger = np.linalg.pinv(np.matmul(np.transpose(X), X))
    pInv = np.matmul(tagger, np.transpose(X))
    return np.matmul(pInv, Y)

def project(arr):
    """
        Project the 2-D data to the 6-dim

        Arg:    The 2-D data list
        Ret:    The 6-D data list
    """
    res = np.ndarray([N, 6])
    res[:, 0] = 1
    res[:, 1] = arr[:, 1]
    res[:, 2] = arr[:, 2]
    res[:, 3] = np.matmul(arr[:, 1], arr[:, 2])
    res[:, 4] = np.square(arr[:, 1])
    res[:, 5] = np.square(arr[:, 2])
    return res

def generateTag(x):
    """
        Generate the tag and give some probability filpping

        Arg:    The x array
        Ret:    The corresponding tags
    """
    Y = np.ndarray([N, 1])
    for i in range(N):
        Y[i][0] = f(X[i][1], X[i][2])
        if random.randint(0, 9) == 0:
            Y[i][0] = -Y[i][0]
    return Y

def predict(x):
    """
        Predict the corresponding result with the ideal model

        Arg:    The validate input
        Ret:    The corresponding result
    """
    return np.sign(np.matmul(x, W))

if __name__ == "__main__":
    # Get the high-dimension model
    X = (np.random.rand(N, 3) - 0.5) * 2
    Y = generateTag(X)
    X = project(X)
    W = linearRegression()
    print "Ein: ", Ein(Y, predict(X))

    # Measure Eout
    Eout = 0
    for i in xrange(time):
        X_ = (np.random.rand(N, 3) - 0.5) * 2
        X_ = project(X)
        Y_ = generateTag(X_)
        Ypre = predict(X_)
        Eout += Ein(Ypre, Y_)
    print "Eout: ", Eout / time