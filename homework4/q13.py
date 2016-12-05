import numpy as np

trainX = np.ndarray([200, 3])
trainY = np.ndarray([200, 1])
testX = np.ndarray([1000, 3])
testY = np.ndarray([1000, 1])
weight = None

def load():
    """
        Load the training data and testing data
    """
    global trainX
    global trainY
    global testX
    global testY
    with open('hw4_train.dat', 'r') as f:
        count = 0
        while True:
            string = f.readline().split()
            if string == []:
                break
            trainX[count][0] = 1.0
            for i in range(len(string) - 1):
                trainX[count][i + 1] = string[i]
            trainY[count][0] = string[-1]
            count += 1
    with open('hw4_test.dat', 'r') as f:
        count = 0
        while True:
            string = f.readline().split()
            if string == []:
                break
            testX[count][0] = 1.0
            for i in range(len(string) - 1):
                testX[count][i + 1] = string[i]
            testY[count][0] = string[-1]
            count += 1

def ridgeRegression(alpha):
    """
        Do the linear regression with regularization with specific lambda

        Arg :   alpha - the lambda
    """
    global weight
    inside = np.matmul(np.transpose(trainX), trainX) + alpha * np.eye(np.shape(trainX)[1])
    first = np.linalg.inv(inside)
    weight = np.matmul(np.matmul(first, np.transpose(trainX)), trainY)

def sign(x):
    """
        Sign function for single value

        Arg :   x - the value
        Ret :   The judge result
    """
    if x >= 0:
        return 1
    else:
        return -1

def Ein():
    """
        Calculate the in-sample error

        Ret :   The Ein value
    """
    errorCount = 0.0
    for i in range(np.shape(trainX)[0]):
        result = sign(np.matmul(np.transpose(weight), trainX[i]))
        if not result == trainY[i]:
            errorCount += 1
    return errorCount / np.shape(trainX)[0]

def Eout():
    """
        Calculate the out of sample error

        Ret :   The Eout value
    """
    errorCount = 0.0
    for i in range(np.shape(testX)[0]):
        result = sign(np.matmul(np.transpose(weight), testX[i]))
        if not result == testY[i]:
            errorCount += 1
    return errorCount / np.shape(testX)[0]

if __name__ == "__main__":
    load()
    ridgeRegression(10)
    print "Ein : ", Ein()
    print "Eout: ", Eout()