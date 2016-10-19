import matplotlib.pyplot as plt
import numpy.linalg as alg
import numpy as np

"""
    This program show the concept of linear regression
    In this implementation, I add bias that enable the line moving on the y axis
    It can find the more reasonable result

    Author: SunnerLi
    Finish: 19/10/2016
"""
N = 7                       # The number of data we want to approximate
d = 1                       # The dimension of the training data
x = np.ndarray([N, d+1])    # The training data
y = np.ndarray([N, 1])      # The target tags
w = np.ndarray([d+1, 1])    # The weight matrix
numberOfSample = 100        # The number of the regression sample points
bias = 0                    # The bias on the y axis
scale = 5                   # The scale number to extend the sample line

# The name points file
# Notice:   There're three points file. Once you change the name,
#           you should remember to revise N. Otherwise the IndexError would occur.
pointFileName = 'lr.dat'

def read():
    """
        Read the points file and store the data
    """
    global x
    global y
    count = 0
    with open(pointFileName, 'r') as f:
        while True:
            string = f.readline()
            string = string.split(' ')
            x[count][0] = 1             # for threshold
            x[count][1] = string[0]
            y[count][0] = string[1][:len(string[1])-1]
            count += 1
            if count == N:
                break

def draw():
    """
        Draw the regression sample points and the training points
    """
    # Draw regression line
    _ = np.random.rand(numberOfSample, d)
    _ = np.concatenate((_ * scale, _ * -scale))
    plt.plot(_, _*np.transpose(w) + bias, 'r', color='r')

    # Draw testing points
    plt.plot([ i[1] for i in x], y, 'or', color='g')
    plt.title("Linear Regression")
    plt.show()

def linearRegression():
    """
        The linear regression process
        It would find the pseudo-inverse weight, and get the best weight matrix
    """
    global w
    tagger = alg.pinv( np.matmul( np.transpose(x), x ) )
    pseudoInverse = np.matmul( tagger, np.transpose(x) )
    w = np.matmul( pseudoInverse, y )
    w = np.asarray([w[1]])

def biasDecisionStump():
    """
        To consider the bias on the y axis, 
        I use 1-dimension decision stump to decide the value
    """
    global bias
    minErr = 0
    minBias = 0
    _y =  np.sort(y)
    for i in range(len(_y)):
        bias = y[i]
        _Err, _bias = squareErr()
        if i == 0:
            minErr = _Err
        if _Err <= minErr:
            minErr = _Err
            minBias = _bias
    bias = [minBias] * numberOfSample * 2

def squareErr():
    """
        Compute the square error toward the regression line

        Ret:    The sum of the square error and the corresponding bias
    """
    errSum = 0
    for i in range(len(x)):
        diff = np.subtract( y[i], x[i]*np.transpose(w)+bias )
        err = np.sum(np.square(diff))
        errSum += err
    return errSum, bias

if __name__ == "__main__":
    read()
    linearRegression()
    biasDecisionStump()
    draw()