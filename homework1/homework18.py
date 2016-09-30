from homework15_IO import *
from homework16 import *
from pocket import *

# Variable
trainingTimes = 200


if __name__ == "__main__":
    # Variable
    errors = []
    WPocket = None

    # Get the data
    D, X, Y = read(fileName="18_train.dat")
    DTest, XTest, YTest = read(fileName="18_test.dat")

    # Experiment
    for i in range(trainingTimes):
        X, Y = shuffle(X, Y)
        #error = pocket(X, Y, XTest, YTest)
        error = pocket2(X, Y, XTest, YTest)
        #print "Epoch: ", i, "\tError: ", error
        errors.append(error)
    print "Error Rate: ", float( np.sum(np.array(errors)) / trainingTimes ) 
    pocketDraw("Error Rate(update time: 50, with pocket weight)", errors)