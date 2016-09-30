from homework15_IO import *
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random
import os

# Variable
epoches = []
random.seed(datetime.now())

def shuffle(a, b):
    """
        Shuffle the two array for 2000 times

        Arg:    The input array and the tag array
        Ret:    The shuffle result 
    """
    if not len(a) == len(b):
        print "The length aren't the same...'"
        return None
    for i in range(2000):
        index1 = random.randint(0, len(a)-1)
        index2 = random.randint(0, len(b)-1)
        a[index1], a[index2] = a[index2], a[index1]
        b[index1], b[index2] = b[index2], b[index1]
    return a, b

def draw(title, array):
    """
        Draw the histogram about the pattern with the specific title

        Arg:    The title string
    """
    plt.hist(array)
    plt.title(title)
    plt.xlabel("Experiment Time")
    plt.ylabel("Epoch")
    plt.show()



if __name__ == "__main__":
    D, X, Y = read()
    
    # Do 2000 experiments
    for i in range(2000):
        X, Y = shuffle(X, Y)
        epoch = pla(X, Y)
        #print "Epoch: ", i, "\tResult: ", epoch
        epoches.append(epoch)
    
    print np.sum(np.array(epoches)) / 2000
    draw("The PLA with eta = 1", epoches)