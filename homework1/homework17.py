from homework15_IO import *
from homework16 import *
import numpy as np

epoches = []

if __name__ == "__main__":
    D, X, Y = read()
    for i in range(2000):
        X, Y = shuffle(X, Y)
        epoch = pla(X, Y, eta=0.5)
        epoches.append(epoch)
    print "The average epoch: ", np.sum(np.array(epoches)) / 2000
    draw("The eta is 0.5", epoches)