import numpy as np
import pylab

sign = lambda x: 1 if x > 0 else -1

def wave(x, alpha=1):
    """
        Return the result of the wave

        Arg:    The input sample points and the threshold
        Ret:    The output shape of the function
    """
    return np.sign( np.abs( np.mod(alpha * x , 4) - 2) - 1)
    
if __name__ == "__main__":
    # Plot the hypothesis
    dots = np.linspace(-10, 10, 20)
    f = wave(dots)
    pylab.plot(f)
    pylab.ylim( [ -1.5, 1.5 ] )
    
    # Plot the points
    redX = [4.5, 7.5]
    redY = [1, -0.5]
    greenX = [4.5, 9.5]
    greenY = [-0.5, 0.5]
    pylab.plot(redX, redY, 'or', color='r')
    pylab.plot(greenX, greenY, 'or', color='g')
    
    pylab.show()