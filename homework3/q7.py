import math

def gradientU(u, v):
    """
        Return the value of the gradient which is partual difference toward u
    """
    return math.exp(u) + (v*math.exp(u*v)) + 2*u - 2*v - 3

def gradientV(u, v):
    """
        Return the value of the gradient which is partual difference toward v
    """
    return 2*math.exp(2*v) + (u*math.exp(u*v)) - 2*u + 4*v - 2

def E(u, v):
    """
        Compute the error
    """
    return math.exp(u) + math.exp(2*v) + math.exp(u*v) + u**2 - 2*u*v + 2*math.pow(v, 2) - 3*u - 2*v

if __name__ == "__main__":
    u = 0
    v = 0
    eta = 0.01
    for i in range(5):
        gradU, gradV = gradientU(u, v), gradientV(u, v)
        u -= eta * gradU
        v -= eta * gradV
        print "Update times: ", i+1, "\t( u , v ): ( ", u, " , ", v, " )"
    print "E: ", E(u, v)