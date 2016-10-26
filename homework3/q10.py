from q8 import *

def newton(u, v):
    """
        Compute the NewTon direction
        (Assume we know the format of Hessian matrix)
    """
    _u = -(deltaU_1(u, v) / (deltaU_2(u, v)))
    _v = -(deltaV_1(u, v) / (deltaV_2(u, v)))
    return _u, _v

if __name__ == "__main__":
    u = 0
    v = 0
    eta = 1
    for i in range(5):
        _u, _v = newton(u, v)
        u += eta * _u
        v += eta * _v
    print E(u, v)