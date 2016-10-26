import math

# Parameters want to examine
buu = None
bvv = None
buv = None
bu = None
bv = None
b = None

def deltaU_1(u, v):
    return math.exp(u) + (v*math.exp(u*v)) + 2*u - 2*v - 3

def deltaV_1(u, v):
    return 2*math.exp(2*v) + (u*math.exp(u*v)) - 2*u + 4*v - 2

def deltaU_2(u, v):
    return math.exp(u) + pow(v, 2)*math.exp(u*v) + 2

def deltaV_2(u, v):
    return 4*math.exp(2*v) + pow(u, 2)*math.exp(u*v) + 4

def E(u, v):
    return math.exp(u) + math.exp(2*v) + math.exp(u*v) + u**2 - 2*u*v + 2*math.pow(v, 2) - 3*u - 2*v

def taylor(u, v):
    return buu*deltaU_2(u, v) + bvv*deltaV_2(u, v) + buv*deltaU_1(u, v)*deltaV_1(u, v) + bu*deltaU_1(u, v) + bv*deltaV_1(u, v) + b

"""
def taylor(u, v):
    return buu*pow(deltaU_1(u, v), 2) + bvv*pow(deltaV_1(u, v), 2) + buv*deltaU_1(u, v)*deltaV_1(u, v) + bu*deltaU_1(u, v) + bv*deltaV_1(u, v) + b
"""


if __name__ == "__main__":
    # Find the error of origin
    u = 0
    v = 0
    u += deltaU_1(u, v)
    v += deltaV_1(u, v)
    print "Origin value: ", E(u, v)

    # Find the error of taylor's expansion
    buu, bvv, buv, bu, bv, b = 1.5, 4, -0.5, -1, -2, 0
    print "Selection 1 value: ", taylor(0, 0)

    buu, bvv, buv, bu, bv, b = 1.5, 4, -1, -2, 0, 3
    print "Selection 2 value: ", taylor(0, 0)

    buu, bvv, buv, bu, bv, b = 3, 8, -0.5, -1, -2, 0
    print "Selection 3 value: ", taylor(0, 0)

    buu, bvv, buv, bu, bv, b = 3, 8, -1, -2, 0, 3
    print "Selection 4 value: ", taylor(0, 0)