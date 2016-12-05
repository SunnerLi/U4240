import math

def selection1():
    return pow((pow(3, 0.5) + 4), 0.5)

def selection2():
    return pow((pow(3, 0.5) - 1), 0.5)

def selection3():
    return pow((pow(96, 0.5) + 9), 0.5)

def selection4():
    return pow((-1 * pow(6, 0.5) + 9), 0.5)

def h1_leave_one_out(x):
    """
    up = 4 * x * (pow(x, 2) + 1)
    down = ( pow(x, 2) - 1 ) ** 2
    return up/down
    """
    first = ( 2 / (x + 1) ) ** 2
    down = ( -2 / (x + 1) ) ** 2
    return first + down

if __name__ == "__main__":
    print "selection 1: ", h1_leave_one_out( selection1() )
    print "selection 2: ", h1_leave_one_out( selection2() )
    print "selection 3: ", h1_leave_one_out( selection3() )
    print "selection 4: ", h1_leave_one_out( selection4() )