import csv
import numpy as np
import random
import cPickle
import time

"""
    This file is the PLA implementation.
    It simulate if the bank want to give the credit to the custom.
    We assume there're three factor effecting the decision.

    We get the credit result if the salary is more than 30,000.
    The program would get the result according to the above rule.
    Next, we train the perceptron module by the data stored in .csv file.

    At last, it give the user to test the training result.
"""

# Constant
N = 30
normN = 100
showTime = 2

# Macro
unit_step = lambda x: 0 if x < 0 else 1
genderTrans = lambda x: 1 if x == "female" else -1

# Characteristic
year = []
salary = []
gender = []
give = []
weight = []

def Load(csvName="test.csv"):
    """
        Load the .csv bank data

        Arg:    The .csv data file
    """
    global year, salary, gender

    f = open(csvName, 'r')
    for row in csv.DictReader(f):
        year.append( int(row['year']) )
        salary.append( int(row['salary']) )
        gender.append( row['gender'] )
    f.close()

    for i in range(len(year)):
        gender[i] = genderTrans(gender[i])
        salary[i] /= normN 

def GenerateByF():
    """
        Generate the give list by F
        If Her/His salary > 30000, give the cradit
    """
    global give

    for i in range(len(year)):
        if salary[i] > 30000/normN:
            give.append(1)
        else:
            give.append(0)

def Perceptron():
    """
        Perceptron algorithm
    """
    global weight
    global give

    n = 10000
    eta = 0.002
    weight = np.random.rand(3)
    
    for i in range(n):
        noErr = True
        for j in range(N):
            x = np.asarray([year[j], salary[j], gender[j]], dtype=object)
            result = np.dot(x, weight.T)
            error = give[j] - unit_step( result )
            if not error == 0:
                noErr = False
                weight += eta * error * x

        # Judge if reach the training ending
        if noErr:
            print "Epoch ", i, " break!"
            break
        else:
            print "Epoch ", i, "end, \tWeight: ", weight

def Testing(fileName="./weight.pkl"):
    """
        Give the year, salary and gender to predict the result

        Arg:    The name of weight file
    """
    # Load the weight file
    try:
        fh = open(fileName, 'rb')
        weight = cPickle.load(fh)
    except:
        print "The weight file no found, please train perceptron first!"
        return

    # Get the input
    _year = raw_input("Enter the year\n")
    _salary = raw_input("Enter the salary\n")
    _gender = raw_input("Enter the gender\n")
    _gender = genderTrans(_gender)

    # Cauculate the result
    x = np.asarray([int(_year), int(_salary)/normN, _gender], dtype=object)
    print x
    x = np.dot(x, weight.T)
    #x = unit_step(x)
    print x
    print "Predict..."
    if x > 0:
        print "Result: Allow!"
    else:
        print "Result: Not allow..."

def Training(fileName="./weight.pkl"):
    """
        The main function of training

        Arg:    The name of weight file
    """
    Load()
    GenerateByF()
    Perceptron()
    f = open(fileName, 'wb')
    cPickle.dump(weight, f)

def Main():
    """
        Our defined main function
    """
    while True:
        print "What do you want to do?"
        print "1. train"
        print "2. test"
        print "3. quit"
        c = raw_input()
        if c == "1":
            Training()
        elif c == "2":
            Testing()
        else:
            break

        # Clean the terminal
        time.sleep(showTime)
        print("\033c")

if __name__ == "__main__":
    Main()