"""
    This program can compute the upper bound of the bounding function.
    It means any shatter wouldn't happen in any subset of the result

    In professor CharlieL's tutorial, a recursive formula can be declared.
    However, this program just use brute-force algorithm to solve the problem.
    So the complexity is O( (2 ^ N) * (K ^ N) ^ (N choose K) ) 
"""
from Stack import *
import numpy as np
import math
import sets

# The list to store the whole pattern
wholePattern = []

# The stack to generate the combination about the value of the points
stack = Stack() 

# The list to store the combination about the value of the points
chooseList = []

# Variable in the table of bounding function
N = 6
K = 5
    
def combination(i=1):
    """
        Generate the whole combination for the specific N and K
        For example, if N=4, K=2, the result is shown below:
        [
            [1, 2], [1, 3], [1, 4],
            [2, 3], [2, 4],
            [3, 4]
        ]

        Arg:    The base of the data that would be push,
                The range that the value would reach,
                The length of the each pair
        Ret:    None
    """
    global stack, chooseList
    j = stack.len()
    if stack.len() < K:
        while i+j-stack.len() <= N:
            stack.push(i+j-stack.len())       
            combination(i+j-stack.len()+2)    # remember to enhence the base double
            j += 1
            stack.pop()
    else:
        chooseList.append(stack.show())

def generate(k):
    """
        Generate the whole binary pattern for lengh k

        Arg:    The length of the pattern
    """
    for i in range(int(math.pow(2, k))):
        pat = []
        count = i
        for j in range(k):
            pat.append(count%2)
            count = count >> 1
        wholePattern.append(list(reversed(pat)))

def isValid(table):
    """
        Check if the shatter is occur

        Arg:    The testing table
        Ret:    If the shatter is occur
    """
    #print chooseList
    maxLength = -1
    
    # For each way to choose the value, e.g: [1, 2]
    for i in range(len(chooseList)):

        # Initialize the set which would store the sub-vectors
        checkSet = set()

        # For each row of the testing table, e.g: [0, 1, 0]
        for j in range( len(table) ):

            # Initialize the list to store the new single sub-vector
            newPat = []

            # Get each element of the sub-vector and store
            for k in range(len(chooseList[i])):
                newPat.append(table[j][chooseList[i][k]-1])

            # Add to the set
            checkSet.add(tuple(newPat))

        #print checkSet
        maxLength = max(maxLength, len(checkSet))

    # Return the result
    if maxLength >= len(wholePattern):
        return False
    return True

def copy(_list):
    """
        Copy the new list

        Arg:    The origin list
        Ret:    The new list
    """
    res = []
    for i in _list:
        res.append(i)
    return res

if __name__ == "__main__":
    table = []
    generate(K)
    combination()

    # For each pattern
    for i in range(pow(2, N)):

        # Generate the new pattern
        _pat = []
        index = i
        _table = copy(table)
        for j in range(N):
            _pat.append(index%2)
            index = index >> 1
        _table.append(list(reversed(_pat)))
        
        # Validate the pattern
        if isValid(_table):
            table = _table
    
    # Print the final table
    print "N: ", N, "\tK: ", K
    print "Table:"
    print table
    print "Number of combinations: ", len(table)