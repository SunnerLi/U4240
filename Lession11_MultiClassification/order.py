import logging
logging.basicConfig(filename='ovo.log', level=logging.DEBUG, filemode='w')
orderLogger = logging.getLogger('orderGen')

pattern = []
patterns = []

def getOrder(n, k):
    global patterns
    __getOrder(n, k, 1)
    return __reduce(patterns)

def __getOrder(n, k, s):
    global patterns

    if len(pattern) < k:
        for i in xrange(s, n+1, 1):
            pattern.append(i)
            __getOrder(n, k, i+1)
            pattern.pop()
    elif len(pattern) == k:
        orderLogger.debug(pattern)
        add(pattern)

def add(element):
    global patterns
    _ = []
    for i in range(len(element)):
        _.append(element[i])
    patterns.append(_)

def __reduce(combinations):
    for i in range(len(combinations)):
        for j in range(len(combinations[i])):
            combinations[i][j] -= 1
    return combinations 
