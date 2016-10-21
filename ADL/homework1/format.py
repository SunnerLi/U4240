import cPickle

with open("embedded.pkl", 'r') as f:
    vectors = cPickle.load( f )
with open("mapping.pkl", 'r') as f:
    mapping = cPickle.load( f )

# Reverse the mapping
reverseMap = dict()
for key in mapping:
    reverseMap[mapping[key]] = key

with open("wordVec.txt", 'w') as f:
    for i in range(100000):
        _string = reverseMap[i]
        print "write: ", i, '\t', _string
        for j in range(80):
            _string = _string + ' ' + str(vectors[i][j])
        _string += '\n'
        f.write(_string)
