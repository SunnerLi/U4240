import tf_glove as tg

"""
    This program would dump the word embedded into the format of the homework1
    You should train the model first!

    Author: SunnerLi
    Finish: 23/10/2016
"""

def dump():
    model = tg.GloVeModel(embedding_size=80, context_size=10, max_vocab_size=200000)
    model.load("model.pkl")

    vocabs = open("fullVocab.txt", 'r').readlines()
    print vocabs


    with open("wordVec.txt", 'w') as f:
        _index = 0
        while True:
            try:
                _word = (vocabs[_index])[:-1]
                _vector = model.embedding_for(_word)
                _string = _word
                for i in range(len(_vector)):
                    _string = _string + ' ' + str(_vector[i])
                _string += '\n'
                f.write(_string)
            except:
                print "word( ", _word, " ) didn't in the matrix..."
                #print "Index: ", _index
            
            _index += 1
            if _index == len(vocabs):
                break

if __name__ == "__main__":
    dump()