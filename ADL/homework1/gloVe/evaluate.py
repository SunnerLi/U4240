import tf_glove as tg

def _eval():
    model = tg.GloVeModel(embedding_size=80, context_size=10)
    model.fit_to_corpus([open('text10', 'r').read().split()])
    model.load("model.pkl")
    print model.embedding_for("using")

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
