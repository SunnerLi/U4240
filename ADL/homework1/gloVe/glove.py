import tf_glove as tg

"""
    This program would train a glove model with the spesific corpus
    If you didn't have a powerful RAM, you should set the split parameter as True
    At first running time, you should un-comment the below two function.
    The two task would spend a period of time to deal with the co-occurence matrix
    (for 2,000,000 character, it may cost 3 minutes)

    At the second time, you don't need to conduct the comment function again.
    Unless you delete the temporary file

    Author: SunnerLi
    Finish: 23/10/2016
"""

model = tg.GloVeModel(embedding_size=80, context_size=100, max_vocab_size=100000)

# To statistic the word index in the corpus
# It's an expensive work, I suggest that you need to do it for once
model.fit_to_corpus([open('text10', 'r').read().split()], split=True)

# To record the frequency toward the co-occurence matrix
# It's an expensive work too, so I suggest that you need to do it for once
# This function should do after fit_to_corpus done
model.buils_matrix()

model.train(num_epochs=5, savePath="./model.pkl")