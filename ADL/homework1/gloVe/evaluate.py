import tf_glove as tg

"""
    This program can evaluate the gloVe model
    If you didn't delete the .pkl file, you can load the model and test it directly

    Author: SunnerLi
    Finish: 23/10/2016
"""

model = tg.GloVeModel(embedding_size=80, context_size=10, max_vocab_size=100000)
model.load("model.pkl")
print model.cosSim("is", "were")
print model.cosSim("has", "were")