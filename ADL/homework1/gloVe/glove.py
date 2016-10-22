from evaluate import _eval
import tf_glove as tg

model = tg.GloVeModel(embedding_size=80, context_size=100)
model.fit_to_corpus([open('text10', 'r').read().split()])
model.train(num_epochs=400, savePath="./model.pkl")
_eval()