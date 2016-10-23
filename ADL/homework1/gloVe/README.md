# Word Embedded (GloVe)

##  Abstract    
This is the revision edition about the gloVe model, which is clone from [here](https://github.com/GradySimon/tensorflow-glove). The origin repository didn't have save function. Moreover, my computer cannot train the whole corpus instantly. As the result, this program provide the additional functions.    
<br/>
## Usage
After the statistic, you can reboot the computer. This indirect change can release the memory, and reduce the loading of the computer.     
<br/>
Statistic:

```python
model = tg.GloVeModel(embedding_size=80, context_size=100, max_vocab_size=100000)
model.fit_to_corpus([open('text10', 'r').read().split()], split=True)
model.buils_matrix()
```
<br/>
Train:

```python
model = tg.GloVeModel(embedding_size=80, context_size=100, max_vocab_size=100000)
model.train(num_epochs=5, savePath="./model.pkl")
```
<br/>
Evaluation:

```python
model = tg.GloVeModel(embedding_size=80, context_size=100, max_vocab_size=100000)
model.load("model.pkl")
print model.cosSim("is", "were")
```

<br/>
## Dataset
```terminal
wget http://mattmahoney.net/dc/text8.zip -O text8.zip
unzip text8.zip
```

<br/>
## References
- Pennington, J., Socher, R., & Manning, C. D. (2014). [Glove: Global vectors for word representation](http://nlp.stanford.edu/pubs/glove.pdf). Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014), 12, 1532-1543.
