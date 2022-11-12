from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np

def learn_w2v(learns):
    model = word2vec.Word2Vec(learns,vector_size=300,min_count=1,sg=0)
    model.save("word2vec.model")

def get_model():
    # return word2vec.Word2Vec.load("word2vec.model")
    return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin" ,binary=True)

def get_w2v(doc,model):
    sum_vec = np.zeros(300)
    word_count = 0
    for word in doc:
        if word in model:  
            sum_vec += model[word]
            word_count += 1
            
    if word_count == 0:
        return sum_vec
    else:
        return sum_vec / word_count