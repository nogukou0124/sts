from train_model import w2v
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def train(learns):
    w2v.learn_w2v(learns)

def test(tests):
    model = w2v.get_model()
    cos_sims = []
    it = iter(tests[0:len(tests)])
    for doc1,doc2 in zip(it,it):
        words1 = w2v.get_w2v(doc1,model)
        words2 = w2v.get_w2v(doc2,model)
        cos_sim = cosine_similarity(np.array([words1]),np.array([words2]))
        cos_sims.append(cos_sim[0][0])
    return cos_sims
    