from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# word2vecモデルの学習
def learn_w2v(learns):
    model = word2vec.Word2Vec(learns,vector_size=300,min_count=1,sg=0)
    model.save("word2vec.model")

# モデルの呼び出し
def get_model():
    # return word2vec.Word2Vec.load("word2vec.model")
    return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin" ,binary=True)

# word2vecのベクトル生成
def cre_w2v(word,wv):
    if word in wv:
       return wv[word]
    else:
       return np.zeros(300)

# word2vecのベクトル生成
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

# 訓練とテスト
def train_and_test(tests):
    model = get_model()
    cos_sims = []
    it = iter(tests[0:len(tests)])
    for doc1,doc2 in zip(it,it):
        words1 = get_w2v(doc1,model)
        words2 = get_w2v(doc2,model)
        cos_sim = cosine_similarity(np.array([words1]),np.array([words2]))
        cos_sims.append(cos_sim[0][0])
    print(cosine_similarity(np.array([model["couch"]]),np.array([model["cow"]])))
    return cos_sims