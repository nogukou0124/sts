from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from train_model import w2v,lstm,cnn,lstm2
    
# 訓練とテスト
def train_and_test(learns,gs,tests):
    # return w2v.train_and_test(tests)
    return lstm2.train_and_test(learns,tests)
    