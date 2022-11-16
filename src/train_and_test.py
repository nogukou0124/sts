from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from train_model import w2v,lstm,transformer,transformer2,rnn,attention_encoder
   
# 訓練とテスト
# def train_and_test(learns,tests):
    # return transformer.train_and_test(tests)
    # return w2v.train_and_test(tests)
    # return transformer2.train_and_test(tests)

def train_and_test(learns,y_train,tests,y_test):
    # return attention_encoder.train_and_test(learns,y_train,tests,y_test)
    return lstm.train_and_test(learns,y_train,tests,y_test)
    # return rnn.train_and_test(learns,y_train,tests,y_test)
    
    
    