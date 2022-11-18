from train_model import w2v 
import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

def train_and_test(tests):
    max = max_words(tests)
    wv = w2v.get_model()
    x_tests = cre_vec_x(tests,wv,max)
    x_tests = np.array(x_tests)
    print(x_tests.shape)
    x_tests = torch.from_numpy(x_tests)
    x_tests = x_tests.to(torch.float32)
    encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=5)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    # out = transformer_encoder(x_tests)
    # print(out)

def cre_vec_x(learns,wv,max):
    x_train = []
    for learn in learns:
        x_train.append(cre_array(wv,learn,max))
    return x_train

def cre_array(wv,doc,max):
    arrays = []
    zero_vec = max - len(doc)
    for i in range(0,zero_vec):
        arrays.append(np.zeros(300))
    for words in doc:
        arrays.append(cre_w2v(words,wv))
    return arrays

# word2vecのベクトル生成
def cre_w2v(word,wv):
    if word in wv:
       return wv[word]
    else:
       return np.zeros(300)

def max_words(tests):
    max = 0
    for test in tests:
        if max < len(test):
            max = len(test)      
    return max