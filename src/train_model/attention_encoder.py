from train_model import w2v
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,TimeDistributed,SimpleRNN
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt 
from train_model import transformer
import torch

# 訓練とテスト
def train_and_test(learns,y_train,tests,y_test):
    wv = w2v.get_model()
    model_bert = transformer.get_model()
    max = max_words(learns,tests)
    x_train = cre_vec_x(learns,wv,max)
    x_train = np.array(x_train)
    # y_train = cre_vec_y(learns,wv)
    y_train = cre_vec_y2(y_train,model_bert)
    y_train = np.array(y_train)
    
    print(x_train.shape)
    print(y_train.shape)

    model = Sequential()
    # model.add(LSTM(100,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(LSTM(100,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(LSTM(100,return_sequences=False,input_shape=(x_train.shape[1],x_train.shape[2])))
    # model.add(SimpleRNN(100,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    # model.add(SimpleRNN(100,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    # model.add(SimpleRNN(100,return_sequences=False,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Dense(y_train.shape[1]))
    # model.add(TimeDistributed(Dense(y_train.shape[1])))
    model.compile(loss="mean_squared_error", optimizer="adam")
    
    print(model.summary)
    
    print("start train")
    history = model.fit(x_train,y_train,epochs=100)
    print("finish train")
    
    x_test = cre_vec_x(tests,wv,max)
    x_test = np.array(x_test)
    
    # y_test = cre_vec_y(tests,wv)
    y_test = cre_vec_y2(y_test,model_bert)
    y_test = np.array(y_test)
    print(model.evaluate(x_test,y_test))
    
    view_loss(history)
    predicts = model.predict(x_test)
    print(np.array(predicts).shape)
    cos_sims = []
    list = predicts.tolist()
    it = iter(list[0:len(list)])
    for p1,p2 in zip(it,it):
        cos_sim = cosine_similarity(np.array([p1]),np.array([p2]))
        cos_sims.append(cos_sim[0][0])
    return cos_sims

def max_words(learns,tests):
    max = 0
    for learn in learns:
        if max < len(learn):
            max = len(learn)
    for test in tests:
        if max < len(test):
            max = len(test)      
    return max

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

def cre_vec_y(learns,wv):
    y_train = []
    for learn in learns:
        y = w2v.get_w2v(learn,wv)
        y_train.append(y)
    return y_train

def cre_vec_y2(learns,model):
    y_train = []
    for learn in learns:
        y = model.encode(learn)
        y_train.append(y)
    return y_train
    

# word2vecのベクトル生成
def cre_w2v(word,wv):
    if word in wv:
       return wv[word]
    else:
       return np.zeros(300)

# 損失関数のグラフ表示
def view_loss(history):
    fig = plt.figure()
    plt.plot(history.history["loss"])
    fig.savefig("./images/loss.jpg")