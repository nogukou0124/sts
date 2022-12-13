from train_model import w2v
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt 
from train_model import sentence_bert
import sys

# 訓練とテスト
def train_and_test(learns,y_train,tests,y_test,trained_data):
    wv = w2v.get_model() # word2vecモデルの生成
    model_bert = sentence_bert.get_model() # sentence-bertモデルの生成
    
    # 文書の最大単語数
    max = max_words(learns,tests)
    
    # 訓練データの生成
    x_train = cre_vec_x(learns,wv,max)
    x_train = np.array(x_train)
    
    if trained_data == "word2vec":
        y_train = cre_vec_by_w2v(tests,wv)
        
    elif trained_data == "sentence-bert":
        y_train = cre_vec_by_bert(y_test,model_bert)
    
    else:
        print('please correct trained data', file=sys.stderr)
        sys.exit(1)     
    
    y_train = np.array(y_train)
    
    print(x_train.shape)
    print(y_train.shape)

    # モデルの生成
    model = Sequential()
    model.add(SimpleRNN(100,return_sequences=False,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    
    print("start train")
    history = model.fit(x_train,y_train,epochs=100)
    print("finish train")
    
    # テストデータの生成
    x_test = cre_vec_x(tests,wv,max)
    x_test = np.array(x_test)
    
    if trained_data == "word2vec":
        y_test = cre_vec_by_w2v(learns,wv)
        
    elif trained_data == "sentence-bert":
        y_test = cre_vec_by_bert(y_train,model_bert)
    
    else:
        print('please correct trained data', file=sys.stderr)
        sys.exit(1)     
    
    # モデルの評価
    model.evaluate(x_test,y_test)
    
    # 損失関数の表示
    view_loss(history)
    
    # テストデータの予測
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

def cre_vec_x(data,wv,max):
    x_list = []
    for d in data:
        x_list.append(cre_array(wv,d,max))
    return x_list

def cre_array(wv,doc,max):
    arrays = []
    zero_vec = max - len(doc)
    for i in range(0,zero_vec):
        arrays.append(np.zeros(300))
    for words in doc:
        arrays.append(w2v.cre_w2v(words,wv))
    return arrays

def cre_vec_by_w2v(learns,wv):
    y_train = []
    for learn in learns:
        y = w2v.get_w2v(learn,wv)
        y_train.append(y)
    return y_train

def cre_vec_by_bert(learns,model):
    y_train = []
    for learn in learns:
        y = model.encode(learn)
        y_train.append(y)
    return y_train

# 損失関数のグラフ表示
def view_loss(history):
    fig = plt.figure()
    plt.plot(history.history["loss"])
    fig.savefig("./images/loss.jpg")