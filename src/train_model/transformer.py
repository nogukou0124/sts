from train_model import w2v ,sentence_bert,PositionalEncoding
import sys
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed
from matplotlib import pyplot as plt 

def train_and_test(train_x,train_y,test_x,test_y,trained_data):
    # モデルの生成及び準備
    print("----start model prepare----")
    wv = w2v.get_model()
    model_bert = sentence_bert.get_model()
    word_max = max_words(train_x,test_x)
    
    print("----finish model prepare----")
    
    
    # 学習データの準備
    print("----start preparing learn data----")
    train_x = prepare_x(train_x, word_max, wv)
    print(type(train_x))
    print(train_x.shape)
    
    train_y = prepare_y(train_x, train_y, trained_data, wv, model_bert)
    train_y = np.array(train_y) 
    print(type(train_y))
    print(train_y.shape)
    print("---- finish preparing learn data----")
    
    # モデルの構築
    print("----model start----")
    model = Sequential()
    model.add(TimeDistributed(Dense(train_y.shape[1])))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    print("----model finish----")
    
    # 訓練
    history = model.fit(train_x, train_y, epoch=10)
    
    # 損失関数の表示
    view_loss(history)
    
    # テストデータの生成
    test_x = prepare_x(test_x, word_max, wv)
    test_x = np.array(test_x)
    
    test_y = prepare_y(test_x, test_y, trained_data, wv, model_bert)
    test_y = np.array(test_y) 
    
    # モデルの評価
    model.evaluate(test_x,test_y)
    
    # テストデータの予測
    predicts = model.predict(test_x)
    print(np.array(predicts).shape)
    cos_sims = []
    list = predicts.tolist()
    it = iter(list[0:len(list)])
    for p1,p2 in zip(it,it):
        cos_sim = cosine_similarity(np.array([p1]),np.array([p2]))
        cos_sims.append(cos_sim[0][0])
    return cos_sims

# 目的データの生成
def prepare_x(data_x, word_max, wv):
    pos_encoder = PositionalEncoding.PositionalEncoding(d_model=300, dropout=0.5)
    encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=2)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    data_x = embedding(data_x, word_max, wv)
    
    list_x = []
    for data in data_x:
        x = np.array(data)
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        # x = pos_encoder(x)
        # x = transformer_encoder(x)
        x = x.to(torch.float64)
        x = np.array(x)
        list_x.append(x)
    
    return np.array(list_x).squeeze()

# 埋め込み処理
def embedding(docs,word_max,wv):
    data_x = []
    for doc in docs:
        words = []
        zero_vec = word_max - len(doc)
        for i in range(0,zero_vec):
            words.append(np.zeros(300))
        for word in doc:
            words.append(cre_w2v(word,wv))    
        data_x.append(words)
    return data_x

# word2vecのベクトル生成
def cre_w2v(word,wv):
    if word in wv:
       return wv[word]
    else:
       return np.zeros(300)

# 最大単語数の計算
def max_words(docs1,docs2):
    word_max = 0
    for doc in docs1:
        if word_max < len(doc):
            word_max = len(doc) 
    for doc in docs2:
        if word_max < len(doc):
            word_max = len(doc)      
    return word_max

# 正解データの準備
def prepare_y(data_x, data_y, trained_data, wv, model_bert):
    if trained_data == "word2vec":
        return cre_vec_by_w2v(data_x,wv)
        
    elif trained_data == "sentence-bert":
        return cre_vec_by_bert(data_y,model_bert)
    
    else:
        print('please correct trained data', file=sys.stderr)
        sys.exit(1)

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

