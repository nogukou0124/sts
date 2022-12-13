import sys
from train_model import w2v, lstm, sentence_bert, transformer, rnn, attention_encoder
   
# 訓練とテスト
def train_and_test(train_x, train_y, test_x, test_y, train_model, trained_data):
    if train_model == "word2vec":
        return w2v.train_and_test(test_x)
    
    elif train_model == "rnn":
        return rnn.train_and_test(train_x, train_y, test_x, test_y, trained_data)
    
    elif train_model == "lstm":
        return lstm.train_and_test(train_x, train_y, test_x, test_y, trained_data)
    
    elif train_model == "attention":
        return attention_encoder.train_and_test(train_x, train_y, test_x, test_y)
    
    elif train_model == "transformer":
        return transformer.train_and_test(train_x, train_y, test_x, test_y, trained_data)
    
    elif train_model == "sentence-bert":
        return sentence_bert.train_and_test(test_y)
    
    else:
        print('please correct train model',  file=sys.stderr)
        sys.exit(1)
        
    
    
    