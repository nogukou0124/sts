#　ストップワードの削除
def stop_word(words):
    from nltk.corpus import stopwords
    new_words = [word for word in words if word not in stopwords.words('english')]
    return new_words
    
# 単語の分割
# 形態素解析
def nltk(text):
    from nltk.tokenize import TreebankWordTokenizer
    tree_tok = TreebankWordTokenizer()
    
    from nltk import stem
    stemmer = stem.PorterStemmer()
    
    docs = []
    for sp in tree_tok.span_tokenize(text):
        s = stemmer.stem(text[sp[0]:sp[1]])
        # s = text[sp[0]:sp[1]]
        docs.append(s)
    return docs

def pre_processing(docs):
    # 前処理
    doc_list = [] # 文書のリスト
    for doc in docs:
        lower = nltk(doc.lower())
        doc_list.append(stop_word(lower))
    
    return doc_list