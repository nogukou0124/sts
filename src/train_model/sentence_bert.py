from sentence_transformers import SentenceTransformer, util 

def train_and_test(tests):
    cos_sims = []
    model = SentenceTransformer('stsb-xlm-r-multilingual') 
    it = iter(tests[0:len(tests)])
    for doc1,doc2 in zip(it,it):
        words1 = model.encode(doc1)
        words2 = model.encode(doc2)
        # print(words1)
        cos_sim = util.cos_sim(words1, words2)
        cos_sims.append(float(cos_sim[0][0]))
    
    return cos_sims

def get_model():
    return SentenceTransformer('stsb-xlm-r-multilingual') 

# model = get_model()
# print(model.encode("Black and white photo of couch with purse at one end.",convert_to_tensor=True).shape)