

import pandas as pd
import nltk
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import cv2


lemm =nltk.stem.WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
class LemmaTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaTfidfVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

    
    
## Method for Extraction of Words and Topics ##

def extract_words_topics_1(H, W, feature_names, documents, no_top_words, no_top_documents):
    topic_1_top_40_words = []
    for topic_idx, topic in enumerate(H):
        count = 0
        if topic_idx==0:
            topic_1_top_40_words = ([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
            top_doc_indices_topic_1 = np.argsort( W[:,topic_idx] )[::-1][0:]  
    return top_doc_indices_topic_1, topic_1_top_40_words


def nmf_operation(data_set, total_topic):

    with open(data_set) as f_in:
        lines = [line.rstrip() for line in f_in] 
        lines = [line for line in lines if line]
    
    text_1 = len(lines)
    a = []
    if text_1 <= 3:
        for line in lines:
            a=line.split(".")
        text_data = a
    else:
        text_data = lines

    tfidf_vectorizer = LemmaTfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',decode_error='ignore')
    
    tfidf = tfidf_vectorizer.fit_transform(text_data)


    nmf = NMF(n_components=1,  max_iter=100, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')

    nmf.fit(tfidf)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    nmf_H = nmf.components_
    nmf_W = nmf.transform(tfidf)

    list_nmf = []
    
    list_nmf = extract_words_topics_1(nmf_H, nmf_W, tfidf_feature_names, text_data, 40, 3)
    n = int(len(list_nmf))

    tit = time.time()
    count = 55
    nmf_topic_top_words = WordCloud().generate(" ".join(list_nmf[1]))
    # nmf_topic_top_words = WordCloud(max_words=100,
    #                           stopwords=STOPWORDS,
    #                           background_color='black',
    #                           width=2500,
    #                           height=1800
    #                          ).generate(" ".join(list_nmf[1]))
    # print("====================ttt==========", time.time() - tit)
    plt.imshow(nmf_topic_top_words)
    plt.axis('off')
    plt.title('Wordcloud of key using NMF')
    filename = "/home/ashok/Desktop/NLP/1_demo/src/imgs/nmf_topic_top_words__{}.jpg".format(count)
    plt.savefig(filename)

    # plt.show()
    plt.close()

    
    return list_nmf, text_data


def text_summarisation_1(data_set, total_topic):
    nmf_topic , text_data = nmf_operation(data_set, total_topic)
    
    top_doc_indices_topic_1 = nmf_topic[0].tolist()
    l = len(top_doc_indices_topic_1)

    nmf_topic_1_sentences = []
    for doc_index in top_doc_indices_topic_1:
        nmf_topic_1_sentences.append(text_data[doc_index])
    
    ## Summary of Topic 1 (Using NMF) ##
    
    print("-"*40)
    print("Summary of Topic 1 (Using NMF)")
    print("-"*40)
    print()
    # str = []
    # for item in nmf_topic_1_sentences[:3]:
    #     str.append(item)
    
    # print('\n'.join('{}'.format(item) for item in nmf_topic_1_sentences[:3]))

    str = ""
    for item in nmf_topic_1_sentences[:3]:
        str += item
        str += ". "


    return str

        
import time

def main(data_set, total_topic):
    if total_topic == 1:
       str = text_summarisation_1(data_set, total_topic)
    elif total_topic == 2:
      str = text_summarisation_2(data_set, total_topic)
    return str

def main(data_set, total_topic):
    tic = time.time()
    if total_topic == 1:
       str = text_summarisation_1(data_set, total_topic)
    elif total_topic == 2:
      str = text_summarisation_2(data_set, total_topic)

    toc = time.time()
    print(toc-tic)

    return str
    
        
data_set = 'input.txt'


str = main(data_set, 1)

print("=============")
print(str)

