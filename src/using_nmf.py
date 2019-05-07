

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



## Method for Extraction of Words and Topics ##

def extract_words_topics_2(H, W, feature_names, documents, no_top_words, no_top_documents):
    topic_1_top_40_words = []
    topic_2_top_40_words = []

    for topic_idx, topic in enumerate(H):
        count = 0
        if topic_idx==0:
            topic_1_top_40_words = ([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
            top_doc_indices_topic_1 = np.argsort( W[:,topic_idx] )[::-1][0:]  
        else:
            topic_2_top_40_words = ([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
            top_doc_indices_topic_2 = np.argsort( W[:,topic_idx] )[::-1][0:]

    return top_doc_indices_topic_1,top_doc_indices_topic_2, topic_1_top_40_words, topic_2_top_40_words




def nmf_operation(data_set, total_topic):

    with open(data_set) as f_in:
        lines = [line.rstrip() for line in f_in] 
        lines = [line for line in lines if line]
    
    text_1 = len(lines)
    
    readf = open(data_set)

    for line in readf:

      a=line.split(".")

    readf.close()
    
    text_2 = len(a)
    
    if text_1 <= 3:
        text_data = a
    else:
        text_data = lines

    tfidf_vectorizer = LemmaTfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',decode_error='ignore')
    
    tfidf = tfidf_vectorizer.fit_transform(text_data)


    if total_topic == 2:
        nmf = NMF(n_components=2,  max_iter=100, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
    elif total_topic == 1:
        nmf = NMF(n_components=1,  max_iter=100, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')

    nmf.fit(tfidf)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    nmf_H = nmf.components_
    nmf_W = nmf.transform(tfidf)

    list_nmf = []
    if total_topic == 2:
        list_nmf = extract_words_topics_2(nmf_H, nmf_W, tfidf_feature_names, text_data, 40, 3)
    elif total_topic == 1:
        list_nmf = extract_words_topics_1(nmf_H, nmf_W, tfidf_feature_names, text_data, 40, 3)
    n = int(len(list_nmf))

    count = 1
    for i in range(n//2,n):
        # print(list_nmf[i])

        nmf_topic_top_words = WordCloud(
                              stopwords=STOPWORDS,
                              background_color='black',
                              width=2500,
                              height=1800
                             ).generate(" ".join(list_nmf[i]))
        plt.imshow(nmf_topic_top_words)
        plt.axis('off')
        plt.title('Wordcloud of key using NMF')
        filename = "/home/ashok/Desktop/NLP/text_summarisation_using_pointer_generator/src/ptr_gnrtr/imgs/nmf_topic_top_words__{}.jpg".format(count)
        plt.savefig(filename)

        # plt.show()
        plt.close()
        count+=1

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
    print('\n'.join('{}'.format(item) for item in nmf_topic_1_sentences[:3]))
    


    
def text_summarisation_2(data_set, total_topic):
    
    nmf_topic, text_data = nmf_operation(data_set, total_topic)
    
    top_doc_indices_topic_1 = nmf_topic[0].tolist()
    top_doc_indices_topic_2 = nmf_topic[1].tolist()
    l = len(top_doc_indices_topic_1)

    count = 0

    for i in top_doc_indices_topic_1:
        top_doc_indices_topic_1_end_val = top_doc_indices_topic_1[l-1-count]

        if top_doc_indices_topic_2.index(i) >= top_doc_indices_topic_1.index(i):
            del top_doc_indices_topic_2[top_doc_indices_topic_2.index(i)]

        if top_doc_indices_topic_1.index(top_doc_indices_topic_1_end_val)>= top_doc_indices_topic_2.index(top_doc_indices_topic_1_end_val):
            del top_doc_indices_topic_1[top_doc_indices_topic_1.index(top_doc_indices_topic_1_end_val)]
        count += 1

    nmf_topic_1_sentences = []
    nmf_topic_2_sentences = []
    for doc_index in top_doc_indices_topic_1:
        nmf_topic_1_sentences.append(text_data[doc_index])
    for doc_index in top_doc_indices_topic_2:
        nmf_topic_2_sentences.append(text_data[doc_index])

    
    ## Summary of Topic 1 (Using LDA) ##

    print("-"*40)
    print("Summary of Topic 1 (Using NMF)")
    print("-"*40)
    print()
    print('\n'.join('{}'.format(item) for item in nmf_topic_1_sentences[:3]))
    
    
    # ## Summary of Topic 2 (Using NMF) ##

    print("-"*40)
    print("Summary of Topic 2 (Using NMF)")
    print("-"*40)
    print()
    print('\n'.join('{}'.format(item) for item in nmf_topic_2_sentences[:3]))

    
import time


def main(data_set, total_topic):
    tic = time.time()
    if total_topic == 1:
        text_summarisation_1(data_set, total_topic)
    elif total_topic == 2:
        text_summarisation_2(data_set, total_topic)
    toc = time.time()
    print(toc-tic)

        
path = 'input.txt'
# path = "/home/ashok/Desktop/NLP/text_summarisation_using_pointer_generator/src/ptr_gnrtr/test_data/news9.txt"


main(path, 1)

