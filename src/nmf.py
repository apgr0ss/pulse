# misc
import pandas as pd
import numpy as np
from cleantext import clean
import pickle

# nltk
import nltk
from nltk.corpus import stopwords

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

# Import stop words
STOP_WORDS = stopwords.words('english')

def clean_text(docs):
    clean_docs = []
    for doc in docs:
        doc = str(doc)
        clean_doc = clean(doc,
                          fix_unicode=True,             # fix various unicode errors
                          to_ascii=True,                # transliterate to closest ASCII representation
                          lower=True,                   # lowercase text
                          no_line_breaks=True,          # fully strip line breaks as opposed to only normalizing them
                          no_urls=True,                  # replace all URLs with a special token
                          no_emails=True,                # replace all email addresses with a special token
                          no_phone_numbers=True,         # replace all phone numbers with a special token
                          no_numbers=True,               # replace all numbers with a special token
                          no_digits=False,                # replace all digits with a special token
                          no_currency_symbols=True,      # replace all currency symbols with a special token
                          no_punct=True,                 # remove punctuations
                          replace_with_punct="",          # instead of removing punctuations you may replace them
                          replace_with_url="",
                          replace_with_email="",
                          replace_with_phone_number="",
                          replace_with_number="",
                          replace_with_digit="",
                          replace_with_currency_symbol="",
                          lang="en")

        clean_doc = ' '.join([word for word in clean_doc.split() if word not in STOP_WORDS])
        clean_docs += [clean_doc]
    return clean_docs

def vectorizeCorpus(docs, vectorizer_class):
    corpus = clean_text(docs)
    vectorizer = vectorizer_class(ngram_range=(1,4),max_df=0.95,min_df=0.05)
    X = vectorizer.fit_transform(corpus)
    id2word = {idnum: word for word, idnum in vectorizer.vocabulary_.items()}
    with open('data/corpus2tfidf.pickle','wb') as f:
        pickle.dump(X, f)
    with open('data/id2word.pickle','wb') as f:
        pickle.dump(id2word, f)
    return X, id2word

def get_topic_desc(topic_num, W, H, id2word, n_words=10):
    topic_order = np.argsort(-1*H[topic_num,:]) # -1 to reverse order
    topic_words = [id2word[idnum] for idnum in topic_order[:n_words]]
    return topic_words

def run_NMF(X, n_topics):
    model = NMF(n_components=n_topics, max_iter=2000)
    W = model.fit_transform(X)
    H = model.components_
    return W, H

def gen_doc2topic(docs, W, id2word, topic_list):
    docs['topic'] = np.argsort(-1*W, axis=1)[:,0]
    df = {'domain_name':[], 'sentence':[], 'url':[], 'topic':[]}
    for doc in docs.iterrows():
        for sentence in doc[1].text.split('.'):
            topic = topic_list[doc[1].topic][1]
            if any([term in sentence for term in topic]):
                df['domain_name'] += [doc[1].domain_name]
                df['sentence'] += [sentence]
                df['url'] += [doc[1].url]
                df['topic'] += [doc[1].topic]
    return pd.DataFrame(df)

def nmf_main(path, first_call = True, n_topics = 5):
    docs = pd.read_csv(path)
    docs.text = docs.text.fillna('')
    if first_call:
        X, id2word = vectorizeCorpus(docs.text.values, TfidfVectorizer)
    else:
        with open('data/corpus2tfidf.pickle','rb') as f:
            X = pickle.load(f)
        with open('data/id2word.pickle','rb') as f:
            id2word = pickle.load(f)
    W, H = run_NMF(X, n_topics)
    topic_list = [(i, get_topic_desc(i, W, H, id2word)) for i in range(n_topics)]
    doc2topic = gen_doc2topic(docs, W, id2word, topic_list)
    doc2topic.to_csv('data/sentence_by_topic.csv', index=False)
    return topic_list
