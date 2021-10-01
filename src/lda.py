import re
import numpy as np
import pandas as pd
from pprint import pprint
from cleantext import clean
import time
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases

# Stop words
import nltk
nltk.set_proxy('https://p1web4.frb.org:8080')
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.stem import PorterStemmer

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# Import stop words
STOP_WORDS = stopwords.words('english')

def preprocess_text(doc):
    doc = str(doc)
    doc_clean = clean(doc,
                        fix_unicode=True,               # fix various unicode errors
                        to_ascii=True,                  # transliterate to closest ASCII representation
                        lower=True,                     # lowercase text
                        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
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
                        lang="en"
                    )
    return doc_clean

def stem_word(word):
    ps = PorterStemmer()
    return ps.stem(word)

def preprocess_docs(docs):
    docs_clean = []
    for i, doc in enumerate(docs.text.values):
        if i%100 == 0:
            print('{0}/{1}'.format(i, len(docs)))
        docs_clean += [preprocess_text(doc)]
    return docs_clean

def custom_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    return 10000 if bigram_count > min_count else 0

def tokenize(docs):
    corpus = []
    for doc in docs:
        tokenized = gensim.utils.simple_preprocess(doc, deacc=True)
        tokenized = [word for word in tokenized if word not in STOP_WORDS]
        # tokenized = [stem_word(word) for word in tokenized]
        corpus += [tokenized]

    bigram  = Phrases(corpus, min_count=100, scoring=custom_scorer, threshold = 0, delimiter=' ')
    trigram = Phrases(bigram[corpus], min_count=100, scoring=custom_scorer, threshold = 0, delimiter=' ')
    quadgram = Phrases(bigram[trigram[corpus]], min_count=100, scoring=custom_scorer, threshold = 0, delimiter=' ')


    bigrams_  = np.unique([b for sent in corpus for b in bigram[sent] if b.count(' ') == 1])
    trigrams_ = np.unique([t for sent in corpus for t in trigram[bigram[sent]] if t.count(' ') == 2])
    quadgrams_ = np.unique([q for sent in corpus for q in quadgram[trigram[bigram[sent]]] if q.count(' ') == 3])


    final_corpus = []
    for doc in corpus:
        raw = ' '.join(doc)
        ngram_list = [quadgrams_, trigrams_, bigrams_]
        for ngram in ngram_list:
            for term in ngram:
                raw = raw.replace(term, '_'.join(term.split())) if term in raw else raw
        tokenized = gensim.utils.simple_preprocess(raw, deacc=False, max_len = 200)
        final_corpus += [tokenized]

    id2word = corpora.Dictionary(final_corpus)
    final_corpus_input = [id2word.doc2bow(text) for text in final_corpus]
    return final_corpus_input, id2word


def gen_doc2topic(lda_model, bows, original_docs):
    topic_keywords = lda_model.show_topics(num_topics=-1, formatted=False)
    topic_dict = {topic_num: [keyword[0] for keyword in keyword_list] for (topic_num, keyword_list) in topic_keywords}
    df = {'name':[], 'sentence':[], 'url':[], 'topic':[]}
    original_docs.text = original_docs.text.replace(np.nan, '', regex=True)
    for doc, bow in zip(original_docs.iterrows(), bows):
        predicted_topic = sorted(lda_model.get_document_topics(bow), key=lambda x: x[1])[-1][0]
        doc_by_sentence = doc[1].text.split('.')
        for sentence in doc_by_sentence:
            if any([keyword in sentence for keyword in ' '.join(topic_dict[predicted_topic]).replace('_',' ')]):
                df['name'] += [doc[1].name]
                df['sentence'] += [sentence]
                df['url'] += [doc[1].url]
                df['topic'] += [predicted_topic]
    return pd.DataFrame(df)

def lda(corpus, id2word, n_topics):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics,
                                                random_state=3,
                                                update_every=1,
                                                chunksize=100,
                                                passes=20,
                                                alpha='auto',
                                                per_word_topics=False)
    return lda_model

def lda_main(path, first_call = True, n_topics = 5):
    docs = pd.read_csv(path)
    if first_call:
        docs_clean = preprocess_docs(docs)
        corpus_tokenized, id2word = tokenize(docs_clean)

        with open('data\\corpus_tokenized.pickle', 'wb') as f:
            pickle.dump(corpus_tokenized, f)
        with open('data\\id2word.pickle', 'wb') as f:
            pickle.dump(id2word, f)
    else:
        with open('data\\corpus_tokenized.pickle', 'rb') as f:
            corpus_tokenized = pickle.load(f)
        with open('data\\id2word.pickle', 'rb') as f:
            id2word = pickle.load(f)

    lda_model = lda(corpus_tokenized, id2word, n_topics)

    # Produce csv with topic descriptions (top 10 words/topic)
    doc2topic = gen_doc2topic(lda_model, corpus_tokenized, docs)
    doc2topic.to_csv('data\\sentence_by_topic.csv', index=False)

    return lda_model.show_topics(num_topics=-1, formatted=False)
