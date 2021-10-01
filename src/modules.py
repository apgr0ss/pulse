import pandas as pd
import numpy as np

def mod_topic_by_domain(topic):
    """
    Return data table of entities sorted by most relevant
    for provided topic (highest proportion of documents
    labeled as topic)
    """
    topic = int(topic)
    df = pd.read_csv('data\\sentence_by_topic.csv')
    # Aggregate to document level
    topic_doc_list = (df.groupby(['domain_name', 'url'])
                        .apply(lambda x: (x.name[0], x.topic.values[0]))
                        .values)
    topic_doc_df = pd.DataFrame.from_records(topic_doc_list, columns=['domain_name', 'topic'])
    topic_doc_table = (topic_doc_df.groupby('domain_name')
                                   .apply(lambda x: np.round(100*sum(x.topic==topic)/len(x), 1))
                                   .sort_values(ascending=False)
                                   .reset_index())
    topic_doc_table.columns = ['domain_name', '% pages']
    return topic_doc_table
