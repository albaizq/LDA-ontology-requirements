import pandas as pd
import nltk
# model building package
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# package to clean text
import re

stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'


# topics
def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % topic_idx] = ['{}'.format(feature_names[i])
                                                    for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % topic_idx] = ['{:.1f}'.format(topic[i])
                                                      for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# terms association
def obtain_terms(req):
    req = req.lower()
    req = re.sub('[' + punctuation + ']+', ' ', req)
    req = re.sub(r'\s+', ' ', req)
    req = re.sub(r'([0-9]+)', '', req)
    stopwords.append("yes")

    req_token_list = [word for word in req.split(' ')
                      if word not in stopwords]

    req_token_list_unique = pd.unique(req_token_list).tolist()
    tags = nltk.pos_tag(req_token_list_unique)
    nouns = []

    for elem in tags:
        if elem[1] == 'NN' and elem[0] != '':
            nouns.append(elem[0].capitalize())
    final = ', '.join(nouns)
    return final


# cleaning
def clean_req(req):
    req = req.lower()
    req = re.sub('[' + punctuation + ']+', ' ', req)
    req = re.sub(r'\s+', ' ', req)
    req = re.sub(r'([0-9]+)', '', req)
    req_token_list = [word for word in req.split(' ')
                      if word not in stopwords]

    req_token_list = [word_rooter(word) if '#' not in word else word
                      for word in req_token_list]

    req = ' '.join(req_token_list)
    return req


def get_topics(df):
    clean_reqs = df.Requirement.apply(clean_req)
    vectorizer = CountVectorizer(max_df=0.8, min_df=5, token_pattern=r'\w+|\$[\d\.]+|\S+')
    # apply transformation
    tf = vectorizer.fit_transform(clean_reqs).toarray()
    # tf_feature_names tells us what word each column in the matric represents
    tf_feature_names = vectorizer.get_feature_names()
    number_of_topics = 10
    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model.fit(tf)
    no_top_words = 5
    print("3) Extracted topics with LDA")
    pd.set_option('display.max_columns', None)
    print(display_topics(model, tf_feature_names, no_top_words))
    print("4) Terms identification")
    new_column = df.Requirement.apply(obtain_terms)
    df.insert(loc=3,
              column='Terms',
              value=new_column)
    print(df)
    file_name = r"\dataset\ReqsWithTerms.csv"
    df.to_csv(file_name, encoding='utf-8', sep=';', index=False)
