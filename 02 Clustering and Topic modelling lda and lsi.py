# this workbook contains
#  ---------------------
#  NER for most common personalities in the corpus
#
#  Clustering Techniques:
#   - KMeans Clustering
#
#  Topic Modelling
#  - LDA
#  - LSI
# ----------------------
#

from imports import *

import spacy
nlp = spacy.load('en_core_web_sm')
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')
get_dir = os.getcwd()
stopWords = set(stopwords.words('english'))

#  get files details
file_name = os.path.join(get_dir,"data/abcnews-date-text.csv")
df = pd.read_csv(file_name, parse_dates=[0], infer_datetime_format=True)
df.head(10)
print(df.shape)
df = df.sample(100000).reset_index(drop=True)

# feature engineering:
df["day"] = df["publish_date"].dt.day
df["weekday"] = df["publish_date"].dt.weekday_name
df["week"] = df["publish_date"].dt.week
df["month"] = df["publish_date"].dt.month
df["year"] = df["publish_date"].dt.year
print (df.info())
print (df.describe())

# data transformation
df["headline_lower"] = df["headline_text"].str.lower()
df["headline_tokens"] = df["headline_lower"].str.split()
titles = df["headline_tokens"].values


# pre-process the titles
def tokenize_only(text):
    tokens = [str(t.lemma_).lower() for t in nlp(text) if str(t.lemma_).lower() not in stopWords and str(t.lemma_).isalnum()]
    unigram = [i.lemma_ for i in nlp(' '.join(tokens)) if i.pos_ in ['PROPN','NOUN','ADJ']]
    return ' '.join(unigram)


def tfidfVectorizer(faqs):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=25000, ngram_range=(1, 1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(faqs)  # fit the vectorizer to the questions
    terms = tfidf_vectorizer.get_feature_names()  # get the vocabulary terms
    return tfidf_matrix, terms

def kcluster(tfidf_matrix, cluster_num, text):
    # kmeans clustering
    km = KMeans(algorithm='full', n_clusters=cluster_num, precompute_distances='auto')
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    ### MAP clusters to questions
    clustered_data = {'clusters': clusters, 'text': text}
    frame = pd.DataFrame(clustered_data)
    grouped = frame['clusters'].value_counts()
    nclusters = grouped[(grouped > 80)].keys()

    new_clusters = collections.defaultdict(list)
    for i, label in enumerate(km.labels_):
        new_clusters[label].append(i)

    for i in range(max(nclusters)):
        print()
        print("cluster of words %d:" % i)
        for ind in order_centroids[i, :5]:
            print('%s' % terms[ind])


# Pre-process and tokenize and clean the headlines:
formatted_lines = df["headline_text"].tolist()
formatted_lines = [tokenize_only(q) for q in formatted_lines]
formatted_lines = [re.sub("[^a-zA-Z]+", " ", s) for s in formatted_lines]

# NER - identifying most common personalities and their temporal relationship
e_docs = news_text['headline_text'].values
ner_text = []
for doc in e_docs:
    en_doc = nlp(u'' + doc)
    ner_docs = [(ent.text, ent.label, ent.label_) for ent in en_doc.ents]
    ner_text.append(ner_docs)
ner_text = [text for text in ner_text if len(text) > 1]
ner_text = sorted(ner_text)

# write ner text - file
# with open('entity_text.txt', 'w+', encoding = "utf-8", errors="ignore") as fout:
#         fout.write('\n'.join('{} {}'.format(x[0],x[1]) for x in ner_text))

# most common personalities in the corpus
most_entity = ['hackett','turnbull','ferguson','mackay','ferrer',
               'mugabe','mitcham','beattie','beckham']
for most_ent in most_entity:
    count = len(news_text[news_text['headline_text'].str.contains(most_ent)])
    print(most_ent,count)


# generate tf-df and dtm for cluster analysis
tfidf_matrix, terms = tfidfVectorizer(formatted_lines)
totalvocab_lemma_ = [tokenize_only(line) for line in formatted_lines]
num_clusters = 10
# k-means cluster
kcluster(tfidf_matrix, num_clusters, totalvocab_lemma_)

# ### Summary: - K-means Clustering -----------------------------------------------------------------------------###
# As we know that the k-means is optimizing a non-convex objective function, it will likely end up in a local
# optimum.  I have tried several runs with independent random init (different random clusters) for a good
# convergence. However, as the data is large, the process is computationally expensive to summarise the
# documents
# ------------------------------------------------------------------------------------------------------####


frequency = defaultdict(int)
for title in formatted_lines:
    title = title.split()
    for token in title:
        frequency[token] += 1

word_freq = {x: y for x, y in sorted(frequency.items(), key=operator.itemgetter(1), reverse=False) if y > 1000}


token_formatted_lines = [token.split() for token in formatted_lines]
titles_token_lem_filtered = [[token for token in title
                              if frequency[token] > 1 and len(token) > 1]
                             for title in token_formatted_lines]

np.save("titles", titles_token_lem_filtered)
titles_token_lem_filtered = np.load("titles.npy")

############
#creating a dictionary
#

dictionary = corpora.Dictionary(titles_token_lem_filtered)
dictionary.save('newsheadlines.dict')
print(dictionary)

dictionary = corpora.Dictionary.load('newsheadlines.dict')
titles_bow = [dictionary.doc2bow(title) for title in titles_token_lem_filtered]
corpora.MmCorpus.serialize('news-corpus.mm', titles_bow)
titles_bow = corpora.MmCorpus('news-corpus.mm')
tfidf = models.TfidfModel(titles_bow)
titles_tfidf = tfidf[titles_bow]
print(titles_bow)


n_train = 50000
n_test = 100000
titles_bow_train = ClippedCorpus(titles_bow, n_train)
titles_tfidf_train = ClippedCorpus(titles_tfidf, n_train)
titles_token_test = ClippedCorpus(titles_token_lem_filtered, n_test)
titles_bow_test = ClippedCorpus(titles_bow, n_test)
titles_tfidf_test = ClippedCorpus(titles_tfidf, n_test)

def print_topics(model, n_topics, n_words):
    topics = model.show_topics(num_topics=n_topics, num_words=n_words, log=False, formatted=False)
    for topic in topics:
        print("Topic %d: " %(topic[0]), *[word[0] for word in topic[1][:n_words]])

def get_top_topics(topic_probs, prob_limit=0.1):
    return list(filter(lambda item: item[1] > prob_limit, topic_probs))


def get_first_topic(topic_probs):
    return max(topic_probs, key=lambda item: item[1])[0]


def get_topic_words(model, title, n_words):
    topics = model.show_topics(num_topics=-1, num_words=n_words, log=False, formatted=False)
    if len(model[title]) > 0:
        topic = get_first_topic(model[title])
        return " ".join(["Title %d:" %(topics[topic][0])] + [word[0] for word in topics[topic][1][:n_words]])
    else:
        return "no results"

n_topics = 100

def get_lsi_models(n_topics,titles_tfidf_train,dictionary):
    lsi = models.LsiModel(titles_tfidf_train, id2word=dictionary, num_topics=n_topics)
    lsi.save("lsi-topics")
    lsi_model = models.LsiModel.load("lsi-topics")
    return lsi_model


def get_lda_model(titles_bow_train,dictionary,n_topics):
    lda = models.LdaModel(titles_bow_train, id2word=dictionary, num_topics=n_topics)
    print_topics(lda, n_topics=5, n_words=4)
    lda.save("lda-topics")
    lda_model = models.LdaModel.load("lda-topics")
    return lda_model


lsi_model = get_lsi_models(n_topics,titles_tfidf_train,dictionary)
lda_model = get_lda_model(titles_bow_train,dictionary,n_topics)
for idx, title in enumerate(titles_tfidf[:5]):
    print("Title:", df.loc[idx, "headline_text"])
    print("Title transformed: ", titles_token_lem_filtered[idx])
    print("LDA topic =>", get_topic_words(lda_model, titles_bow[idx], n_words=4))
    print("LSI topic =>", get_topic_words(lsi_model, title, n_words=4))
    print()

# Note above that the LSI model predicts nonzero probabilities for each theme.
# To compare the distribution of the number of themes by title,
# I decided to keep only themes with a probability of membership greater than 1/10.
# Below the treatment performed:
number_lsi_top_topics = [len(get_top_topics(title_lsi, prob_limit=0.1)) for title_lsi in lsi_model[titles_tfidf_test]]
number_lda_top_topics = [len(get_top_topics(title_lda, prob_limit=0.1)) for title_lda in lda_model[titles_bow_test]]
plt.close('all')
fig, axes = plt.subplots(figsize=(16,8))
plt.hist(number_lsi_top_topics, bins=100, label="LSI")
plt.hist(number_lda_top_topics, bins=100, label="LDA")
plt.xlabel("Number of most likely themes")
plt.ylabel("Frequency")
plt.legend(loc='best')
plt.title('Distribution of the number of most likely topics by title')
plt.show()

# to get the first topic
lsi_first_topic = [get_first_topic(title_lsi) for title_lsi in lsi_model[titles_tfidf_test] if len(title_lsi) > 0]
lda_first_topic = [get_first_topic(title_lda) for title_lda in lda_model[titles_bow_test] if len(title_lda) > 0]
plt.close('all')
fig, axes = plt.subplots(figsize=(16,8))
plt.hist(lsi_first_topic, bins=200, label="LSI")
plt.hist(lda_first_topic, bins=200, label="LDA")
plt.xlabel("Titles")
plt.ylabel("Frequency")
plt.legend(loc='best')
plt.title('Distribution of the first theme')
plt.show()

# The distribution of titles by theme seems more suitable for the LDA model than for the LSI model.
# Indeed, for the LSI model, many themes have no title and some have a lot more than others.


# ----------Topic Modelling--------------------------------#
# topic modelling to understand the document topics
# # summarise the topics:
print("running lda topic model")
print()
lda = LatentDirichletAllocation(n_components=10)
vect = CountVectorizer(ngram_range=(1, 2), stop_words='english')
dtm = vect.fit_transform(totalvocab_lemma_)
features = np.array(vect.get_feature_names())
lda.fit_transform(dtm)
# fit lda and sort the topic compnents
lda_dtf = lda.fit_transform(dtm)
sorting = np.argsort(lda.components_)[:, ::-1]

# get the topics
print("results - lda topics models")
print()
mglearn.tools.print_topics(topics=range(10), feature_names=features, sorting=sorting, topics_per_chunk=5, n_words=10)

# LDA - summarise & visualise the topics:

#pyLDAvis.enable_notebook() #--for ipython notebook
panel = pyLDAvis.sklearn.prepare(lda, dtm, vect, mds='tsne')
print("Please check the local server - http://127.0.0.1:8889/")
pyLDAvis.show(panel)

# In order to understand about the major topics and also to interpret the contextual performance
# with other clustering techniques, I have tried to identify the most common in the 10 topics using
# LDA. The results seems to be similiar with respect to k-means algorithm.
#
# We can see that there is substantial overlap between some topics, others are hard to interpret,
# and most of them have at least some terms that seem out of place. Assigning a name to a topic
# requires a human touch and an hour of your time, but the pyLDAvis tool is tremendously helpful.
# Once labelled, we can start analyzing the topics.
#
#  Relevant topics are retrieved using LDA compared as it is related to probabilistic models.However,
# while increasing the document size and dimension, we can not expect the same results with LDA
# Comparatively LDA seems to be better for this case to identify the topics and also flexible for text
# summarisation


