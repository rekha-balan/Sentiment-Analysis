# This work book contains the Sentiment Analysis
# 1. Polarity
# 2. Subjectivity
# 3. Naive Bayes Approach to predict subjectivity and objectivity in the headlines
# 4. Correlation between subjectivity and polarity scores

from imports import *
from copy import deepcopy
import pickle

# get files details

get_dir = os.getcwd()
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

titles_token_lem_filtered = np.load("titles.npy")
dictionary = corpora.Dictionary.load('newsheadlines.dict')
titles_bow = [dictionary.doc2bow(title) for title in titles_token_lem_filtered]
corpora.MmCorpus.serialize('news-corpus.mm', titles_bow)
titles_bow = corpora.MmCorpus('news-corpus.mm')
tfidf = models.TfidfModel(titles_bow)
titles_tfidf = tfidf[titles_bow]
print(titles_bow)


n_train = 5000
n_test = 10000
titles_bow_train = ClippedCorpus(titles_bow, n_train)
titles_tfidf_train = ClippedCorpus(titles_tfidf, n_train)
titles_token_test = ClippedCorpus(titles_token_lem_filtered, n_test)
titles_bow_test = ClippedCorpus(titles_bow, n_test)
titles_tfidf_test = ClippedCorpus(titles_tfidf, n_test)

sid = SentimentIntensityAnalyzer()
df["polarity_score"] = [sid.polarity_scores(" ".join(title_token))["compound"]
                            for title_token in titles_token_test]

df.loc[df["polarity_score"] > 0, "polarity"] = "pos"
df.loc[df["polarity_score"] == 0, "polarity"] = "neu"
df.loc[df["polarity_score"] < 0, "polarity"] = "neg"
plt.close('all')
fig, axes = plt.subplots(figsize=(8,5))
df["polarity_score"].hist(bins=50)
plt.xlabel("Polarity")
plt.ylabel("Frequency")
pl_freq_name = 'Distribution of Polarity'
plt.title(pl_freq_name)
plt.show()


# polarity score
ps_by_week_and_year = df.groupby(["year", "week"])["polarity_score"].mean()
ps_by_month_and_year = df.groupby(["year", "month"])["polarity_score"].mean()
ps_by_year = df.groupby(["year"])["polarity_score"].mean()
std_ps_by_week_and_year = df.groupby(["year", "week"])["polarity_score"].std()
std_ps_by_month_and_year = df.groupby(["year", "month"])["polarity_score"].std()
std_ps_by_year = df.groupby(["year"])["polarity_score"].std()
ps_by_year = df.groupby(["year", "polarity"])["polarity"].count().unstack()

plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))
ps_by_year.plot(ax=ax)
plt.xlabel("Year")
plt.ylabel("Polarity")
plt.title('Average polarity score of titles per year')
plt.show()


plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))
ps_by_year.plot.bar(stacked=True, ax=ax)
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.title('Distribution of headlines polarity by year')
plt.show()


plt.close('all')
fig, ax = plt.subplots(figsize=(16, 5))
df.groupby(['year', 'polarity'])["polarity"].count().unstack().div(
    df.groupby(['year', 'polarity'])["polarity"].count().unstack().sum(axis=1),
    axis=0).plot.bar(ax=ax, stacked=True)
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.title('Share of the polarities of titles by year')
plt.show()


plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))
ps_by_month_and_year.plot()
plt.xlabel("(Year, Month)")
plt.ylabel("Polarity score")
plt.title('Polarity score of median headings by month and year')
plt.show()

plt.close('all')
fig, ax = plt.subplots(figsize=(20,5))
ps_by_week_and_year.plot()
plt.xlabel("(Year, Month)")
plt.ylabel("Polarity score")
pl_name = 'Polarity score of median headings by week and year'
plt.title(pl_name)
plt.show()

# The variation in the average polarity score per month can be significant as shown in the graph above.#
# There is a trend of several months on the rise over the period 2011-2015.
# This variability is even stronger when choosing a weekly periodicity.

# Hypothesis - It can be hypothesised that this polarity is strongly influenced by the nature of the topics.
# The subjectivity of a title is a metric that could be indicative of the journalistic
# quality of an article. So we could:
#   - Suggest to a reader more objective articles on a theme
#   - Follow the evolution of the objectivity of titles to correct an editorial line
#


# 3. Training of a Bayesian Naive classifier from a set of subjective and objective labelled sentences

n_instances = 5000 # Number of subjective / objective labeled documents in the complete dataset
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
train_subj_docs = subj_docs[:int(n_instances*0.8)]
test_subj_docs = subj_docs[int(n_instances*0.8):n_instances]
train_obj_docs = obj_docs[:int(n_instances*0.8)]
test_obj_docs = obj_docs[int(n_instances*0.8):n_instances]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

# We use simple unigram word features, handling negation:
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# We apply features to obtain a feature-value representation of our datasets:
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

# We can now train our classifier on the training set, and subsequently output the evaluation results:
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))



# Save to file in the current working directory
pkl_filename = "sentim_model_classifier.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(sentim_analyzer, file)


# Results
# -------
# Training classifier
# Evaluating NaiveBayesClassifier results...
# Accuracy: 0.91
# F-measure [obj]: 0.908256880733945
# F-measure [subj]: 0.9116781157998036
# Precision [obj]: 0.9261954261954262
# Precision [subj]: 0.894990366088632
# Recall [obj]: 0.891
# Recall [subj]: 0.929
# -----------

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# %%time -- heads up ---30 mins processing time ---
df["subjectivity"] = [sentim_analyzer.classify(title) for title in titles_token_test]

plt.close('all')
fig, axes = plt.subplots(figsize=(10, 5))
df['subjectivity'].value_counts().plot(kind='bar')
plt.xlabel("Level of subjectivity")
plt.ylabel("Frequency")
plt.title('Distribution of the subjectivity of the headlines')
plt.show()


plt.close('all')
fig, ax = plt.subplots(figsize=(15, 5))
df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().div(df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().sum(axis=1), axis=0).plot.bar(ax=ax, stacked=True)
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.title('Title subjectivity share by year')
plt.show()


# There is some stability on the part of objective and objective titles
# over time. However, there are fluctuations that are difficult to quantify with the
# observation of the graph.
# Any conclusion at this stage is still rushed, but one could measure
# the correlation between the subjectivity and objective titles


sub_prop_by_year = df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().\
    div(df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().sum(axis=1),axis=0)

polarity_proportion_by_year = df.groupby(['year', 'polarity'])["polarity"].count().unstack().\
    div(df.groupby(['year', 'polarity'])["polarity"].count().unstack().sum(axis=1),
    axis=0)

print(" Correlations (Annual frequency of 2003-2017) between the proportion of subjective titles and:")
print(" Negative: %.2f" %(sub_prop_by_year["subj"].corr(polarity_proportion_by_year["neg"], method='pearson')))
print("Neutral : %.2f" %(sub_prop_by_year["subj"].corr(polarity_proportion_by_year["neu"], method='pearson')))
print("Positve : %.2f" %(sub_prop_by_year["subj"].corr(polarity_proportion_by_year["pos"], method='pearson')))


sub_prop_by_month_year = df.groupby(['year', 'month', 'subjectivity'])["subjectivity"].count().unstack().div(
                                    df.groupby(['year', 'month', 'subjectivity'])["subjectivity"].count().unstack().sum(axis=1),
                                    axis=0)

polarity_proportion_by_month_year = df.groupby(['year', 'month', 'polarity'])["polarity"].count().unstack().div(
    df.groupby(['year', 'month', 'polarity'])["polarity"].count().unstack().sum(axis=1),
    axis=0)

print("Correlations (MONTHLY periodicity 2003-2017) between the proportion of subjective titles and:")
print("Negative: %.2f" %(sub_prop_by_month_year["subj"].corr(polarity_proportion_by_month_year["neg"], method='pearson')))
print("Neutral : %.2f" %(sub_prop_by_month_year["subj"].corr(polarity_proportion_by_month_year["neu"], method='pearson')))
print("Positve : %.2f" %(sub_prop_by_month_year["subj"].corr(polarity_proportion_by_month_year["pos"], method='pearson')))

# Correlations (monthly periodicity 2003-2017) between the proportion of
# subjective titles and polarity scores:
# - negative: -0.40
# - neutral: 0.50
# - positive: -0.39

# The influence of the proportion of subjectivity on the proportion of neutral opinions is
# significant over the period studied between 2003 and 2017.

