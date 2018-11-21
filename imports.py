import pandas as pd
import os
import ast
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from IPython.display import display
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as dates
from pylab import rcParams
import seaborn as sb
from numpy.random import randn
from wordcloud import WordCloud

import spacy
import collections
import operator
import pyLDAvis.sklearn
import warnings


from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer
from nltk import skipgrams
from nltk.corpus import stopwords, wordnet, subjectivity
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("vader_lexicon")
nltk.download("subjectivity")


import gensim
from gensim import corpora, models, similarities
from gensim.utils import ClippedCorpus, SlicedCorpus