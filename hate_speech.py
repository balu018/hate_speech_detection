# Install necessary libraries if not already installed
!pip install textstat nltk wordcloud seaborn

# Import libraries
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
dataset = pd.read_csv("HateSpeechData.csv")

# Adding text length as a feature
dataset['text length'] = dataset['tweet'].apply(len)

# Visualizing the data distribution based on text length
graph = sns.FacetGrid(data=dataset, col='class')
graph.map(plt.hist, 'text length', bins=50)

# Defining stopwords and additional Twitter-specific words
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

# Preprocess function to clean and prepare the text data
def preprocess(tweet):
    # Remove extra spaces
    regex_pat = re.compile(r'\s+')
    tweet_space = tweet.str.replace(regex_pat, ' ', regex=True)

    # Remove @name[mention]
    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = tweet_space.str.replace(regex_pat, '', regex=True)

    # Remove links [https://abc.com]
    giant_url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweet_name.str.replace(giant_url_regex, '', regex=True)

    # Remove punctuations and numbers
    punc_remove = tweets.str.replace(r"[^a-zA-Z]", " ", regex=True)

    # Remove whitespace with a single space
    newtweet = punc_remove.str.replace(r'\s+', ' ', regex=True)

    # Remove leading and trailing whitespace
    newtweet = newtweet.str.replace(r'^\s+|\s+?$', '', regex=True)

    # Replace numbers with 'numbr'
    newtweet = newtweet.str.replace(r'\d+(\.\d+)?', 'numbr', regex=True)

    # Convert to lowercase
    tweet_lower = newtweet.str.lower()

    # Tokenize and remove stopwords
    tokenized_tweet = tweet_lower.apply(lambda x: [item for item in x.split() if item not in stopwords])

    # Stem the words
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    # Join tokens back into a single string
    processed_tweets = tokenized_tweet.apply(lambda x: ' '.join(x))

    return processed_tweets

# Apply the preprocess function
processed_tweets = preprocess(dataset['tweet'])
dataset['processed_tweets'] = processed_tweets

# Visualize most commonly used words in the dataset
all_words = ' '.join([text for text in dataset['processed_tweets']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
tfidf = tfidf_vectorizer.fit_transform(dataset['processed_tweets'])
X = tfidf
y = dataset['class'].astype(int)

# Train-test split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Model Training and Evaluation
# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)
y_preds_rf = rf.predict(X_test_tfidf)
acc_rf = accuracy_score(y_test, y_preds_rf)
report_rf = classification_report(y_test, y_preds_rf)
print("Random Forest, Accuracy Score:", acc_rf)
print(report_rf)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_preds_lr = model.predict(X_test_tfidf)
acc_lr = accuracy_score(y_test, y_preds_lr)
report_lr = classification_report(y_test, y_preds_lr)
print("Logistic Regression, Accuracy Score:", acc_lr)
print(report_lr)

# Naive Bayes
nb = GaussianNB()
X_train_nb, X_test_nb, y_train, y_test = train_test_split(X.toarray(), y, random_state=42, test_size=0.2)
nb.fit(X_train_nb, y_train)
y_preds_nb = nb.predict(X_test_nb)
acc_nb = accuracy_score(y_test, y_preds_nb)
report_nb = classification_report(y_test, y_preds_nb)
print("Naive Bayes, Accuracy Score:", acc_nb)
print(report_nb)

# Support Vector Machine (SVM)
support = LinearSVC(random_state=20)
support.fit(X_train_tfidf, y_train)
y_preds_svm = support.predict(X_test_tfidf)
acc_svm = accuracy_score(y_test, y_preds_svm)
report_svm = classification_report(y_test, y_preds_svm)
print("SVM, Accuracy Score:", acc_svm)
print(report_svm)

# Accuracy comparison bar chart
objects = ('Logistic', 'RandomForest', 'NaiveBayes', 'SVM')
y_pos = np.arange(len(objects))
performance = [acc_lr, acc_rf, acc_nb, acc_svm]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Algorithm Comparison for F1')
plt.show()
