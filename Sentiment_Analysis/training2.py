import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import nltk
from nltk.corpus import stopwords
from textblob import Word
from nltk.tokenize import word_tokenize
import string

def process_string(s):
    s = " ".join(x.lower() for x in s.split())
    s = ''.join([i for i in s if not i.isdigit()])
    s = s.translate(str.maketrans('', '', string.punctuation))
    sw = stopwords.words("english")
    s = " ".join(x for x in s.split() if x not in sw)
    s = " ".join([Word(s).lemmatize()])
    s = TextBlob(s).words
    return s

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('./Sentiment_Analysis/Mental-Health-Twitter-Modified.csv')

# Change all characters in tweets to lower case
df["post_text"] = df["post_text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Remove numbers from tweets
df["post_text"] = df["post_text"].str.replace("\d","")

# Remove punctuation from tweets
df["post_text"] = df["post_text"].str.replace("[^\w\s]","")

# Remove stop words
# nltk.download("stopwords")
sw = stopwords.words("english")
df["post_text"] = df["post_text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

# nltk.download("wordnet")
# nltk.download("omw-1.4")
df["post_text"] = df["post_text"].apply(lambda x: " ".join([Word(x).lemmatize()]))

# nltk.download('punkt')
df["tokens"] = df["post_text"].apply(lambda x: TextBlob(x).words)

# Applying sentiment to entire dataset

blob_emptylist = []

for i in df["post_text"]:
    blob = TextBlob(i).sentiment # returns polarity
    blob_emptylist.append(blob)

# Create a new dataframe to show polarity and subjectivity for each tweet
df2 = pd.DataFrame(blob_emptylist)

df3 = pd.concat([df.reset_index(drop=True), df2], axis=1)
df4 = df3[['post_text','tokens','polarity','severity','severity_int']]
# print(df4["polarity"])
# print(df4["severity_int"])
# df4["Sentiment"] =  np.where(df4["polarity"] >= 0 , "Positive", "Negative")
df4["10Polarity"] = df4["polarity"].round(1) * 10
df4["roundedPolarity"] = df4['polarity'].round(1)

# print(df4["polarity"])
# print(df4["10polarity"])
# print(df4["roundedPolarity"])

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# from sklearn.datasets import make_regression

# X, y = make_regression(n_samples=10, n_features=4, random_state=0)

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(df4['post_text'], df4['10Polarity'], test_size=0.2, random_state=2)

# Convert the text data into numerical features using a CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# print(X_train)
# print(y_train)

# X_train, X_test, y_train, y_test = train_test_split(df4['roundedPolarity'], df4['severity'], test_size=0.2, random_state=2)

# print(X_train)

# print(X_train)
# print(y_train)
# X_train = np.reshape(X_train, (-1, 1))
# X_test = np.reshape(X_test,(-1, 1))

# Create a DecisionTreeRegressor
# regr = DecisionTreeRegressor()  # You can adjust max_depth as needed
# regr = RandomForestRegressor(max_depth=100)
regr = DecisionTreeClassifier()
# regr = RandomForestClassifier()
# regr = KNeighborsClassifier()
# regr = MultinomialNB()
regr.fit(X_train, y_train)

# Evaluate the classifier on the testing set
accuracy = regr.score(X_test, y_test)
print('Accuracy:', accuracy)

import joblib

# Save the model to a file
model_filename = "./Sentiment_Analysis/model/model.pkl"
joblib.dump(regr, model_filename)

# Create a classification report
# print(classification_report(y_test, regr.predict(X_test)))