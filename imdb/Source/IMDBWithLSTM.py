from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#nltk.download('wordnet')
#Load the data set
df = pd.read_csv('C:/Rahul/NLP/IMDBreviewsPreProcessed.csv')
df.drop(columns='Unnamed: 0', axis=1, inplace=True)
print(df.columns)
print(df.head)
#Add label for ach review
y1 = df['rating'].values
Label = []
for i in y1:
    if i <=4:
        Label.append("Bad")
    elif i >4 and i <+7:
        Label.append("Good")
    else:
        Label.append("Best")

df['label'] = Label
print(df.head)

#Add
data = df.copy()
data.drop('id', axis = 1, inplace = True)
print(data.head)
print(data.columns)

#Word count
data['wordcount'] = data['review'].apply(lambda x: len(str(x).split(" ")))
print(data['wordcount'].describe())
#plot the word count
plt.hist(data['wordcount'], bins = 400)
plt.xlim(0, 1000)
plt.show()

#Stopwords and their frequencies
'''

preprocessed_review = []
for Review in df['review']:
    review = re.sub('[^A-Za-z0-9]+', ' ', Review)
    preprocessed_review.append(review)

df['preprocessed_review'] = preprocessed_review
#print(df.head())
print(df['preprocessed_review'].head)
#df.to_csv('C:/Rahul/NLP/imdb/IMDBreviewsPreProcessed.csv')

#Preprocessing complete
lowercase = []
for review in df['preprocessed_review']:
    Review = review.lower()
    tokens = word_tokenize(Review)
    Review = [word for word in tokens if word not in stop_words]
    Review = [lemmatiser.lemmatize(word) for word in Review]
    lowercase.append(Review)

#print(lowercase[0])
df['preprocessed_review'] = lowercase
#df.to_csv('C:/Rahul/NLP/imdb/IMDBreviewsPreProcessed.csv')
print(df['preprocessed_review'].head())

#vectorization
data = df.copy()

#Create labels for movierating as Good, Bad, Best
y1 = df['rating'].values
y = pd.DataFrame
Label = []
for i in y1:
    if i <=4:
        Label.append("Bad")
    elif i >4 and i <+7:
        Label.append("Good")
    else:
        Label.append("Best")

y = pd.DataFrame({'label':Label})

'''
data.drop('review', axis = 1, inplace = True)
print(data.columns)

#enerate wordcloud
split_words = ' '.join(data['preprocessed_review'])
wordcloud = WordCloud( width = 1000, height = 1000, background_color = 'black', max_words = 1000, min_font_size = 20).generate(split_words)

#plot wordcloud
plt.figure(figsize = (12,12),facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#ngrams feature
x = data['preprocessed_review']
y = data['label']

#Train test split
xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.2,stratify=y)
xtrain, ytrain = np.array(xtrain), np.array(ytrain)
print("training ", xtrain.shape,ytrain.shape)
print("test ", xtest.shape,ytest.shape)

#form vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=10, tokenizer=lambda doc: doc,lowercase=False,ngram_range=(1, 2))
xtraintfidf = tfidf.fit_transform(xtrain)
xtesttfidf = tfidf.transform(xtest)
print("xtraintfidf ", xtraintfidf.shape)
print("xtesttfidf ", xtesttfidf.shape)

#LSTM
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Dropout
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(10000,128))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
#output layers
model.add(Dense(units=1))

#Model compile
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

#Train the model
xtraintfidf = xtraintfidf.reshape(xtraintfidf, (xtraintfidf.shape[0],xtraintfidf.shape[1],1))
model.fit(xtraintfidf, ytrain, epochs = 25, batch_size = 100)
