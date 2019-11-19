import re
import nltk
#nltk.download('stopwords')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

stop_words = stopwords.words('english')
stop_words.remove('not')
lemmatiser = WordNetLemmatizer()

def remove(input, pattern):
    temp = re.findall(pattern, input)
    for i in temp:
        input = re.sub(i, '', input)

    return input

#load the data
data = pd.read_csv('C:/Rahul/NLP/av/train_2kmZucJ.csv')
data.drop('id', axis=1,inplace=True)
print(data.columns)

#remove twitter handles
print('removing twitter handles')
data['tidy'] = np.vectorize(remove)(data['tweet'], "@[\w]*")
print(data.head)

#removing short word
data['tidy'] = data['tidy'].apply(lambda x: ' '.join([w for w in x.split() if len(w) >3]))
print(data['tidy'].head)

#stemming and tokenizing
Temp = []
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()
for i in data['tidy']:
    i = i.lower().split()
    temp = [lemmatiser.lemmatize(temp) for temp in i]
    Temp.append(temp)

data['tidy'] = Temp
print(data.head)

#Printing wordcloud
split_words = ' '.join([str(x) for x in data['tidy']])
wordcloud = WordCloud( width = 1000, height = 1000, background_color = 'black', max_words = 1000, min_font_size = 20).generate(split_words)

#plot wordcloud
plt.figure(figsize = (12,12),facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#word clod for negative tweets
neg_words = ' '.join([str(x) for x in data['tidy'][data['label'] == 0]])
wordcloud = WordCloud( width = 1000, height = 1000, background_color = 'black', max_words = 1000, min_font_size = 20).generate(neg_words)

#plot wordcloud
plt.figure(figsize = (12,12),facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#word clod for positive tweets
neg_words = ' '.join([str(x) for x in data['tidy'][data['label'] == 1]])
wordcloud = WordCloud( width = 800, height = 800, background_color = 'black', max_words = 1000, min_font_size = 20).generate(neg_words)

#plot wordcloud
plt.figure(figsize = (12,12),facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#remove twitter handles
print('getting # tags')
tags = []
for i in data['tidy']:
    temp = re.findall(r"#[\w]*", str(i))
    tags.append(temp)
data['tags'] = tags

Tags = sum(tags, [])
Tags = nltk.FreqDist(Tags)

Tag = pd.DataFrame({'tags': list(Tags.keys()), 'value': list(Tags.values())})
d = Tag.nlargest(columns = 'value', n=10)
sns.barplot(data=d, x='tags', y='value')


