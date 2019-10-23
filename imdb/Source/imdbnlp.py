import re
import nltk
#nltk.download('stopwords')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')
stop_words.remove('not')
lemmatiser = WordNetLemmatizer()

def dataPreprocessing(review):

    #lowercase
    review = review.lower()

    #tokenizer
    tokens = word_tokenize(review)

    #stop word removal
    review = [word for word in review if word not in stop_words]

    #lmmtization
    review = [lemmatiser.lemmatize(word) for word in review]

    review = ' '.join(review)

    return review

#Extractingthe data
'''
#Extracting the data and storing it to a file
#Get all the training pos and neg reviews
#initialise all the paths
path = ["C:/Rahul/NLP/imdb/aclImdb/train/pos/", "C:/Rahul/NLP/imdb/aclImdb/train/neg/", "C:/Rahul/NLP/imdb/aclImdb/test/neg/", "C:/Rahul/NLP/imdb/aclImdb/test/neg/"]
#initialize lists
ids = []
rating = []
review = []
content = ""

#for each path, get all the files and get the reviews
for path_name in path:
    file_names = os.listdir(str(path_name))
    for files in file_names:
        names = files.split(".")
        idrating = names[0].split("_")
        file_content = open(str(str(path_name)+str(files)),"r", encoding="utf-8")
        content = file_content.read()
        ids.append(idrating[0])
        rating.append(idrating[1])
        review.append(content)
        file_content.close()

#write data to a csv file
df = pd.DataFrame({'id':ids,'rating':rating,'review':review})
print(df.head())
df.to_csv('C:/Rahul/NLP/imdb/IMDBreviews.csv')
'''

#load the data
import pandas as pd
df = pd.read_csv('C:/Rahul/NLP/imdb/IMDBreviews.csv')
df.drop(columns='Unnamed: 0', axis=1, inplace=True)
print(df.columns)
print(df.head)

#Remove spcl characters using RE
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
#print(data.columns)
data.drop('rating', axis = 1, inplace = True)
data.drop('review', axis = 1, inplace = True)
data.drop('id', axis = 1, inplace = True)

#Split for train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data, y, test_size=0.2,stratify=y)
print("training ", xtrain.shape,ytrain.shape)
print("test ", xtest.shape,ytest.shape)

#Creating with bow
'''
#Bow
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(min_df=10, tokenizer=lambda doc: doc,lowercase=False,stop_words="english")

xtrainbow = vector.fit_transform(xtrain['preprocessed_review'])
xtestbow = vector.transform(xtest['preprocessed_review'])

print("xtrainbow ", xtrainbow.shape)
print("xtestbow ", xtest.shape)
'''
#TfIdf vectoriser
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=10, tokenizer=lambda doc: doc,lowercase=False,ngram_range=(1, 2))
xtraintfidf = tfidf.fit_transform(xtrain['preprocessed_review'])
xtesttfidf = tfidf.transform(xtest['preprocessed_review'])
print("xtraintfidf ", xtraintfidf.shape)
print("xtesttfidf ", xtesttfidf.shape)

#building classifiers
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#with bow data
clf = MultinomialNB()
clf.fit(xtrainbow,ytrain)

ypredbow = clf.predict(xtestbow)
print("accuraccy : ", accuracy_score(ytest, ypredbow))
#model.save('C:\Rahul\NLP\imdb\imdbmodelbow.h5')
#model.save_weights('C:\Rahul\NLP\imdb\imdbweightsbow.h5')

#with iftdf data
clf.fit(xtraintfidf, ytrain)
yprediftdf = clf.predict(xtesttfidf)
print("accuraccy : ", accuracy_score(ytest, yprediftdf))
#model.save('C:\Rahul\NLP\imdb\imdbmodeliftdf.h5')
#model.save_weights('C:\Rahul\NLP\imdb\imdbweightsiftdf.h5')

#logestic regression model
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression()
logclf.fit(xtraintfidf,ytrain)

logypred = logclf.predict(xtesttfidf)
print("Logestic regression accuraccy : ", accuracy_score(ytest, logypred))

#ploting and metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(ytest, logypred)
plt.figure(figsize=(5,3))
sns.heatmap(cm,annot=True,fmt='d')
plt.title('Test confusion matrix')
plt.show()
'''

#LineatSVM model
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(xtraintfidf,ytrain)

svcpred = svc.predict(xtesttfidf)
print("SVC accuraccy : ", accuracy_score(ytest, svcpred))

#ploting and metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(ytest, svcpred)
plt.figure(figsize=(8,8))
sns.heatmap(cm,annot=True,fmt='d')
plt.title('Test confusion matrix')
plt.show()



