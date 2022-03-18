#!/usr/bin/env python
# coding: utf-8

# # Utilized the scrappy package to scrape reviews of the noise smartwatch and stored them in a file named reviews2.

# # Import Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Load Data sets

# In[2]:


data = pd.read_csv("E:\DS Assignments1\\reviews2.csv",encoding='Latin-1')
data


# Our attention will be focused on the "rating", "title", and "review" columns, since they contain the most qualitative information. The "rating" column would not be subjected to any pre-processing. Let's start by dropping the "name" column

# In[3]:


data.drop(['name'],inplace=True,axis=1)
data.head()


# In[4]:


data["reviews"] = data["title"]+data["review"]
data.head()


# Merging review and title column & create reviews column

# In[5]:


data.drop(['title'],inplace=True,axis=1)
data.drop(['review'],inplace=True,axis=1)
data.head()


# In[6]:


sample_data = data.copy()


# # Data Preprocessing

# Looking for NA values

# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


data[data.isnull().any(axis=1)].head()


# In[10]:


data = data.dropna(axis=0)


# In[11]:


data.isnull().sum()


# In[12]:


data.head()


# Descriptive Statistics

# In[13]:


data.describe()


# In[14]:


data['rating'].value_counts()


# # Visualization

# In[15]:


sns.histplot(data.rating)
plt.show()


# In[16]:


data.rating.value_counts().plot(kind='pie')
plt.show()


# check duplicate

# In[17]:


data.duplicated().sum()


# # Text Preprocessing

# In[18]:


import nltk
nltk.download('punkt')


# In[19]:


from nltk.corpus import stopwords
import string
string.punctuation
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def clean_review(text):
    text=text.lower()   #lower
    text=nltk.word_tokenize(text) #tokenize 
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
           
    text=y[:]  
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english')and i not in string.punctuation:   #Removing StopWords
            y.append(i)
         
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))   #Stemming
    
    return " ".join(y)


# In[20]:


data['clean_review']=data['reviews'].apply(clean_review)


# In[21]:


pd.options.display.max_rows=None
data


# # Generate Word Cloud

# In[22]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(50,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

from wordcloud import WordCloud, STOPWORDS

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=5000,height=3000,background_color='black',max_words=500,
                   colormap='Set1',stopwords=STOPWORDS).generate(str(data['clean_review']))
plot_cloud(wordcloud)
plt.show()


# # Feature Extaction

# # 1. Using CountVectorizer

# Import Libraries

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# In[24]:


cv=CountVectorizer()
reviewcv=cv.fit_transform(data['clean_review'])
print(cv.get_feature_names())


# In[25]:


cv = CountVectorizer()

reviewcv = cv.fit_transform(data['clean_review'])
sum_words = reviewcv.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word', 'count']


pd.options.display.max_rows=None
wf_df


# # 2. CountVectorizer with N-grams (Bigrams & Trigrams)

# Bi-gram

# In[26]:


#Bi-gram
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2), 
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[27]:


top2_words = get_top_n2_words(data['clean_review'], n=5000) 
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df


# Bi-gram plot

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (8, 5))
plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])
plt.show()


# Tri-gram

# In[29]:


def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[30]:


top3_words = get_top_n3_words(data['clean_review'], n=5000)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]

top3_df


# Tri-gram plot

# In[31]:


import seaborn as sns
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])
plt.show()


# # Named Entity Recognition (NER)

# Part Of Speech Tagging

# In[32]:


import string 
import re #regular expression
import spacy


# In[33]:


nlp = spacy.load("en_core_web_sm")

one_block = str(data['clean_review'])
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[34]:


nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs)


# In[35]:


len(nouns_verbs)


# Counting the noun & verb tokens

# In[36]:



from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df


# In[37]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(8,5),title='Top 10 nouns and verbs');
plt.show()


# # Generate Word Cloud

# In[38]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

from wordcloud import WordCloud, STOPWORDS

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=5000,height=3000,background_color='black',max_words=100,
                   colormap='Set1',stopwords=STOPWORDS).generate(str(wd_df))
plot_cloud(wordcloud)
plt.show()


# In[39]:


data


# In[40]:


data['num_words']=data['clean_review'].apply(lambda x:len(nltk.word_tokenize(x)))


# count the no of word in reviews

# In[41]:


data


# # Define Subjectivity & Polarity

# In[42]:


from textblob import TextBlob
def getSubjectivity(clean_review):
    return TextBlob(clean_review).sentiment.subjectivity

def getPolarity(clean_review):
    return TextBlob(clean_review).sentiment.polarity

data['Subjectivity'] = data['clean_review'].apply(getSubjectivity)
data['Polarity'] = data['clean_review'].apply(getPolarity)


# In[43]:


data


# # Create Positive, Negative & Neutral Reviews

# In[44]:


# function to analyze the reviews
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

    
data['Analysis'] = data['Polarity'].apply(getAnalysis)


# In[45]:


data


# In[46]:


data.reset_index(inplace=True)
data


# In[47]:


data.drop(['index'],inplace=True,axis=1)


# In[48]:


data


# # Generate Positive Reviews Word Cloud

# In[49]:


from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,max_words=300,background_color='black')


# In[50]:


Positive=wc.generate(data[data['Polarity']>0]['clean_review'].str.cat(sep=""))


# In[51]:


plt.figure(figsize=(10,10))
plt.imshow(Positive)
plt.show()


# # Generate Negative Reviews Word Cloud

# In[52]:


Negative=wc.generate(data[data['Polarity']<0]['clean_review'].str.cat(sep=""))


# In[53]:


plt.figure(figsize=(10,10))
plt.imshow(Positive)
plt.show()


# # Generate Neutral Reviews Word Cloud

# In[54]:


Neutral=wc.generate(data[data['Polarity']==0]['clean_review'].str.cat(sep=""))


# In[55]:


plt.figure(figsize=(10,10))
plt.imshow(Positive)
plt.show()


# In[56]:


import seaborn as sns


# Comparing Reviews

# In[57]:


plt.figure(figsize=(8,5))
sns.set_style("whitegrid")
sns.countplot(x="Analysis",data=data )
plt.show()


# # Positive reviews

# In[58]:


j=1
sortedDF = data.sort_values(by=['Analysis'])
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Positive'):
        print(str(j) + ') '+sortedDF['clean_review'][i])
        print()
        j=j+1


# # Negative reviews

# In[59]:


j=1
sortedDF = data.sort_values(by=['Analysis'])
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Negative'):
        print(str(j) + ') '+sortedDF['clean_review'][i])
        print()
        j=j+1


# # Neutral reviews

# In[60]:


j=1
sortedDF = data.sort_values(by=['Analysis'])
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Neutral'):
        print(str(j) + ') '+sortedDF['clean_review'][i])
        print()
        j=j+1


# In[61]:


data


# In[62]:


import plotly.express as px
data['Analysis'] = data['Analysis'].replace({-1 : 'negative'})
data['Analysis'] = data['Analysis'].replace({1 : 'positive'})
fig = px.histogram(data, x="Analysis")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[63]:


# assign reviews with score > 3 as positive sentiment
# score < 3 negative sentiment
# remove score = 3
data = data[data['rating'] != 3]
data['Analysis'] = data['rating'].apply(lambda rating : +1 if rating > 3 else -1)


# # The new data frame should only have two columns — “ clean_review ” (the review text data), and “Analysis” (the target variable).

# In[64]:


#Splitting Dataframe
dfNew = data[['clean_review','Analysis']]
dfNew.head()


# # We will now split the data frame into train and test sets. 80% of the data will be used for training, and 20% will be used for testing.

# In[99]:


# random split train and test data
index =dfNew .index
dfNew['random_number'] = np.random.randn(len(index))
train = dfNew[dfNew['random_number'] <= 0.8]
test = dfNew[dfNew['random_number'] > 0.8]


# In[100]:


dfNew


# In[101]:


# count vectorizer:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['clean_review'])
test_matrix = vectorizer.transform(test['clean_review'])


# # Model Building

# In[102]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=6000)


# In[103]:


#Split target and independent variables
X_train = train_matrix
X_test = test_matrix
y_train = train['Analysis']
y_test = test['Analysis']


# In[104]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[105]:


#Fitting Model on Data
lr.fit(X_train,y_train)


# In[106]:


#predictions
predictions = lr.predict(X_test)
print(predictions)


# We have successfully built a simple logistic regression model, and trained the data on it. We also made predictions using the model.

# # Testing

# In[107]:


# find accuracy, precision, recall:

from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)


# In[108]:


print(classification_report(predictions,y_test))


# The overall accuracy of the model on the test data is around 87%, which is pretty good considering we didn’t do any feature extraction or much preprocessing.

# In[109]:


from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.api as smf
roc_score=roc_auc_score(predictions,y_test)
roc_score


# In[110]:


# Confusion Matrix for the model accuracy
cm = confusion_matrix(predictions,y_test)
print(cm)


# In[ ]:




