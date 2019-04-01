#http://brandonrose.org/clustering
#https://github.com/rahgoar/clustering_practices/blob/master/NLTK_KMeans.py
"""
Created on Wed Feb 27 17:34:06 2019

@author: Preeti, Sofana, Tarun, Rajhav
"""

# importing libraries
import nltk
import re
from urllib import request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd


#Data preparation and preprocess
def custom_preprocessor(text):
        text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations
        text =  re.sub(r'\s+',' ',text) #remove multiple spaces into a single space
        text = re.sub(r"\s+[a-zA-Z]\s+",' ',text) #remove a single character
        text = text.lower()
        text = nltk.word_tokenize(text)       #tokenizing
        text = [word for word in text if not word in stop_words] #English Stopwords
        text = [lemmatizer.lemmatize(word) for word in text]              #Lemmatising
        return text



filepath_dict = {'Book1':   'https://www.gutenberg.org/files/58764/58764-0.txt',
               'Book2': 'https://www.gutenberg.org/files/58751/58751-0.txt',
               'Book3':   'http://www.gutenberg.org/cache/epub/345/pg345.txt'}


for key, value in filepath_dict.items():
   if (key == "Book1"):
        bookLoc = filepath_dict[key]
        response = request.urlopen(bookLoc)
        raw = response.read().decode('utf-8')
        len(raw)
        first_book = custom_preprocessor(raw)
   elif (key == "Book2"):
        bookLoc = filepath_dict[key]
        response = request.urlopen(bookLoc)
        raw = response.read().decode('utf-8')
        len(raw)
        second_book = custom_preprocessor(raw)
   elif (key == "Book3"):
        bookLoc = filepath_dict[key]
        response = request.urlopen(bookLoc)
        raw = response.read().decode('utf-8')
        len(raw)
        third_book = custom_preprocessor(raw)
   else:
       pass


##Building First Book
first_book_text = ' '.join(first_book)
fileLoc = 'C:\\Users\\Hp\\EBC7100\\FirstBook\\a.txt'
with open(fileLoc, 'a') as fout:
    fout.write(first_book_text)
    fout.close()

#Building Second Book
second_book_text = ' '.join(second_book)
fileLoc = 'C:\\Users\\Hp\\EBC7100\\SecondBook\\b.txt'
with open(fileLoc, 'a') as fout:
    fout.write(second_book_text)
    fout.close()


#Building Third Book
third_book_text = ' '.join(third_book)
fileLoc = 'C:\\Users\\Hp\\EBC7100\\ThirdBook\\c.txt'
with open(fileLoc, 'a') as fout:
    fout.write(third_book_text)
    fout.close()


# labeling
# Cretaing tuple
# aBooklist = []
def readAtxtfile(bookText, docs, labels):
    x = 0
    i = 0
    n = 150
    while x < 200:
        temp = ""
        words = bookText.split(" ")[i:n]
        for word in words:
            temp = word + " " + temp
        docs.append(temp)
        labels.append(0)
        i += 150
        n += 150
        x += 1
    return docs, labels


# Cretaing tuple
# bBooklist = []
def readBtxtfile(bookText, docs, labels):
    x = 0
    i = 0
    n = 150
    while x < 184:
        temp = ""
        words = bookText.split(" ")[i:n]
        for word in words:
            temp = word + " " + temp
        docs.append(temp)
        labels.append(1)
        i += 150
        n += 150
        x += 1
    return docs, labels

# Cretaing tuple
# cBooklist = []
def readCtxtfile(bookText, docs, labels):
    x = 0
    i = 0
    n = 150
    while x < 200:
        temp = ""
        words = bookText.split(" ")[i:n]
        for word in words:
            temp = word + " " + temp
        docs.append(temp)
        labels.append(2)
        i += 150
        n += 150
        x += 1
    return docs, labels


docs = []
labels = []
docs, labels = readAtxtfile(first_book_text, docs, labels)
# print(aBooklist)
docs, labels = readBtxtfile(second_book_text, docs, labels)
# print(bBooklist)
docs, labels = readCtxtfile(third_book_text, docs, labels)
# print(cBooklist)


#print(len(docs))
#print(docs)
#print(labels)
#print(len(labels))


#Tf-IDF Model Implementation
# Creating the Tf-Idf model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6)
TF_X = vectorizer.fit_transform(docs)
TF_X.toarray()
type(TF_X)
#**********************Expectation Maximization********************************
def tfidf_EM(TF_X):
    from sklearn.mixture import GaussianMixture
    X_EM = TF_X.toarray()
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm = gmm.fit(X_EM)
    EM_labels = gmm.predict(X_EM)
    return EM_labels

#*****************************calculation**************************************
from scipy.stats import spearmanr
from time import time
from sklearn import metrics

name = 'EM-tfidf'
t0 = time()
EM_label = tfidf_EM(TF_X)
print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tkappa\tcorr\tsilh_Clus\tsilh_HMN')
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%-9s\t%.3f\t%.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, EM_label),
             metrics.completeness_score(labels, EM_label),
             metrics.v_measure_score(labels, EM_label),
             metrics.adjusted_rand_score(labels, EM_label),
             metrics.adjusted_mutual_info_score(labels,  EM_label),
             metrics.cohen_kappa_score(labels, EM_label,weights='linear'),
             str(spearmanr(labels,EM_label)),
             metrics.silhouette_score(TF_X, EM_label,
                                      metric='euclidean'),
             metrics.silhouette_score(TF_X, labels,
                                      metric='euclidean'),
             ))

#**************************error analysis**************************************
from sklearn.metrics.cluster import contingency_matrix
x = labels #actual labels
y = EM_label #predicted labels
error_analysis = contingency_matrix(x, y)
#***************************plot************************************************
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
# 0. Create dataset
X,Y = make_blobs(cluster_std=0.35,random_state=0,n_samples=1000,centers=3)
# Stratch dataset to get ellipsoid data
X = np.dot(X,np.random.RandomState(0).randn(2,2))
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T
GMM = GaussianMixture(n_components=3).fit(TF_X.toarray()) # Instantiate and fit the model
print('Converged:',GMM.converged_) # Check if the model has converged
means = GMM.means_
covariances = GMM.covariances_
# Predict
Y = np.array([[0.5],[0.5]])
#prediction = GMM.predict_proba(Y.T)
#print(prediction)
# Plot
fig = plt.figure(figsize=(8,5))
ax0 = fig.add_subplot(111)
ax0.scatter(X[:,0],X[:,1])
ax0.scatter(Y[0,:],Y[1,:],c='grey',zorder=10,s=100)
for m,c in zip(means,covariances):
    multi_normal = multivariate_normal(mean=m,cov=c)
    ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)

plt.show()
#********************************************************************
from sklearn.mixture import GaussianMixture
from itertools import islice
from itertools import cycle
import pandas as pd

type(TF_X)
clf = GaussianMixture(n_components=3, covariance_type='full')
clf.fit(TF_X.toarray())
EM_label = clf.predict(TF_X.toarray())
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                 '#f781bf', '#a65628', '#984ea3',
                                 '#999999', '#e41a1c', '#dede00']),
                          int(max(EM_label) + 1))))
plt.scatter(TF_X[0,:], TF_X[1,:], s=10, color=colors[EM_label])
plt.show()
