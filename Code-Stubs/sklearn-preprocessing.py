#!/usr/bin/python

from sklearn import preprocessing
import numpy as np

#Scaling: essentially replacing each value with its pull/standard score 
#(to produce a normal-type distribution)
X = np.array([some values])
X_scaled = preprocessing.scale(X)

#MinMaxScaler: linearly transform data to range [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
##when given an explicit feature range feature_range=(min,max), the transformed data will be in [min,max]
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.transform(x_test)

#MaxAbsScaler works similarly, but transforms the data to the range [-1,1]; it is meant for data
#already centered at zero or sparse data

##Normalization and Binarization functions also exist

#######################################################################

#Preparing text

##Stemming is converting words to some stem (such as realize, realizes, realizing to realiz)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
temp = text_string.split()
for i in range(len(temp)):
    temp[i] = stemmer.stem(temp[i])
works = " ".join(temp)

#you can also use nltk to remove stopwords
import nltk.corpus import stopwords
sw = stopwords.words("english")
for ele in sw:
    text.replace(ele,"")

##Note: there will often be signature words (such as names) you will also want
##to remove

#sklearn implements a CountVectorizer to tokenize and count word occurrances of text (needs text).
#It has TfidfTransformer to normalize/weight tokens by their frequency (needs token counts).
#Cleverly, it has a TfidfVectorizer that does both.
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
#stop_words option can either remove preset stopwords, or a user-generated list
#lowercase option makes vectorizer case-insensitive (default True)
#min_df and max_df can be used to set min and max document frequency limits for words to be included
#floats are taken as fractions of documents; integers as numbers of documents
#max_features=n limits number of features used to top n
vectorizer.fit_transform(word_data)
#also have separate fit, transform, and inverse_transform functions
feature_names = vectorizer.get_feature_names() #returns map of feature indices to names


#######################################################################

#Decomposition: Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=<N>)
#n_components is how many dimensions are kept; by default it is min(n_samples,n_features)
x_transform_train = pca.fit_transform(x_train)
#also has separate fit function, and an inverse_transform
x_transform_test = pca.transform(x_test)
print pca.components_ #returns array of principal axes in feature space

##Note: PCA is computationally intensive; sklearn has several variations: incremental, randomized, kernal, sparse

################################################################

#k-Fold and preparing training/testing samples

#For basic splitting of data
from sklearn import datasets
from sklearn import cross_validation

x_train,x_test,y_train,y_text = cross_validation.train_test_split(data,target,test_size=0.4,random_state=0)
#options train_size or test_size are fractions representing the data devoted to each
#default is test_size=0.25
#random_state is the seed used for random sampling

#Cross-Validation strategies
#k-Fold: divide the sample into k groups or folds; use k-1 for training and 1 for testing
#Repeat k times

from sklearn.cross_validation import KFold
kf = KFold(data,n_folds=2) 
#kf will be a list of train,test pairs of indices
X = np.array([],[],[],[])
y = np.array([,,,])
X_train,X_test,y_train,y_test = X[train],X[test],y[train],y[test]
#n_folds = 3 by default, must be at least 2
#also has a shuffle and a random_state option

#Variations:
#StratifiedKFold will ensure each subset has about the same percentage 
#of each target class as the full set
#LabelKFold will ensure that the same label is not in both testing and training sets



