#!/usr/bin/python
import numpy
#####################################
##Supervised Learning
#####################################

##Gaussian Naive Bayes: classification
##The likelihood of features is assumed to be Gaussian, with mean and sigma
##estimated using maximum likelihood.  Class with higher probability of
##generating the point is given as classification.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#no arguments
clf.fit(features_train,labels_train)
#after fitting, the class_prior_(probability of each class), class_count_(training samples in each class)
#theta_ (mean of each feature_, and sigma_ (variance of each feature) are available
score = clf.score(features_test,labels_test)

#Other Naive Bayes classifiers use different probability distributions; Multinominal and Bernoulli
#are implemented in sklearn

#####################################

##Support Vector Machines: classification, regression, outlier detection
##This defines a hyper-plane as a decision boundary between classes;
##a kernel function can be used to translate to higher dimensions where
##separations are linear.
##This is more likely to need feature rescaling, so that 'distances' have meaning.
from sklearn import svm
clf = svm.SVC(C=10000.0,kernel='rbf')
#sklearn has three types of classifier: SVC, NuSVC, LinearSVC
#kernel option specifics function used: rbf(default),linear,poly,sigmoid,precomputed, or a callable
#LinearSVC class uses a linear kernel and doesn't have this option
#C: penalty parameter of error term; contrals trade-off between smooth boundary and accurate classification
#gamma: kernel coefficient of rbf, poly, and sigmoid
#class_weight: controls weighting of classes; their importance in classification
clf.fit(features_train,labels_train)
pred = clf.predict(features_test_)
accuracy = accuracy_score(label_test,pred)

##Can also be used for regression: SVR, NuSVR, and LinearSVR
##Novelty/outlier detection can be done with OneClassSVM

#####################################

##Decision Tree: classification, regression
##Make a series of cuts; feature/value used is selected to maximize information gain, or 
##via some other criterion
##They are easy to visualize and don't require feature rescaling, but are prone to 
##overfitting if dataset has a large number of features
##Unbalanced datasets (with a high fraction of one class) are also problematic; balance
##your dataset first.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)
##This can handle binary and multiclass labelling
#criterion: how the splits are calculated/evaluated.  Can be 'entropy' or 'gini' (default)
#splitter: can be 'best' (default) or 'random'
#max_features: number of features to consider when looking for best split; can be an int, a float (a fraction), "auto" (sqrt(n)),
#sqrt (same as auto), log2 (log2(n)), or None (will use all n).
#max_depth: levels of the tree
#min_samples_split: the minimum number of sample needed to split a node
#max_leaf_nodes: trees grows until it has this many leaves, in best-first fashion
#class_weight: can be a dictionary of class_label:weights, or "balanced" to automatically adjust class weights
#random_state: seed for finding best splits
clf.fit(features_train,labels_train)
#after fitting, it is possible to get clf.feature_importances_ to compare most valuable features
pred = clf.predict(features_test)

print accuracy_score(labels_test,pred)

#DecisionTreeRegressor extends this to regression problems

#####################################

##k-Nearest Neighbors
##Instance-based learning where class of test point is based on classes
##of the k nearest neighbors
##Since distances are being calculated, features will probably need scaling
##Can be used for regression as well
import sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10,weights='distance')
#n_neighbors: also the k of the name.  Larger k decreases impact of noise, but makes boundaries less distinct
#weights: uniform-all neighbors have equal weight, distance-neighbors weighted by inverse distance, or callable
#algorithm used to compute nearest neighbors: ball_tree, kd_tree, brute, or auto (which will pick one)
#metric: how the distance is calculated; default is minkowski
#p: power parameter of Minkowski metric: 1 = manhatten distance, 2 = euclidean distance
clf.fit(features_train,labels_train)
print clf.score(features_test,labels_test)

#####################################

##Random Forest
##A group of many decision trees; output from each is averaged to produce the final result
##Each tree is built with a subsample drawn with replacement from the training set.
##Splits are also created using only a random subset of the features.
##Available as RandomForestClassifier and RandomForestRegressor
##Extremely Randomized Trees go one step further by randomly generated thresholds
##and picking the best of that set (ExtraTreesClassifier and ExtraTreesRegressor)
rf = RandomForestClassifier(n_estimators=100)
#DecisionTree options can also be used here
rf.fit(features_train,labels_train)
print rf.feature_importances_
print clf.sccore(features_test,labels_test)

#####################################

##k-Means
##A cluster algorithm that works by (a) randomly selecting point, (b) assign closest points
##to each center, (c) move center such that total quadratic error is minimized, (d) repeat
##with new centers until stable solution is found
##Solutions are not unique! Exact solution depends on clustering and number of centers,
##and can get stuck in local minima
from sklearn.cluster import KMeans
cls = KMeans(n_clusters=2)
##n_clusters: number of clusters/centroids to generate
##max_iter: maximum number of iterations to do
##n_init: number of times algorithm will run with different centroid seeds.  Solution will be best output of n_init runs
pred = cls.fit_predict(finance_features)
##cluster_centers_, inertia, and labels_ available after fitting

#####################################

##Linear Regression via Ordinary Least Squares
##Line is fitted to points by minimizing the deviations from the line squared
##This assumes inputs are uncorrelated!
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(feature_train,target_train)
#After fitting, coef_ will return an array of coefficients and intercept_ the 
#independent term
reg.score(feature_test,target_test)
##The score returned here is the r^2

##Ridge regression expands this by minimizing the SSE + alpha*(coeff_i)^2
##This penalizes large coefficients; larger values of alpha make coefficients
##more robust to collinearity
reg = linear_model.Ridge(alpha = 0.5)
##RidgeCV implements this with built-in cross-validation of the alpha parameter

##Lasso regression is meant for estimating sparse (ie, as few as possible) coefficients; it favors solutions with
##fewer parameters, reducing the number of variables on which the solution depends
##It minimizes SSE+alpha*||coeff_i||
reg = linear_model.Lasso(alpha=0.1)
##LassoCV also exists
