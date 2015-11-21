#!/usr/bin/python

##First, cross-validated metrics
#The simplest way is to use the cross_val_score
from sklearn import cross_validation

clf = svm.SVC(kernel='linear',C=1)
scores = cross_validation.cross_val_score(clf,data,target,cv=5)
#This will run an SVM classifier on 5 different splits and return an array of scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2)
#By default, the score computed is the score method of the estimator: accuracy_score for classifiers
# and r2_score for regression
#This can be set bia the scoring option
#If cv is an integer, this function uses KFold (StratifiedKFold if the estimator
#derives from ClassifierMixin)
#If you don't want to use KFold, specific a different cross-validation
cv = cross_validation.ShuffleSplit(n_samples,n_iter=3,test_size=0.3,random_state=0)
cross_validation.cross_val_score(clf,data,target,cv=cv)
#You can also return the predictions for elements in the test set
predicted = cross_validation.cross_val_predict(clf,data,target,cv=10)
metrics.accuracy_score(target,predicted)
#This score will NOT exactly match the mean score


##You can also include selecting the best parameters in this process
##GridSearchCV does this on a grid of parameter options
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=0)
tuned_parameters = [{'kernal':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},
                    {'kernal':['linear'],'C':[1,10,100,1000]}]
scores = ['precision','recall']
for score in scores:
    clf = GridSearchCV(SVC(C=1),tuned_parameters,cv=5,scoring='%s_weighted'%score)
    clf.fit(x_train,y_train)
    print clf.best_params_
    for params,mean_score, scores in clf.grid_scores_:
        print ("%0.3f (+/-%0.03f) for %r" % (mean_score,scores.std()*2,params))

    y_true,y_pred = y_test,clf.predict(X_test)
    print (classification_report(y_true,y_pred))

#Checking the whole grid of parameters can be lengthy, so
#RandomizedSearchCV works by sampling parameters from a distribution
#of possible values
[{'C':scipy.stats.expon(scale=100),'gamma':scipy.stats.expon(scale=0.1),
  'kernal':['rbf'],'class_weight':['auto',None]}]


##Notes on scoring: the following are all possible scoring options
#Classification
accuracy   metrics.accuracy_score
average_precision   metrics.average_precision_score
f1   metrics.f1_score
f1_micro   metrics.f1_score
f1_macro   metrics.f1_score
f1_weighted   metrics.f1_score
f1_samples   metrics_f1_score
log_loss   metrics.log_loss
precision(_<same as f1>)   metrics.precision_score
recall(_<same as f1>)   metrics.recall_score
roc_auc   metrics.roc_auc_score
#Clustering
adjusted_rand_score   metrics.adjusted_rand_score
#Regression
mean_absolute_error   metrics.mean_absolute_error
mean_squared_error   metrics.mean_squared_error
median_absolute_error   metrics.median_absolute_error
r2   metrics.r2_score

#Other useful evaluation functions
accuracy_score(y_true,y_pred)
classification_report(y_true,y_pred)
confusion_matrix(y_true,y_pred)
f1_score(y_true,y_pred)
precision_score(y_true,y_pred)
recall_score(y_true,y_pred)

#Frankly, this is a really long page in the sklearn user's manual;
#Return to it often
