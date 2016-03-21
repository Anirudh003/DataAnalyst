#!/usr/bin/python

import sys
import pickle
import numpy
import pandas
import sklearn

import matplotlib
from sklearn.ensemble import ExtraTreesClassifier
%matplotlib inline

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] 

# You will need to use more features
### Load the dictionary containing the dataset

enron_data = pickle.load(open("final_project_dataset.pkl", "r") )
f = open("poi_names.txt", "r")
df = pandas.DataFrame.from_records(list(enron_data.values()))
persons = pandas.Series(list(enron_data.keys()))
print enron_data.keys()
# Convert to numpy nan
df.replace(to_replace='NaN', value=numpy.nan, inplace=True)
print df.isnull().sum()
print "Number of POI in DataSet: ", (df['poi'] == 1).sum()
print "Number of non-POI in Dataset: ", (df['poi'] == 0).sum()


# DataFrame dimeansion
print df.shape
for column, series in df.iteritems():
    if series.isnull().sum() > 72:
        df.drop(column, axis=1, inplace=True)
       
# Drop email address column
if 'email_address' in list(df.columns.values):
    df.drop('email_address', axis=1, inplace=True)


### Load the dictionary containing the dataset
    
### Task 2: Remove outliers
### removing 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E' 
### because most of the values in it are NaN
df_imp = df.replace(to_replace=numpy.nan, value=0)
df_imp = df.fillna(0).copy(deep=True)
df_imp.columns = list(df.columns.values)
print df_imp.isnull().sum()
print df_imp.head()

df_imp.describe()
total_index = enron_data.keys().index("TOTAL")
print(total_index)
travel_index = enron_data.keys().index("THE TRAVEL AGENCY IN THE PARK")
print(travel_index)
df_subset = df_imp.drop(df_imp.index[[total_index,travel_index]])
df_subset.describe()
print "Features:", list(df_subset.columns.values)

print "Shape: ", df_subset.shape

print "Number of POI in DataSet: ", (df_subset['poi'] == 1).sum()
print "Number of non-POI in Dataset: ", (df_subset['poi'] == 0).sum()

labels = df_subset['poi'].copy(deep=True).astype(int).as_matrix()
features = (df_subset.drop('poi', axis=1)).fillna(0).copy(deep=True).as_matrix()

gnb_clf = GaussianNB()
scores = sklearn.cross_validation.cross_val_score(gnb_clf, features, labels)
print scores

rf_clf = RandomForestClassifier(n_estimators=10)
scores = sklearn.cross_validation.cross_val_score(rf_clf, features, labels)

print scores

ab_clf = AdaBoostClassifier(n_estimators=100)
scores = sklearn.cross_validation.cross_val_score(ab_clf, features, labels)
print scores 

cv = sklearn.cross_validation.StratifiedShuffleSplit(labels, n_iter=10)

def scoring(estimator, features_test, labels_test):
     labels_pred = estimator.predict(features_test)
     p = sklearn.metrics.precision_score(labels_test, labels_pred, average='binary')
     r = sklearn.metrics.recall_score(labels_test, labels_pred, average='binary')
     if p > 0.3 and r > 0.3:
            return sklearn.metrics.f1_score(labels_test, labels_pred, average='binary')
     return 0
sum=0;
for i in range (0,10):
    parameters = {'n_estimators' : [5, 10, 30, 40, 50, 100,150], 'learning_rate' : [0.1, 0.5, 1, 1.5, 2, 2.5], 'algorithm' : ('SAMME', 'SAMME.R')}
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))
    adaclf = grid_search.GridSearchCV(ada_clf, parameters, scoring = scoring, cv = cv)
    adaclf.fit(features, labels)
    sum += adaclf.best_score_
    print adaclf.best_estimator_
    print adaclf.best_score_

print "average=",+sum/10;#average= 0.407380952381
#Using Kbest
##
print features.shape
transform = SelectKBest(f_classif, k=10 )
features_new  = transform.fit_transform(features, labels)
print transform.scores_


print features_new.shape
print transform.scores_
gnb_clf = GaussianNB()
scores = sklearn.cross_validation.cross_val_score(gnb_clf, features_new, labels)
print scores

rf_clf = RandomForestClassifier(n_estimators=10)
scores = sklearn.cross_validation.cross_val_score(rf_clf, features_new, labels)

print scores

ab_clf = AdaBoostClassifier(n_estimators=100)
scores = sklearn.cross_validation.cross_val_score(ab_clf, features_new, labels)
print scores 
sum=0;
for i in range (0,10):

    parameters = {'n_estimators' : [5, 10, 30, 40, 50, 100,150], 'learning_rate' : [0.1, 0.5, 1, 1.5, 2, 2.5], 'algorithm' : ('SAMME', 'SAMME.R')}
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))
    adaclf = grid_search.GridSearchCV(ada_clf, parameters, scoring = scoring, cv = cv)
    adaclf.fit(features_new, labels)
    
    print adaclf.best_estimator_
    print adaclf.best_score_
    sum +=adaclf.best_score_
    
print sum/10; #average= 0.406666666667


### Task 3: Create new feature(s)
poi_ratio = (df_subset['from_poi_to_this_person'] + df_subset['from_this_person_to_poi']) / (df_subset['from_messages'] + df_subset['to_messages'])
fraction_to_poi = (df_subset['from_this_person_to_poi']) / (df_subset['from_messages'])
fraction_from_poi = (df_subset['from_poi_to_this_person']) / (df_subset['to_messages'])
scale = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100), copy=True)
labels = df_subset['poi'].copy(deep=True).astype(int).as_matrix()
features = (df_subset.drop('poi', axis=1)).fillna(0).copy(deep=True).as_matrix()

df_subset['poi_ratio'] = pandas.Series(poi_ratio) * 100
df_subset['fraction_to_poi'] = pandas.Series(fraction_to_poi) * 100
df_subset['fraction_from_poi'] = pandas.Series(fraction_from_poi) * 100
salary_scaled = scale.fit_transform(df_subset['salary'])
print df_subset.describe()

print "Features Used for modelling:", list(df_subset.columns.values)
### Store to my_dataset for easy export below.
### Extract features and labels from dataset for local testing
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

shuffle = sklearn.cross_validation.StratifiedShuffleSplit(labels, 4, test_size=0.3, random_state=0)

gnb_clf = GaussianNB()
scores = sklearn.cross_validation.cross_val_score(gnb_clf, features, labels)
print scores

rf_clf = RandomForestClassifier(n_estimators=10)
scores = sklearn.cross_validation.cross_val_score(rf_clf, features, labels)

print scores

ab_clf = AdaBoostClassifier(n_estimators=100)
scores = sklearn.cross_validation.cross_val_score(ab_clf, features, labels)
print scores 

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

cv = sklearn.cross_validation.StratifiedShuffleSplit(labels, n_iter=10)

def scoring(estimator, features_test, labels_test):
     labels_pred = estimator.predict(features_test)
     p = sklearn.metrics.precision_score(labels_test, labels_pred, average='binary')
     r = sklearn.metrics.recall_score(labels_test, labels_pred, average='binary')
     if p > 0.3 and r > 0.3:
            return sklearn.metrics.f1_score(labels_test, labels_pred, average='binary')
     return 0
parameters = {'max_depth': [2,3,4,5,6],'min_samples_split':[2,3,4,5], 'n_estimators':[10,20,50], 'min_samples_leaf':[1,2,3,4], 'criterion':('gini', 'entropy')}                                     
rf_clf = RandomForestClassifier()
rfclf = grid_search.GridSearchCV(rf_clf, parameters, scoring = scoring, cv = cv)
rfclf.fit(features, labels)

print rfclf.best_estimator_
print rfclf.best_score_
sum=0;
for i in range (0,10):

    parameters = {'n_estimators' : [5, 10, 30, 40, 50, 100,150], 'learning_rate' : [0.1, 0.5, 1, 1.5, 2, 2.5], 'algorithm' : ('SAMME', 'SAMME.R')}
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))
    adaclf = grid_search.GridSearchCV(ada_clf, parameters, scoring = scoring, cv = cv)
    adaclf.fit(features, labels)
    
    print adaclf.best_estimator_
    print adaclf.best_score_
    sum +=adaclf.best_score_
    
print sum/10;


rf_best_clf = rfclf.best_estimator_
list_cols = list(df_subset.columns.values)
list_cols.remove('poi')
list_cols.insert(0, 'poi')
data = df_subset[list_cols].fillna(0).to_dict(orient='records')
enron_data_sub = {}
counter = 0
for item in data:
    enron_data_sub[counter] = item
    counter += 1
    
    
test_classifier(rf_best_clf, enron_data_sub, list_cols)
ada_best_clf = adaclf.best_estimator_
test_classifier(ada_best_clf, enron_data_sub, list_cols)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(ada_best_clf, enron_data_sub, list_cols)


###helper functions

    