#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi', 'from_poi_to_this_person','from_this_person_to_poi']
feature_names = features_list[1:]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

#check total_payments and total_stock_values
payment_features = ['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
					'expenses', 'loan_advances','long_term_incentive', 'other', 'salary']
stock_features = ['exercised_stock_options','restricted_stock','restricted_stock_deferred']
outliers = []
for key, val in data_dict.iteritems():
	total_payments_calc = 0
	total_stock_values_calc = 0
	for k, v in val.iteritems():
		if k in payment_features and v != "NaN":
			total_payments_calc += v
		elif k in stock_features and v!= "NaN":
			total_stock_values_calc += v

	if val['total_payments'] == "NaN":
		total_payments = 0
	else:
		total_payments = val['total_payments']
	if val['total_stock_value'] == "NaN":
		total_stock = 0
	else:
		total_stock = val['total_stock_value']

	if total_payments_calc != total_payments or total_stock_values_calc != total_stock:
		outliers.append(key)

for outlier in outliers:
	pprint.pprint(data_dict[outlier])
	data_dict.pop(outlier)
	print(outlier), "Removed as outlier since payments or stock valeus do not add up!"


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### INITIAL DATA EXPLORATION
##get a sense of data completeness
#create dict of all the keys, get a count of how many have values
keys = {'_TOTAL_ITEMS' : len(my_dataset)}
for key in my_dataset['YEAP SOON'].iterkeys():
	keys[key] = 0

for k, v in my_dataset.iteritems():
	for key, val in v.iteritems():
		if key == 'poi':
			if val == True:
				keys[key] += 1
		elif val != 'NaN':
			keys[key] += 1
#print how many of each feature have values
pprint.pprint(keys)
#print the percent of all the items that have a value for that feature
for key, val in keys.iteritems():
	print key, ":", round(float(val) / keys["_TOTAL_ITEMS"],3)

#create a new feature "pct_stock_value_excercised" to see how much of the total value of their stock they excercised
for key, val in my_dataset.iteritems():
	if val['exercised_stock_options'] != 'NaN' and val['total_stock_value'] != 'NaN':
		val['pct_stock_value_excercised'] = float(val['exercised_stock_options'])/val['total_stock_value']
	else:
		val['pct_stock_value_excercised'] = 'NaN'

#create a new features "to_and_from_poi" and "pct_to_and_from_poi"
#sum of messages exchanged with a poi and percent of messages exchanged with poi
for key, val in my_dataset.iteritems():
	from_poi = val['from_poi_to_this_person']
	to_poi = val['from_this_person_to_poi']
	from_total = val['from_messages']
	to_total = val['to_messages']
	#if the value is missing make it 0 (makes math simpler later)
	if from_poi == "NaN":
		from_poi = 0
	if to_poi == "NaN":
		to_poi = 0
	if from_total == "NaN":
		from_total = 0
	if to_total == "NaN":
		to_total = 0

	val['to_and_from_poi'] = from_poi + to_poi

	if val['to_and_from_poi'] == 0:
		val['pct_to_and_from_poi'] = 0
	else:
		val['pct_to_and_from_poi'] = val['to_and_from_poi'] / (from_total + to_total)
#pprint.pprint(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
"""from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi']
Accuracy: 0.84764	Precision: 0.44633	Recall: 0.27650
"""

#try using SVMs with grid search to optimize the parameters
"""from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000.0)

param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

as is, this model had .9 accuracy, but 0 recall and precision
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised']
"""

#try a decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=0, max_depth = 7)

"""
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.81521	Precision: 0.34446	Recall: 0.32500	F1: 0.33445	F2: 0.32871
	Total predictions: 14000	True positives:  650	False positives: 1237	False negatives: 1350	True negatives: 10763
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.81307	Precision: 0.33529	Recall: 0.31400	F1: 0.32430	F2: 0.31804
	Total predictions: 14000	True positives:  628	False positives: 1245	False negatives: 1372	True negatives: 10755
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'shared_receipt_with_poi'] 

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.80350	Precision: 0.29334	Recall: 0.26650	F1: 0.27928	F2: 0.27147
	Total predictions: 14000	True positives:  533	False positives: 1284	False negatives: 1467	True negatives: 10716
features_list = ['poi','salary', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.82050	Precision: 0.36920	Recall: 0.36200	F1: 0.36556	F2: 0.36342
	Total predictions: 14000	True positives:  724	False positives: 1237	False negatives: 1276	True negatives: 10763
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi', 'from_poi_to_this_person','from_this_person_to_poi']

***BEST PERFORMANCE YET***
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.82414	Precision: 0.37209	Recall: 0.33600	F1: 0.35313	F2: 0.34265
	Total predictions: 14000	True positives:  672	False positives: 1134	False negatives: 1328	True negatives: 10866
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi', 'from_poi_to_this_person','from_this_person_to_poi']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.81507	Precision: 0.34589	Recall: 0.33050	F1: 0.33802	F2: 0.33347
	Total predictions: 14000	True positives:  661	False positives: 1250	False negatives: 1339	True negatives: 10750
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi','to_and_from_poi']


DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
	Accuracy: 0.80843	Precision: 0.32459	Recall: 0.31550	F1: 0.31998	F2: 0.31728
	Total predictions: 14000	True positives:  631	False positives: 1313	False negatives: 1369	True negatives: 10687
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi','to_and_from_poi']
"""

#try gradient boosting classifier
"""from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
param_grid = {
         'n_estimators': [50, 100, 200, 300, 1000]
          }
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), param_grid)

GradientBoostingClassifier(init=None, learning_rate=1.0, loss='deviance',
              max_depth=1, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              presort='auto', random_state=0, subsample=1.0, verbose=0,
              warm_start=False)
	Accuracy: 0.81100	Precision: 0.27997	Recall: 0.20550	F1: 0.23702	F2: 0.21705
	Total predictions: 14000	True positives:  411	False positives: 1057	False negatives: 1589	True negatives: 10943
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi', 'from_poi_to_this_person','from_this_person_to_poi'] 
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#new evaluation technique using stratified shuffle split
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np

#list of feature lists to test for decision tree
features_list_list = [['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi'],
	['poi','salary', 'exercised_stock_options', 'total_stock_value', 'shared_receipt_with_poi'],
	['poi','salary', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi'],
	['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi', 'from_poi_to_this_person','from_this_person_to_poi'],
	['poi','salary', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi','to_and_from_poi'],
	['poi', 'exercised_stock_options', 'total_stock_value', 'pct_stock_value_excercised', 'shared_receipt_with_poi','to_and_from_poi'],
	['poi', 'total_stock_value', 'total_payments'],
	['poi','salary', 'exercised_stock_options', 'pct_stock_value_excercised', 'shared_receipt_with_poi']]

best_list_score = 0
for features_list in features_list_list:
	print "\n New feature list for decision tree:", features_list
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	#run a stratified shuffle split
	sss = StratifiedShuffleSplit(labels, n_iter = 1000, random_state = 42)

	#find best features
	features_scores = {}
	evaluation_metrics = {"accuracy" : [], "precision" : [], "recall" : []}

	for i_train, i_test in sss:
		features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
		labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]

		#fit selector to training set
		selector = tree.DecisionTreeClassifier(random_state=0, max_depth=7)
		selector.fit(features_train, labels_train)

		pred = selector.predict(features_test)

		#evaluation metrics
		acc = accuracy_score(pred, labels_test)
		precision = precision_score(pred, labels_test)
		recall = recall_score(pred, labels_test)
		evaluation_metrics['accuracy'].append(acc)
		evaluation_metrics['precision'].append(precision)
		evaluation_metrics['recall'].append(recall)

		#get scores for each feature
		sel_features = features_list[1:]
		sel_list = []
		for i in range(len(sel_features)):
			sel_list.append([sel_features[i], selector.feature_importances_[i]])

		#fill features_scores dict
		for feat, score in sel_list:
			if feat not in features_scores:
				features_scores[feat] = {"scores" : []}
			features_scores[feat]['scores'].append(score)

	#avg scores of each feature
	features_scores_1 = [] 	#feature name, avg scores, avg pvals
	for feat in features_scores:
		features_scores_1.append((feat, np.mean(features_scores[feat]['scores'])))

	#avg and print evaluation metrics
	evaluation_metrics_avg = {}
	for key, val in evaluation_metrics.iteritems():
		print key, np.mean(val)
		evaluation_metrics_avg[key] = np.mean(val)

	#sort by scores and print
	import operator

	sorted_feature_scores = sorted(features_scores_1, key=operator.itemgetter(1), reverse=True)
	sorted_feature_scores_str = ["{}, {}".format(x[0], x[1]) for x in sorted_feature_scores]
	print "feature, importance score"
	pprint.pprint(sorted_feature_scores_str)

	#see how the model compares to the others (highest precision + recall)
	if evaluation_metrics_avg['precision'] + evaluation_metrics_avg['recall'] > best_list_score:
		best_list_score = evaluation_metrics_avg['precision'] + evaluation_metrics_avg['recall']
		best_evaluation_metrics = evaluation_metrics_avg
		best_sorted_feature_scores_str = sorted_feature_scores_str
		best_features_list = features_list
		print "***THIS IS THE NEW BEST MODEL***"

#return values for the best model
print "\n THE WINNER IS..."
print "features_list :", best_features_list
pprint.pprint(best_evaluation_metrics)
print "feature, importance score"
pprint.pprint(best_sorted_feature_scores_str)

#change the features list to the one that performed the best so tester.py works correctly
features_list = best_features_list
feature_names = features_list[1:]

#get an idea about what the tree looks like
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "\nBasic validation with test train split (For Tree PDF)"
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=7)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "Feature Importances:"
for i in range(len(clf.feature_importances_)):
	print feature_names[i], clf.feature_importances_[i]

#since the Decision tree has the best results, lets visualize them
with open("EnronDecisionTree.dot", 'w') as f:
	f = tree.export_graphviz(clf, out_file=f, 
		feature_names=feature_names,
		filled=True, rounded=True)
	#this .dot file can be converted into a pdf by:
	#	install graphviz via homebrew: brew install graphviz
	#	convert to pdf in terminal: dot -Tpdf EnronDecisionTree.dot -o EnronDecisionTree.pdf


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)