""" Decision Trees - Supervised learning: 1-Classification*, 2-Regression.
D.T.s are a non-parametric supervised learning method used for classification and regression. The goal is to create a
model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

Some advantages of decision trees are:
1- Simple to understand and to interpret. Trees can be visualised.
2- Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be
created and blank values to be removed. Note however that this module does not support missing values.
3- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
4- Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets
that have only one type of variable. See algorithms for more information.
5- Able to handle multi-output problems.
6- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily
explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be
more difficult to interpret.
7- Possible to validate a model using statistical tests. That makes it possible to account for the reliability
of the model.
8- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.


The disadvantages of decision trees include:
1- Decision-tree learners can create over-complex trees that do not generalise the data well.This is called overfitting.
Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or
setting the maximum depth of the tree are necessary to avoid this problem.
2- Decision trees can be unstable because small variations in the data might result in a completely different tree
being generated. This problem is mitigated by using decision trees within an ensemble.
3- The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and
even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms
such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to
return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner,
where the features and samples are randomly sampled with replacement.
4- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or
multiplexer problems.
5- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the
dataset prior to fitting with the decision tree.

ID3 (Iterative Dichotomiser 3) was developed in 1986 by Ross Quinlan. The algorithm creates a multiway tree, finding
for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for
categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the
ability of the tree to generalise to unseen data.

C4.5 is the successor to ID3 and removed the restriction that features must be categorical by dynamically defining a
discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set
of intervals. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules.
These accuracy of each rule is then evaluated to determine the order in which they should be applied.
Pruning is done by removing a rule's precondition if the accuracy of the rule improves without it.
"""
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets, metrics, tree
from sklearn.cross_validation import train_test_split

cancer = datasets.load_breast_cancer()
data = cancer.data
labels = cancer.target

data = np.asarray(data, dtype='float32')
labels = np.asarray(labels, dtype='int32')
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

print('Tree Learning... Fitting... ')
tree_clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                       random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
tree_clf.fit(X=trainData, y=trainLabels)

print('Tree Predicting... ')
predicted = tree_clf.predict(X=testData)

print("Results: \n %s" % metrics.classification_report(testLabels, predicted))
matrix = metrics.confusion_matrix(testLabels, predicted)
print("Confusion Matrix: \n %s" % matrix)
print("\nMean Accuracy: %.4f " % tree_clf.score(X=testData, y=testLabels))

print("Tree Saving in ... /Output/Tree_model.pkl")
joblib.dump(tree_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/Tree_model.pkl')
