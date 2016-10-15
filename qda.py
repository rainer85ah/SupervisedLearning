""" Quadratic Discriminant Analysis (QDA)
A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using
Bayes rule.
The model fits a Gaussian density to each class.
"""
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets, metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split

cancer = datasets.load_breast_cancer()
data = cancer.data
labels = cancer.target

data = np.asarray(data, dtype='float32')
labels = np.asarray(labels, dtype='int32')
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

print('QDA Learning... Fitting... ')
qda_clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0)
qda_clf.fit(X=trainData, y=trainLabels)

print('QDA Predicting... ')
predicted = qda_clf.predict(X=testData)

print("Results: \n %s" % metrics.classification_report(testLabels, predicted))
matrix = metrics.confusion_matrix(testLabels, predicted)
print("Confusion Matrix: \n %s" % matrix)
print("Mean Accuracy Mean: %.4f " % qda_clf.score(X=testData, y=testLabels))

print("QDA Saving in ... /Output/QDA_model.pkl")
joblib.dump(qda_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/QDA_model.pkl')