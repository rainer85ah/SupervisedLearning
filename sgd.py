""" Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of
linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.

The advantages of Stochastic Gradient Descent are:
1- Efficiency.
2- Ease of implementation (lots of opportunities for code tuning).

The disadvantages of Stochastic Gradient Descent include:
1- SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
2- SGD is sensitive to feature scaling.
"""
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
import numpy as np

cancer = datasets.load_breast_cancer()
data = cancer.data
labels = cancer.target

data = np.asarray(data, dtype='float32')
labels = np.asarray(labels, dtype='int32')
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

print('SGD Learning... Fitting... ')
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5,
                        shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal',
                        eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)
sgd_clf.fit(X=trainData, y=trainLabels)

print('SGD Predicting... ')
predicted = sgd_clf.predict(X=testData)

print("Results: \n %s" % metrics.classification_report(testLabels, predicted))
matrix = metrics.confusion_matrix(testLabels, predicted)
print("Confusion Matrix: \n %s" % matrix)
print("\nMean Accuracy: %.4f " % sgd_clf.score(X=testData, y=testLabels))

print("SGD Saving in ... /Output/SGD_model.pkl")
joblib.dump(sgd_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/SGD_model.pkl')
