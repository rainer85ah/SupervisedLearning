""" Support Vector Machines:
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and
outliers detection.

The advantages of support vector machines are:

1-Effective in high dimensional spaces.
2-Still effective in cases where number of dimensions is greater than the number of samples.
3-Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
4-Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided,
  but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

1-If the number of features is much greater than the number of samples, the method is likely to give poor performances.
2-SVMs do not directly provide probability estimates, these are calculated using an expensive
  five-fold cross-validation (see Scores and probabilities, below).


The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray)
and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data,
it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or
scipy.sparse.csr_matrix (sparse) with dtype=float64.

OJO.. Classification:
SVC, NuSVC and LinearSVC are classes capable of performing multi-class classification on a dataset.
"""
from sklearn import svm, datasets, metrics
from sklearn.externals import joblib
import numpy as np
from sklearn.cross_validation import train_test_split

cancer = datasets.load_breast_cancer()
data = cancer.data
labels = cancer.target

data = np.asarray(data, dtype='float32')
labels = np.asarray(labels, dtype='int32')
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

"""
SVM
"""
print('SVM Learning... Fitting... ')
# Linear Support Vector Classification...
svm_clf = svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto', coef0=0.015, shrinking=True, probability=False,
                  tol=0.001, cache_size=512, class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape=None, random_state=None)

svm_clf.fit(X=trainData, y=trainLabels)

print('SVM Predicting... ')
predicted = svm_clf.predict(X=testData)

print("Results: \n %s" % metrics.classification_report(testLabels, predicted))
matrix = metrics.confusion_matrix(testLabels, predicted)
print("Confusion Matrix: \n %s" % matrix)
print("Score Accuracy Mean: %.4f " % svm_clf.score(X=testData, y=testLabels))

print("SVM Saving in ... /Output/SVM_model.pkl")
joblib.dump(svm_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/SVM_model.pkl')
