"""
Nearest Neighbors Classification is a type of instance-based learning or non-generalizing learning: it does not attempt
to construct a general internal model, but simply stores instances of the training data. Classification is computed
from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has
the most representatives within the nearest neighbors of the point.
SK-Learn have 2 different implementations of this algorithms:
1- KNeighborsClassifier
2- RadiusNeighborsClassifier
"""
from sklearn.neighbors import KNeighborsClassifier
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

print('KNN Learning... Fitting... ')
knn_clf = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                               metric='minkowski', metric_params=None, n_jobs=1)
knn_clf.fit(X=trainData, y=trainLabels)

print('KNN Predicting... ')
predicted = knn_clf.predict(X=testData)

print("Results: \n %s" % metrics.classification_report(testLabels, predicted))
matrix = metrics.confusion_matrix(testLabels, predicted)
print("Confusion Matrix: \n %s" % matrix)
print("\nMean Accuracy: %.4f " % knn_clf.score(X=testData, y=testLabels))

print("KNN Saving in ... /Output/KNN_model.pkl")
joblib.dump(knn_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/KKN_model.pkl')
