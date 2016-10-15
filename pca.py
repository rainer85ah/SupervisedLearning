"""
Supervised learning: 1-Classification*, 2-Regression.
PCA is a linear transformation techniques: PCA is unsupervised(ignores class labels)...

Principal component analysis (PCA):
Linear dimensionality reduction using Singular Value Decomposition of the data and keeping only the most significant
singular vectors to project the data to a lower dimensional space.
This implementation uses the scipy.linalg implementation of the singular value decomposition. It only works for
dense arrays and is not scalable to large dimensional data.
The time complexity of this implementation is O(n^3) assuming n ~ n_samples ~ n_features.

If n ~ n_examples ~ n_features => The time complexity of this implementation is O(n^3). :(
"""
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

cancer = datasets.load_breast_cancer()
data = cancer.data
labels = cancer.target

data = np.asarray(data, dtype='float32')
labels = np.asarray(labels, dtype='int32')
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=0.7, test_size=0.3)

# Pre-processing before PCA always! Good Manners!!
"""
1- Scaling of dataset is a common requirement for many machine learning estimators: [0, 1] and [-1, 1].
"""
min_max_scaler = preprocessing.MinMaxScaler()
trainData = min_max_scaler.fit_transform(X=trainData)
testData = min_max_scaler.transform(X=testData)

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_minmaxabs = max_abs_scaler.fit_transform(X=trainData)
X_test_minmaxabs = max_abs_scaler.transform(X=testData)

"""
2- Normalization.. is the process of scaling individual samples to have unit norm.
Note:
This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel
to quantify the similarity of any pair of samples.
"""
trainData = preprocessing.normalize(X=trainData, norm='l2')  # norm : l1, l2, or max, (l2 by default)
testData = preprocessing.normalize(X=testData, norm='l2')  # norm : l1, l2, or max, (l2 by default)

"""
3- Binarization is the process of thresholding numerical features to get boolean values. This can be useful for
   downstream probabilistic estimators that make assumption that the input data is distributed according to a
   multi-variate Bernoulli distribution.
X_train_binarizer = preprocessing.Binarizer().fit_transform(X=trainData)
"""

"""
4-Enconding..[US, Spain, Cuba] --> [0, 1, 2]

5-Missing values.. Replace with the mean, median, etc..
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imputer = imp.fit_transform(X=trainData)
"""

print('PCA Learning... Fitting... ')
pca_clf = PCA(n_components=30, copy=True, whiten=False)
pca_clf.fit(X=trainData, y=trainLabels)

print('PCA do not have Predicting... ')

print("PCA Results: ")
print("Score, average log-likelihood of all samples: %.4f " % pca_clf.score(testData, testLabels))

print("PCA Saving in ... /Output/PCA_model.pkl")
joblib.dump(pca_clf, '/home/rainer85ah/PycharmProjects/DiagnosticCancerSolution/Output/PCA_model.pkl')
