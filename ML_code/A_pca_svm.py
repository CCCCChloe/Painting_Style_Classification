from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

X_tr = np.loadtxt("keras_X_tr_8.txt")
y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
X_test = np.loadtxt("keras_X_te_8.txt")
y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)


pca = PCA(n_components=50,svd_solver='randomized')
pca.fit(X_tr)
X_tr_reduced = pca.transform(X_tr)
X_test_reduced = pca.transform(X_test)

clf = SVC()
clf.fit(X_tr_reduced, y_tr)
y_p_test = clf.predict(X_test_reduced)

print("PCA SVM test accuracy: {}".format(np.mean(y_p_test==y_test)))

mat = confusion_matrix(y_p_test, y_test)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true style')
plt.ylabel('predicted style')