from sklearn import svm
import numpy as np

X_tr = np.loadtxt("keras_X_tr_8.txt")
y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
X_test = np.loadtxt("keras_X_te_8.txt")
y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)


clf = svm.SVC()
clf.fit(X_tr, y_tr)

y_p_test = clf.predict(X_test)

print("SVM test accuracy: {}".format(np.mean(y_p_test==y_test)))

