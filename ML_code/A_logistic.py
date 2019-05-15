from sklearn.linear_model import LogisticRegression
import numpy as np

X_tr = np.loadtxt("keras_X_tr_8.txt")
y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
X_test = np.loadtxt("keras_X_te_8.txt")
y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)

logreg = LogisticRegression(C=1e5)
logreg.fit(X_tr, y_tr)
y_p_test = logreg.predict(X_test)
test_acc = np.mean(y_p_test==y_test)
print("Logistic test accuracy: {}".format(test_acc))
