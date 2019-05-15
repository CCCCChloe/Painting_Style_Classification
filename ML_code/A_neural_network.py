import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

X_tr = np.loadtxt("keras_X_tr_8.txt")
y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
X_test = np.loadtxt("keras_X_te_8.txt")
y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)


stdsc = StandardScaler()
stdsc.fit(X_tr)
X_train_std = stdsc.transform(X_tr)
X_test_std = stdsc.transform(X_test)

cur_tuple = [40] * 20  
mlp = MLPClassifier(activation='tanh', max_iter=10, hidden_layer_sizes = cur_tuple,verbose=True, random_state = 66)
mlp.fit(X_train_std, y_tr)
test_acc = mlp.score(X_test_std,y_test)
print("CNN test accuracy: {}".format(test_acc))