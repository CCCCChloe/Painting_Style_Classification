import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn import metrics
import matplotlib.pyplot as plt

##input 8 types data
X_tr = np.loadtxt("keras_X_tr_8.txt")
y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
X_test = np.loadtxt("keras_X_te_8.txt")
y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)

##input 4 types data
#X_tr = np.loadtxt("keras_X_tr_8.txt")
#y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
#X_test = np.loadtxt("keras_X_te_8.txt")
#y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)

##input 2 types data
#X_tr = np.loadtxt("keras_X_tr_8.txt")
#y_tr = np.loadtxt("keras_Y_tr_8.txt", dtype=int)
#X_test = np.loadtxt("keras_X_te_8.txt")
#y_test = np.loadtxt("keras_Y_te_8.txt.txt", dtype=int)

#add fully connected layer 
vgg_model = Sequential()
vgg_model.add(Dense(512,activation="relu",input_dim=X_tr.shape[1],name = "layer_1")) 
vgg_model.add(Dropout(0.5))
vgg_model.add(Dense(256,activation="relu",name = "layer_2"))  
vgg_model.add(Dropout(0.25))
vgg_model.add(Dense(8,activation="softmax",name = "layer_3")) 
#vgg_model.add(Dense(4,activation="softmax",name = "layer_3")) 
#vgg_model.add(Dense(2,activation="softmax",name = "layer_3")) 

vgg_model.compile(SGD(lr=0.01),loss="categorical_crossentropy", metrics=["accuracy"])

#fit model
vgg_model_fit = vgg_model.fit(X_tr,y_tr,batch_size=32,validation_split=0.1, epochs=100)
                                       
#make prediction
y_prediction = vgg_model.predict(X_test)
y_pred = []
for i in y_prediction:
    i_arg = i.argmax()
    y_pred.append(i_arg)
    
y_true = []
for j in y_test:
    j_arg = j.argmax()
    y_true.append(j_arg)

#confusion matrix
con_mat = metrics.confusion_matrix(y_true,y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(con_mat.T, cmap="GnBu",square=True,center=0,annot=True,fmt="d")
plt.xlabel('True style',size = 12)
plt.ylabel('Predicted style',size = 12)
plt.title("Confusion matrix",size = 16)
#save image
plt.savefig('prediction.png')
plt.show()

#print accuracy
accuracy = metrics.accuracy_score(y_true,y_pred)
print("Accuracy:",accuracy)  


vgg_df = pd.DataFrame(vgg_model_fit.history)
#model accuracy visualization
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(vgg_df.epochs,vgg_df.acc,"r-",label = "train accuracy")
plt.plot(vgg_df.epochs,vgg_df.val_acc,"b--",label = "validation accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")

#model loss visualization
plt.subplot(1,2,2)
plt.plot(vgg_df.epochs,vgg_df.loss,"r-",label = "train loss")
plt.plot(vgg_df.epochs,vgg_df.val_loss,"b--",label = "validation loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")

#save image
plt.savefig('vgg_model.png')
plt.show()