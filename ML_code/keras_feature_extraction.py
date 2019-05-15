import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from sklearn.model_selection import train_test_split

#8 types
fnames_tr = pd.read_csv("small_train_2.csv")
fnames_va = pd.read_csv("small_val_2.csv")
print(len(fnames_tr))
#39976 train data
print(len(fnames_va))
#17128 validation data

#4 types
#fnames_tr_4 = fnames_tr[:19862,]
#fnames_va_4 = fnames_va[:6415,]

#2 types
#fnames_tr_2 = fnames_tr[:10720,]
#fnames_va_2 = fnames_va[:2573,]

a = len(fnames_tr)
b = len(fnames_va)


data_gen = ImageDataGenerator(rotation_range = 30,
                               rescale=1./255,
                               zoom_range=0.2,
                               shear_range=0.1,
                               horizontal_flip=True,
                               vertical_flip=True)  
                               
train_gen = data_gen.flow_from_dataframe(fnames_tr,  # fnames_tr_2, #fnames_tr_4,
                                         directory=None, 
                                         x_col='fname', 
                                         y_col='label', 
                                         target_size  = (150,150), 
                                         batch_size=32, 
                                         seed=66,  
                                         class_mode="categorical")

va_gen = data_gen.flow_from_dataframe(fnames_va, #fnames_va_2, #fnames_va_4,
                                      directory=None, 
                                      x_col='fname', 
                                      y_col='label',
                                      target_size=(150,150), 
                                      batch_size=32,
                                      seed=66,
                                      class_mode="categorical")  

#romeve the fully connected layer of vgg16 to extract features                                   
vgg = vgg16.VGG16(weights='imagenet',
                            input_shape=(150,150,3),  #image size is 150*150*3 
                            include_top=False) 

#feature extraction and generate X,y
def feature_extract(model, generator, data_count, class_count, feature_shape):
    featureshape = []
    for i in feature_shape:
        featureshape.append(i)
    featureshape.insert(0,data_count)    
    X = np.zeros(shape=tuple(featureshape))
    y = np.zeros(shape=(data_count,class_count))
    gen = generator
    batsize = 32
    j = 0
    for input_batch,label_batch in gen:    
        featurebat = model.predict(input_batch)
        X[j * batsize : (j + 1) * batsize] = featurebat
        y[j * batsize : (j + 1) * batsize] = label_batch
        j += 1
        if j * batsize >= data_count:
            break
    return X,y                                                            
                             
#save 8 styles train data
X_train_8, y_train_8 = feature_extract(vgg,train_gen,39976,8,(4,4,512))
np.savetxt("keras_X_tr_8.txt", X_train_8.reshape((4,-1)))
np.savetxt("keras_Y_tr_8.txt", y_train_8)

#save 4 style train data
#X_train_4, y_train_4 = feature_extract(vgg, train_gen, a, 4, (4,4,512))
#np.savetxt("keras_X_tr_4.txt", X_train_4.reshape((4,-1)))
#np.savetxt("keras_Y_tr_4.txt", y_train_4)

#save 2 style train data
#X_train_2, y_train_2 = feature_extract(vgg, train_gen, a, 2, (4,4,512))
#np.savetxt("keras_X_tr_2.txt", X_train_2.reshape((4,-1)))
#np.savetxt("keras_Y_tr_2.txt", y_train_2)


X_va, y_va = feature_extract(vgg, va_gen, b, 8, (4,4,512))
#seperate the validation set into validation set and test set (0.35)
X_va_8,X_test_8,y_va_8,y_test_8 = train_test_split(X_va,y_va,test_size = 0.35, random_state = 66)

c = len(y_test_8)
print(c)
#5995 test data
np.savetxt("keras_X_va_8.txt", X_va_8.reshape((4,-1)))
np.savetxt("keras_Y_va_8.txt", y_va_8)
np.savetxt("keras_X_te_8.txt", X_test_8.reshape((4,-1)))
np.savetxt("keras_Y_te_8.txt", y_test_8)

#save 4 style validation and test data
#X_va, y_va = feature_extract(vgg, va_gen, b, 4, (4,4,512))
#X_va_4,X_test_4,y_va_4,y_test_4 = train_test_split(X_va,y_va,test_size = 0.35, random_state = 66)
#np.savetxt("keras_X_va_4.txt", X_va_4.reshape((4,-1)))
#np.savetxt("keras_Y_va_4.txt", y_va_4)
#np.savetxt("keras_X_te_4.txt", X_test_4.reshape((4,-1)))
#np.savetxt("keras_Y_te_4.txt", y_test_4)

#save 2 style validation and test data
#X_va, y_va = feature_extract(vgg, va_gen, b, 2, (4,4,512))
#X_va_2,X_test_2,y_va_2,y_test_2 = train_test_split(X_va,y_va,test_size = 0.35, random_state = 66)
#np.savetxt("keras_X_va_2.txt", X_va_2.reshape((4,-1)))
#np.savetxt("keras_Y_va_2.txt", y_va_2)
#np.savetxt("keras_X_te_2.txt", X_test_2.reshape((4,-1)))
#np.savetxt("keras_Y_te_2.txt", y_test_2)

print("Data has been saved")
                        

