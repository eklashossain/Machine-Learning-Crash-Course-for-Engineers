# ----------------------Importing Modules------------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout


# --------------------Loading CIFAR-10 Dataset-------------------
(xtrain,ytrain),(xtest,ytest)= keras.datasets.cifar10.load_data()

#Preprocessing data
xtrain = xtrain/255
xtest = xtest/255
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)


# -----------------------Creating CNN Model----------------------
model=models.Sequential()
model.add(layers.Conv2D(64,(3,3),input_shape=(32,32,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Flatten(input_shape=(32,32)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Train model
xtrain2=xtrain.reshape(50000,32,32,3)
xtest2=xtest.reshape(10000,32,32,3)
model.fit(xtrain2,ytrain,epochs=40,batch_size=56,verbose=True,validation_data=(xtest2,ytest))


# ---------------------------Evaluation--------------------------
test_loss, test_acc = model.evaluate(xtest2, ytest)
print("accuracy:", test_acc)

#Visualising the output
predictions=model.predict(xtest2)
class_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def visualize_output(predicted_label, true_label, img):
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label_index =np.argmax(predicted_label)
  true_label_index=np.argmax(true_label)

  if predicted_label_index == true_label_index:
    color='blue' # accurate prediction
  else:
    color='red' # inaccurate prediction

  plt.xlabel("{} {:2.0f}% ({})".format(class_labels[predicted_label_index], 100*np.max(predicted_label), class_labels[true_label_index]), color=color)

plt.figure(figsize=(12,6))
visualize_output(predictions[1], ytest[1], xtest[1])

plt.show()