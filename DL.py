import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Activation,BatchNormalization,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# Basic EDA

print(x_train.shape)

plt.imshow(x_train[0])
plt.show()

print(y_train[0])

print(x_train.min(),x_train.max())

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.axis('off')

plt.show()

#Normalizing X

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_cat = to_categorical(y_train,num_classes=10)
y_test_cat = to_categorical(y_test,num_classes=10)

# ANN

ANN = Sequential([
    Flatten(input_shape=(32 , 32 , 3)),
    Dense(256,activation='relu'),
    Dense(10,activation='softmax')
])

ANN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = ANN.fit(x_train,y_train_cat,batch_size=32,validation_data=(x_test,y_test_cat),
                  epochs=5)

accuracy_ANN = ANN.evaluate(x_test,y_test_cat,verbose=0)[1]

print('Accuracy ANN :-' , accuracy_ANN)

# Reshaping Data for CNN
x_train_cnn = x_train.reshape(-1,32,32,3)
x_test_cnn = x_test.reshape(-1,32,32,3)

#CNN

CNN = Sequential([
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    Conv2D(32,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history_CNN = CNN.fit(x_train_cnn,y_train_cat,epochs=10,
                      validation_data=(x_test_cnn,y_test_cat),batch_size=32,verbose=1)

accuracy_CNN = CNN.evaluate(x_test_cnn,y_test_cat,verbose=0)[1]

print('Accuracy of CNN :-', accuracy_CNN)


# CNN with augmented data

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train_cnn)

train_generator = datagen.flow(x_train_cnn,y_train_cat,batch_size=32)

steps_per_epoch = len(x_train_cnn) // 32

history_CNN_aug = CNN.fit(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = 10,
    validation_data=(x_test_cnn,y_test_cat),
    verbose=1
)

accuracy_CNN_aug = CNN.evaluate(x_test_cnn,y_test_cat,verbose=0)[1]

print('Accuracy of CNN with augmented data :-', accuracy_CNN_aug)

# CNN with Batch normalization and VGG-style architecture

CNN_VGG_BN = Sequential([

    Conv2D(64,kernel_size=(3,3),padding='same',input_shape=(32,32,3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64,kernel_size=(3,3),padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Conv2D(128,kernel_size=(3,3),padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128,kernel_size=(3,3),padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),

    Conv2D(256,kernel_size=(3,3),padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256,kernel_size=(3,3),padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),

    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')

])

CNN_VGG_BN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history_CNN_VGG_BN = CNN_VGG_BN.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    validation_data=(x_test_cnn,y_test_cat),
    verbose=1
)

accuracy_CNN_VGG_BN = CNN_VGG_BN.evaluate(x_test_cnn,y_test_cat,verbose=0)[1]

print('Accuracy of CNN with Batch normalization and VGG architecture :-' ,accuracy_CNN_VGG_BN)

