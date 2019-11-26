# cifar10-train

### Base network validaton score
83.00

### My network score
83.06

### model definition
model = Sequential()

model.add(SeparableConv2D(48, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3),border_mode='same')) #32 3X3
model.add(BatchNormalization())

model.add(SeparableConv2D(48, 3, 3,border_mode='same')) #32 5X5
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(48, 3, 3,border_mode='same')) #32 7X7
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(48, 3, 3)) #30 9X9
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) #13 11X11
model.add(BatchNormalization())


model.add(SeparableConv2D(96, 3, 3, border_mode='same')) #13 15X15
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(96, 3, 3, border_mode='same')) #13 19X19
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(SeparableConv2D(96, 3, 3, border_mode='same')) #13 23X23
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(96, 3, 3)) #9 27X27
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2))) #4 31X31
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(192, 3, 3, border_mode='same')) #4 39X39
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(Convolution2D(10,1))#4X4X10 39X39
model.add(AveragePooling2D(4))#1X1X10
model.add(Flatten())
model.add(Activation('softmax'))

### 50 epoch logs

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
390/390 [==============================] - 22s 58ms/step - loss: 1.4395 - acc: 0.4767 - val_loss: 1.4035 - val_acc: 0.5290
Epoch 2/50
390/390 [==============================] - 19s 49ms/step - loss: 1.0597 - acc: 0.6280 - val_loss: 1.0319 - val_acc: 0.6365
Epoch 3/50
390/390 [==============================] - 19s 49ms/step - loss: 0.8873 - acc: 0.6873 - val_loss: 0.8565 - val_acc: 0.6950
Epoch 4/50
390/390 [==============================] - 19s 49ms/step - loss: 0.7710 - acc: 0.7295 - val_loss: 0.7580 - val_acc: 0.7338
Epoch 5/50
390/390 [==============================] - 19s 49ms/step - loss: 0.6925 - acc: 0.7579 - val_loss: 0.7690 - val_acc: 0.7355
Epoch 6/50
390/390 [==============================] - 19s 49ms/step - loss: 0.6373 - acc: 0.7790 - val_loss: 0.6998 - val_acc: 0.7562
Epoch 7/50
390/390 [==============================] - 19s 49ms/step - loss: 0.5963 - acc: 0.7930 - val_loss: 0.6602 - val_acc: 0.7725
Epoch 8/50
390/390 [==============================] - 19s 49ms/step - loss: 0.5522 - acc: 0.8073 - val_loss: 0.6687 - val_acc: 0.7717
Epoch 9/50
390/390 [==============================] - 19s 49ms/step - loss: 0.5219 - acc: 0.8191 - val_loss: 0.6072 - val_acc: 0.7912
Epoch 10/50
390/390 [==============================] - 19s 49ms/step - loss: 0.4957 - acc: 0.8279 - val_loss: 0.6586 - val_acc: 0.7768
Epoch 11/50
390/390 [==============================] - 19s 50ms/step - loss: 0.4716 - acc: 0.8357 - val_loss: 0.5914 - val_acc: 0.7990
Epoch 12/50
390/390 [==============================] - 19s 49ms/step - loss: 0.4513 - acc: 0.8429 - val_loss: 0.5842 - val_acc: 0.8027
Epoch 13/50
390/390 [==============================] - 19s 49ms/step - loss: 0.4316 - acc: 0.8481 - val_loss: 0.5776 - val_acc: 0.8074
Epoch 14/50
390/390 [==============================] - 19s 49ms/step - loss: 0.4125 - acc: 0.8553 - val_loss: 0.5746 - val_acc: 0.8088
Epoch 15/50
390/390 [==============================] - 19s 50ms/step - loss: 0.3919 - acc: 0.8630 - val_loss: 0.5985 - val_acc: 0.8043
Epoch 16/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3852 - acc: 0.8653 - val_loss: 0.6014 - val_acc: 0.8055
Epoch 17/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3676 - acc: 0.8716 - val_loss: 0.5647 - val_acc: 0.8162
Epoch 18/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3572 - acc: 0.8740 - val_loss: 0.5657 - val_acc: 0.8168
Epoch 19/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3447 - acc: 0.8775 - val_loss: 0.5588 - val_acc: 0.8156
Epoch 20/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3308 - acc: 0.8838 - val_loss: 0.5593 - val_acc: 0.8151
Epoch 21/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3218 - acc: 0.8869 - val_loss: 0.5594 - val_acc: 0.8153
Epoch 22/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3130 - acc: 0.8897 - val_loss: 0.5583 - val_acc: 0.8225
Epoch 23/50
390/390 [==============================] - 19s 49ms/step - loss: 0.3048 - acc: 0.8906 - val_loss: 0.5589 - val_acc: 0.8247
Epoch 24/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2942 - acc: 0.8955 - val_loss: 0.5809 - val_acc: 0.8215
Epoch 25/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2937 - acc: 0.8960 - val_loss: 0.5643 - val_acc: 0.8257
Epoch 26/50
390/390 [==============================] - 19s 50ms/step - loss: 0.2803 - acc: 0.9010 - val_loss: 0.5528 - val_acc: 0.8250
Epoch 27/50
390/390 [==============================] - 20s 50ms/step - loss: 0.2736 - acc: 0.9021 - val_loss: 0.5760 - val_acc: 0.8266
Epoch 28/50
390/390 [==============================] - 20s 51ms/step - loss: 0.2680 - acc: 0.9049 - val_loss: 0.5690 - val_acc: 0.8264
Epoch 29/50
390/390 [==============================] - 19s 50ms/step - loss: 0.2617 - acc: 0.9078 - val_loss: 0.5668 - val_acc: 0.8284
Epoch 30/50
390/390 [==============================] - 19s 50ms/step - loss: 0.2527 - acc: 0.9105 - val_loss: 0.5560 - val_acc: 0.8322
Epoch 31/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2508 - acc: 0.9105 - val_loss: 0.5793 - val_acc: 0.8210
Epoch 32/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2423 - acc: 0.9136 - val_loss: 0.5893 - val_acc: 0.8224
Epoch 33/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2382 - acc: 0.9144 - val_loss: 0.6105 - val_acc: 0.8215
Epoch 34/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2353 - acc: 0.9141 - val_loss: 0.5756 - val_acc: 0.8285
Epoch 35/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2306 - acc: 0.9174 - val_loss: 0.5825 - val_acc: 0.8263
Epoch 36/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2221 - acc: 0.9196 - val_loss: 0.6017 - val_acc: 0.8277
Epoch 37/50
390/390 [==============================] - 19s 50ms/step - loss: 0.2177 - acc: 0.9214 - val_loss: 0.5978 - val_acc: 0.8285
Epoch 38/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2127 - acc: 0.9224 - val_loss: 0.6252 - val_acc: 0.8200
Epoch 39/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2140 - acc: 0.9228 - val_loss: 0.6235 - val_acc: 0.8227
Epoch 40/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2081 - acc: 0.9239 - val_loss: 0.5964 - val_acc: 0.8253
Epoch 41/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2048 - acc: 0.9255 - val_loss: 0.6209 - val_acc: 0.8270
Epoch 42/50
390/390 [==============================] - 19s 49ms/step - loss: 0.2012 - acc: 0.9278 - val_loss: 0.5786 - val_acc: 0.8314
Epoch 43/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1982 - acc: 0.9283 - val_loss: 0.6272 - val_acc: 0.8216
Epoch 44/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1916 - acc: 0.9309 - val_loss: 0.5861 - val_acc: 0.8362
Epoch 45/50
390/390 [==============================] - 19s 50ms/step - loss: 0.1920 - acc: 0.9298 - val_loss: 0.5980 - val_acc: 0.8304
Epoch 46/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1842 - acc: 0.9340 - val_loss: 0.6118 - val_acc: 0.8301
Epoch 47/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1875 - acc: 0.9326 - val_loss: 0.6012 - val_acc: 0.8312
Epoch 48/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1813 - acc: 0.9348 - val_loss: 0.6284 - val_acc: 0.8346
Epoch 49/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1790 - acc: 0.9359 - val_loss: 0.6183 - val_acc: 0.8312
Epoch 50/50
390/390 [==============================] - 19s 49ms/step - loss: 0.1755 - acc: 0.9361 - val_loss: 0.6357 - val_acc: 0.8306
Model took 966.31 seconds to train

Accuracy on test data is: 83.06
