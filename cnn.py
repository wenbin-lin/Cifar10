import keras.backend as K
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization
from keras.layers import Flatten, Activation, Dropout
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import utils

def getModel(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     kernel_initializer='glorot_uniform', name='conv0'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, name='max_pool0'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='glorot_uniform', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, name='max_pool1'))
    model.add(Flatten())
    model.add(Dense(units=128, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    activation='relu', name='fc0'))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    activation='softmax', name='fc1'))
    return model

if __name__ == "__main__":
    print(K.image_data_format())
    x_train, y_train, x_test, y_test = utils.load_data()
    num_train = 10000
    num_test = 2048
    x_train = x_train[0:num_train, :]
    y_train = y_train[0:num_train]
    x_test = x_test[0:num_test, :]
    y_test = y_test[0:num_test]
    images_train = x_train / 255
    images_test = x_train / 255
    y_train = utils.convert_to_one_hot(y_train, 10)
    y_test = utils.convert_to_one_hot(y_test, 10)
    '''img0 = x_train[100, :, :, :]
    print(img0.shape)
    print(y_train[100])
    plt.imshow(img0)
    plt.show()'''

    model = getModel(x_train.shape[1:], 10)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10, batch_size=128)
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()

    MODEL_NAME = './cifar10_org_img.h5'
    tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    ckpt = keras.callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                                           period=1)

    history = model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test),
                        callbacks=[tb, ckpt])

    (loss, acc) = model.evaluate(x_test, y_test, batch_size=32)
    print('evaluated loss:', loss, 'accuracy', acc)
