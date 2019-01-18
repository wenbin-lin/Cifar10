import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization
from keras.layers import Flatten, Activation, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import utils

def get_model(input_shape, num_classes):
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    print(K.image_data_format())
    x_train, y_train, x_test, y_test = utils.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    num_train = 50000
    num_test = 10000
    x_train = x_train[0:num_train, :]
    y_train = y_train[0:num_train]
    x_test = x_test[0:num_test, :]
    y_test = y_test[0:num_test]
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    y_train = utils.convert_to_one_hot(y_train, 10)
    y_test = utils.convert_to_one_hot(y_test, 10)

    model = get_model(x_train.shape[1:], 10)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()

    predicted_prob = model.predict(x_train[0:5])
    predicted_classes = [utils.label_names[x] for x in np.argmax(predicted_prob, axis=1)]
    ground_truth_classes = [utils.label_names[x] for x in np.argmax(y_train[0:5], axis=1)]
    print(predicted_classes, ground_truth_classes)
