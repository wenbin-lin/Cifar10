from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization
from keras.layers import Flatten, Activation, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
import numpy as np
import utils

def get_model(input_shape, num_classes):
    weight_decay = 0.0001
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
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
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    tic = time.time()
    history = model.fit(x=x_train, y=y_train, epochs=30, batch_size=256, validation_data=(x_test, y_test),
                        callbacks=[TensorBoard(log_dir='./logs')])
    toc = time.time()
    print("train time: " + str(1000 * (toc - tic)) + "ms")
    utils.plot_history(history)

    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()
    model.save('model_cnn_v2.h5')

    y_prob = model.predict(x_test)
    y_pred = [x for x in np.argmax(y_prob, axis=1)]
    y_truth = [x for x in np.argmax(y_test, axis=1)]
    print(confusion_matrix(y_truth, y_pred))
    print(classification_report(y_truth, y_pred, target_names=utils.label_names))
