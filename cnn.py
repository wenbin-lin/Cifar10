import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization
from keras.layers import Flatten, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import utils

def get_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='tanh', kernel_initializer='glorot_uniform', name='conv0'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, name='max_pool0'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='tanh', kernel_initializer='glorot_uniform', name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, name='max_pool1'))
    model.add(Flatten())
    model.add(Dense(units=256, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    activation='tanh', name='fc0'))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    activation='softmax', name='fc1'))
    return model

from sklearn.metrics import roc_curve, auc
def sklearnAUC(test_labels,test_prediction):
    n_classes = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return round(roc_auc[0],3) , round(roc_auc[1],3)

if __name__ == "__main__":
    print(K.image_data_format())
    x_train, y_train, x_test, y_test = utils.load_data()
    num_train = 10000
    num_test = 2000
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
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', mean_pred])
    model.fit(x=x_train, y=y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test),
              callbacks=[utils.roc_callback(training_data=(x_train, y_train), validation_data=(x_test, y_test))])
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()

    predicted_prob = model.predict(x_train[0:5])
    predicted_classes = [utils.label_names[x] for x in np.argmax(predicted_prob, axis=1)]
    ground_truth_classes = [utils.label_names[x] for x in np.argmax(y_train[0:5], axis=1)]
    print(predicted_classes, ground_truth_classes)
