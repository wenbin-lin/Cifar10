from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
import keras
import numpy as np
import utils

def get_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(units=256, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l2(0.01), activation='relu', name='fc0'))
    model.add(Dense(units=256, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l2(0.01), activation='relu', name='fc1'))
    model.add(Dense(units=num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    activation='softmax', name='fc2'))
    return model

if __name__ == "__main__":
    num_train = 50000
    num_test = 10000
    featrue_preserve_radio = .95
    x_train, y_train, x_test, y_test = utils.load_data()
    x_train = x_train[0:num_train, :]
    y_train = y_train[0:num_train]
    x_test = x_test[0:num_test, :]
    y_test = y_test[0:num_test]
    x_train_pca, x_test_pca = utils.pca(x_train, x_test, featrue_preserve_radio)
    y_train = utils.convert_to_one_hot(y_train, 10)
    y_test = utils.convert_to_one_hot(y_test, 10)

    model = get_model(x_train_pca.shape[1:], 10)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train_pca, y=y_train, epochs=10, batch_size=128)
    preds = model.evaluate(x_test_pca, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()

    predicted_prob = model.predict(x_train_pca[0:5])
    predicted_classes = [utils.label_names[x] for x in np.argmax(predicted_prob, axis=1)]
    ground_truth_classes = [utils.label_names[x] for x in np.argmax(y_train[0:5], axis=1)]
    print(predicted_classes, ground_truth_classes)
