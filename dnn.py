from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import time
import utils

def get_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(units=256, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(units=256, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(units=num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    activation='softmax'))
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
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train_pca, x_test_pca = utils.pca(x_train, x_test, featrue_preserve_radio)
    '''x_train_pca, x_test_pca = utils.pca_with_model(pca_model_name='pca_model.sav',
                                                   scaler_model_name='scaler_model.sav',
                                                   x_train=x_train, x_test=x_test)'''
    y_train = utils.convert_to_one_hot(y_train, 10)
    y_test = utils.convert_to_one_hot(y_test, 10)

    model = get_model(x_train_pca.shape[1:], 10)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    tic = time.time()
    history = model.fit(x=x_train_pca, y=y_train, epochs=20, batch_size=256, validation_data=(x_test_pca, y_test),
                        callbacks=[TensorBoard(log_dir='./logs')])
    toc = time.time()
    print("train time: " + str(1000 * (toc - tic)) + "ms")
    utils.plot_history(history)

    preds = model.evaluate(x_test_pca, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()
    model.save('model_dnn.h5')

    y_prob = model.predict(x_test_pca)
    y_pred = [x for x in np.argmax(y_prob, axis=1)]
    y_truth = [x for x in np.argmax(y_test, axis=1)]
    print(confusion_matrix(y_truth, y_pred))
    print(classification_report(y_truth, y_pred, target_names=utils.label_names))
