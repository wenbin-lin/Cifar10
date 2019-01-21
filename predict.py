from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pickle
import utils
import sys

def predict_with_svm():
    _, _, x_test, y_test = utils.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], -1)
    scaler_model = pickle.load(open('scaler_model.sav', 'rb'))
    pca_model = pickle.load(open('pca_model.sav', 'rb'))
    svm_model = pickle.load(open('svm_model.sav', 'rb'))
    x_test = scaler_model.transform(x_test)
    x_test = pca_model.transform(x_test)
    y_pred = svm_model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=utils.label_names))

def predict_with_dnn():
    _, _, x_test, y_test = utils.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], -1)
    scaler_model = pickle.load(open('scaler_model.sav', 'rb'))
    pca_model = pickle.load(open('pca_model.sav', 'rb'))
    x_test = scaler_model.transform(x_test)
    x_test = pca_model.transform(x_test)
    y_test = utils.convert_to_one_hot(y_test, 10)
    model = load_model('model_dnn.h5')
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()
    # plot_model(model, to_file='model_dnn.png', show_shapes=True)

    y_prob = model.predict(x_test)
    y_pred = [x for x in np.argmax(y_prob, axis=1)]
    y_truth = [x for x in np.argmax(y_test, axis=1)]
    print(confusion_matrix(y_truth, y_pred))
    print(classification_report(y_truth, y_pred, target_names=utils.label_names))

def predict_with_cnn1():
    _, _, x_test, y_test = utils.load_data()
    x_test = x_test.astype('float32')
    mean = np.load('x_train_mean.npy')
    std = np.load('x_train_std.npy')
    x_test = (x_test - mean) / (std + 1e-7)
    y_test = utils.convert_to_one_hot(y_test, 10)
    model = load_model('model_cnn_v1.h5')
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()
    # plot_model(model, to_file='model_cnn_v1.png', show_shapes=True)

    y_prob = model.predict(x_test)
    y_pred = [x for x in np.argmax(y_prob, axis=1)]
    y_truth = [x for x in np.argmax(y_test, axis=1)]
    print(confusion_matrix(y_truth, y_pred))
    print(classification_report(y_truth, y_pred, target_names=utils.label_names))

def predict_with_cnn2():
    _, _, x_test, y_test = utils.load_data()
    x_test = x_test.astype('float32')
    mean = np.load('x_train_mean.npy')
    std = np.load('x_train_std.npy')
    x_test = (x_test - mean) / (std + 1e-7)
    y_test = utils.convert_to_one_hot(y_test, 10)
    model = load_model('model_cnn_v2.h5')
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Accuracy = " + str(preds[1]))
    model.summary()
    # plot_model(model, to_file='model_cnn_v2.png', show_shapes=True)

    y_prob = model.predict(x_test)
    y_pred = [x for x in np.argmax(y_prob, axis=1)]
    y_truth = [x for x in np.argmax(y_test, axis=1)]
    print(confusion_matrix(y_truth, y_pred))
    print(classification_report(y_truth, y_pred, target_names=utils.label_names))

def main(argv):
    list = ['svm', 'dnn', 'cnn1', 'cnn2']
    if len(argv) != 1 or str(argv[0]) not in list:
        print('Please input method (svm, dnn, cnn1 or cnn2)')
        return
    if argv[0] == 'svm':
        predict_with_svm()
    if argv[0] == 'dnn':
        predict_with_dnn()
    if argv[0] == 'cnn1':
        predict_with_cnn1()
    if argv[0] == 'cnn2':
        predict_with_cnn2()

if __name__ == "__main__":
   main(sys.argv[1:])
