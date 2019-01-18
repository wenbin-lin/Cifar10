from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import utils

def param_grid():
    print("Fitting the classifier to the training set")
    tic = time.time()
    param_grid = {'C': [1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                       param_grid, cv=5)
    clf = clf.fit(x_train_pca, y_train)
    print("done in %0.3fs" % (time.time() - tic))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print("Predicting people's names on the test set")
    tic = time.time()
    y_pred = clf.predict(x_test_pca)
    print("done in %0.3fs" % (time.time() - tic))

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(classification_report(y_test, y_pred, target_names=classes))
    print(confusion_matrix(y_test, y_pred, labels=range(10)))

if __name__ == '__main__':
    num_train = 4096
    num_test = 1024
    # featrue_preserve_radio = .95
    x_train, y_train, x_test, y_test = utils.load_data()
    x_train = x_train[0:num_train, :]
    y_train = y_train[0:num_train]
    x_test = x_test[0:num_test, :]
    y_test = y_test[0:num_test]
    # x_train_pca, x_test_pca = utils.pca(x_train, x_test, featrue_preserve_radio)
    x_train_pca, x_test_pca = utils.pca_with_model(pca_model_name='pca_model.sav',
                                                   scaler_model_name='scaler_model.sav',
                                                   x_train=x_train, x_test=x_test)
    clf = SVC(kernel='linear')
    # clf = SVC(kernel='rbf', gamma=0.0005, C=10)
    tic = time.time()
    clf.fit(x_train_pca, y_train)
    score = clf.score(x_test_pca, y_test)
    toc = time.time()
    print("train time: " + str(1000 * (toc - tic)) + "ms")
    print("score: {:.6f}".format(score))

    y_pred = clf.predict(x_test_pca)
    print(classification_report(y_test, y_pred, target_names=utils.label_names))
    utils.plot_confusion_matrix(confusion_matrix(y_test, y_pred))

    '''train_sizes, train_scores, test_scores = learning_curve(estimator=clf, X=x_train_pca, y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    utils.plot_learning_curve(train_sizes, train_scores, test_scores)'''
