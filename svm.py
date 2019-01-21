from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import numpy as np
import time
import utils

if __name__ == '__main__':
    num_train = 16384
    num_test = 4096
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
    clf = SVC(kernel='rbf', gamma=0.0005, C=1)
    tic = time.time()
    clf.fit(x_train_pca, y_train)
    toc = time.time()
    score = clf.score(x_test_pca, y_test)
    print("train time: " + str(1000 * (toc - tic)) + "ms")
    print("score: {:.6f}".format(score))

    y_pred = clf.predict(x_test_pca)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=utils.label_names))

    # save model
    pickle.dump(clf, open('svm_model.sav', 'wb'))

    # plot learning curve
    train_sizes, train_scores, test_scores = learning_curve(estimator=clf, X=x_train_pca, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=4, n_jobs=4)
    utils.plot_learning_curve(train_sizes, train_scores, test_scores)
