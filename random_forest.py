from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import time
import utils

if __name__ == '__main__':
    num_train = 4096
    num_test = 512
    x_train, y_train, x_test, y_test = utils.load_data()
    x_train = x_train[0:num_train, :]
    y_train = y_train[0:num_train]
    x_test = x_test[0:num_test, :]
    y_test = y_test[0:num_test]
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    clf = RandomForestClassifier(n_estimators=10)
    tic = time.time()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    toc = time.time()
    print("train time: " + str(1000 * (toc - tic)) + "ms")
    print("score: {:.6f}".format(score))

    y_pred = clf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=utils.label_names))

    '''import graphviz
    dot_data = export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("test")'''

    train_sizes, train_scores, test_scores = learning_curve(estimator=clf, X=x_train, y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    utils.plot_learning_curve(train_sizes, train_scores, test_scores)