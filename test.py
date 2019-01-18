import utils
import numpy as np

if __name__ == '__main__':
    num_train = 50000
    num_test = 10000
    featrue_preserve_radio = .95
    x_train, y_train, x_test, y_test = utils.load_data()
    x_train = x_train[0:num_train, :]
    y_train = y_train[0:num_train]
    x_test = x_test[0:num_test, :]
    y_test = y_test[0:num_test]
    # utils.pca_visualize(x_train, y_train)
    # x_train_pca, x_test_pca = utils.pca(x_train, x_test, featrue_preserve_radio, show_approximation=True, save_model=True)
