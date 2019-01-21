import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

train_set_path = ('./cifar-10-batches-py/data_batch_1',
                  './cifar-10-batches-py/data_batch_2',
                  './cifar-10-batches-py/data_batch_3',
                  './cifar-10-batches-py/data_batch_4',
                  './cifar-10-batches-py/data_batch_5')
test_set_path = './cifar-10-batches-py/test_batch'

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def convert_to_one_hot(y, num_classes):
    y = np.eye(num_classes)[y.reshape(-1)]
    return y

def load_data():
    images_train = np.zeros(shape=(50000, 3, 32, 32), dtype = np.uint8)
    labels_train = np.zeros(shape=50000, dtype = np.uint8)
    for i in range(5):
        f = open(train_set_path[i], 'rb')
        dict = pickle.load(f, encoding='latin1')
        images_train[10000*i:10000*(i+1), :] = np.reshape(dict['data'], (-1, 3, 32, 32))
        labels_train[10000*i:10000*(i+1)] = np.array(dict['labels'])
    f = open(test_set_path, 'rb')
    dict = pickle.load(f, encoding='latin1')
    images_test = np.reshape(dict['data'], (-1, 3, 32, 32))
    labels_test = np.array(dict['labels'])
    # channel last
    images_train = images_train.transpose([0, 2, 3, 1])
    images_test = images_test.transpose([0, 2, 3, 1])
    # shuffle
    rand_train = np.random.permutation(labels_train.size)
    rand_test = np.random.permutation(labels_test.size)
    return images_train[rand_train], labels_train[rand_train], images_test[rand_test], labels_test[rand_test]

def pca(image_train, image_test, featrue_preserve_ratio, show_approximation=False, save_model=False):
    # flatten
    image_train_flatten = image_train.reshape(image_train.shape[0], -1)
    image_test_flatten = image_test.reshape(image_test.shape[0], -1)
    # normalization
    scaler = StandardScaler()
    scaler.fit(image_train_flatten)
    image_train_flatten = scaler.transform(image_train_flatten)
    image_test_flatten = scaler.transform(image_test_flatten)
    # pca
    pca = PCA(featrue_preserve_ratio)
    pca.fit(image_train_flatten)
    image_train_flatten = pca.transform(image_train_flatten)
    image_test_flatten = pca.transform(image_test_flatten)
    print("explained_variance_ratio: " + str(np.sum(pca.explained_variance_ratio_)))
    print("feature preserve:" + str(pca.n_components_))

    # show approximation after pca
    if show_approximation:
        approximation = pca.inverse_transform(image_train_flatten[0])
        approximation = scaler.inverse_transform(approximation).astype(int)
        print(approximation)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_train[0].reshape(32, 32, 3))
        plt.xlabel(str(image_train[0].size) + 'components', fontsize=10)
        plt.title('Original Image', fontsize=10)
        plt.subplot(1, 2, 2)
        plt.imshow(approximation.reshape(32, 32, 3))
        plt.xlabel(str(image_train_flatten[0].size) + 'components', fontsize=10)
        plt.title('95% of Explained Variance', fontsize=10)
        plt.show()
    if save_model:
        pickle.dump(scaler, open('scaler_model.sav', 'wb'))
        pickle.dump(pca, open('pca_model.sav', 'wb'))
    return image_train_flatten, image_test_flatten

def pca_with_model(pca_model_name, scaler_model_name, x_train, x_test):
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)
    scaler_model = pickle.load(open(scaler_model_name, 'rb'))
    pca_model = pickle.load(open(pca_model_name, 'rb'))
    x_train_flatten = scaler_model.transform(x_train_flatten)
    x_train_flatten = pca_model.transform(x_train_flatten)
    x_test_flatten = scaler_model.transform(x_test_flatten)
    x_test_flatten = pca_model.transform(x_test_flatten)
    return x_train_flatten, x_test_flatten

# reference https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(train_sizes, train_scores, test_scores):
    # get mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # plot
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.0, 1.0])
    plt.show()

# reference https://keras.io/visualization/
def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
