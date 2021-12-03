from __future__ import division
import math
import pickle
import sklearn
import numpy as np
from bonnerlib2 import dfContour
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def gen_data(mu0, mu1, cov0, cov1, N0, N1):
    cov0_matrix = np.array([[1, cov0], [cov0, 1]])
    cov1_matrix = np.array([[1, cov1], [cov1, 1]])
    n0_dps = np.random.multivariate_normal(mu0, cov0_matrix, N0)
    n1_dps = np.random.multivariate_normal(mu1, cov1_matrix, N1)
    t0 = np.zeros(N0)
    t1 = np.ones(N1)
    dps = np.append(n0_dps, n1_dps, axis=0)
    t = np.append(t0, t1, axis=0)
    shuffled_dps, shuffled_t = sklearn.utils.shuffle(dps, t)
    return shuffled_dps, shuffled_t


def scatter_plot(cluster_data):
    axis = plt.gca()
    dps, t = cluster_data
    apply_color = np.vectorize(lambda x: 'r' if x == 0 else 'b')
    colors = apply_color(t)
    axis.set_xlim([-3, 6])
    axis.set_ylim([-3, 6])
    axis.scatter(dps[:, 0], dps[:, 1], c=colors, s=2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_tp(pred, t):
    return np.sum((pred == 1) & (t == 1))


def get_fp(pred, t):
    return np.sum((pred == 1) & (t == 0))


def get_fn(pred, t):
    return np.sum((pred == 0) & (t == 1))


def get_precision(pred, t):
    return get_tp(pred, t) / (get_tp(pred, t) + get_fp(pred, t))


def get_recall(pred, t):
    return get_tp(pred, t) / (get_tp(pred, t) + get_fn(pred, t))


def show_nn_result(clf, train_data, test_data):
    test_dps, test_t = test_data
    test_pred = clf.predict(test_dps).astype(int)
    print('accuracy of the neural net on test data is: {}'.format(
        clf.score(test_dps, test_t)))
    print('precision of the neural net on test data is: {}'.format(
        get_precision(test_pred, test_t)))
    print('recall of the neural net on test data is: {}'.format(
        get_recall(test_pred, test_t)))
    scatter_plot(train_data)
    dfContour(clf)
    plt.show()


def q1b(train_data, test_data):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    clf = MLPClassifier(
        hidden_layer_sizes=(1,),
        activation='logistic',
        solver='sgd',
        learning_rate_init=0.01,
        max_iter=1000,
        tol=1e-8
    )
    clf.fit(train_dps, train_t)
    plt.figure()
    plt.title("Question 1(b): Neural net with 1 hidden unit")
    show_nn_result(clf, train_data, test_data)


def multi_nn_train(train_data, test_data, hidden_layer_sizes):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    best_clf = None
    best_accuracy = 0
    for index in range(12):
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='logistic',
            solver='sgd',
            learning_rate_init=0.01,
            max_iter=1000,
            tol=1e-8
        )
        clf.fit(train_dps, train_t)
        accuracy = clf.score(test_dps, test_t)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf
        plt.subplot(4, 3, index + 1)
        scatter_plot(train_data)
        dfContour(clf)
    return best_clf


def stochastic_gradient_descent(val_data, train_data, test_data):
    best_clf = None
    best_accuracy = 0
    for _ in range(10):
        clf = MLPClassifier(
            hidden_layer_sizes=(30,),
            activation='logistic',
            solver='sgd',
            batch_size=100,
            learning_rate_init=1,
            max_iter=10,
            tol=1e-8
        )
        clf.fit(*train_data)
        accuracy = clf.score(*val_data)
        print('validation accuracy of the neural net is: {}'.format(accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf

    train_dps, train_t = train_data
    print('validation accuracy of the best neural net is: {}'.format(
        best_clf.score(*val_data)))
    print('testing accuracy of the best neural net is: {}'.format(
        best_clf.score(*test_data)))
    log_proba = best_clf.predict_log_proba(train_dps)
    one_hot_t = np.zeros((train_t.size, train_t.max()+1))
    one_hot_t[np.arange(train_t.size), train_t] = 1
    cross_entropy = -np.sum(log_proba * one_hot_t)
    print('cross entropy of the best neural net is: {}'.format(cross_entropy))
    print('best learning rate of the neural net is: 1')


def batch_gradient_descent(val_data, train_data, test_data):
    best_clf = None
    best_accuracy = 0
    for _ in range(10):
        clf = MLPClassifier(
            hidden_layer_sizes=(30,),
            activation='logistic',
            solver='sgd',
            batch_size=10000,
            learning_rate_init=1,
            max_iter=10,
            tol=1e-8
        )
        clf.fit(*train_data)
        accuracy = clf.score(*val_data)
        print('validation accuracy of the neural net is: {}'.format(accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf

    train_dps, train_t = train_data
    print('validation accuracy of the best neural net is: {}'.format(
        best_clf.score(*val_data)))
    print('testing accuracy of the best neural net is: {}'.format(
        best_clf.score(*test_data)))
    log_proba = best_clf.predict_log_proba(train_dps)
    one_hot_t = np.zeros((train_t.size, train_t.max()+1))
    one_hot_t[np.arange(train_t.size), train_t] = 1
    cross_entropy = -np.sum(log_proba * one_hot_t)
    print('cross entropy of the best neural net is: {}'.format(cross_entropy))
    print('best learning rate of the neural net is: 1')


def q3c(val_data, train_data, test_data):
    train_dps, train_t = train_data
    clf = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='sgd',
        batch_size=10000,
        learning_rate_init=1,
        max_iter=50,
        tol=1e-8
    )
    clf.fit(*train_data)
    print('training accuracy of the 50 iteration bgd neural net is: {}'.format(
        clf.score(*train_data)))
    print('testing accuracy of the 50 iteration bgd neural net is: {}'.format(
        clf.score(*test_data)))
    log_proba = clf.predict_log_proba(train_dps)
    one_hot_t = np.zeros((train_t.size, train_t.max()+1))
    one_hot_t[np.arange(train_t.size), train_t] = 1
    print('cross entropy of the best neural net is: {}'.format(
        np.sum(log_proba * one_hot_t)))

    clf = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='sgd',
        batch_size=10000,
        learning_rate_init=1,
        max_iter=200,
        tol=1e-8
    )
    clf.fit(*train_data)
    print('training accuracy of the 200 iteration bgd neural net is: {}'.format(
        clf.score(*train_data)))
    print('testing accuracy of the 200 iteration bgd neural net is: {}'.format(
        clf.score(*test_data)))
    log_proba = clf.predict_log_proba(train_dps)
    one_hot_t = np.zeros((train_t.size, train_t.max()+1))
    one_hot_t[np.arange(train_t.size), train_t] = 1
    print('cross entropy of the best neural net is: {}'.format(
        np.sum(log_proba * one_hot_t)))


def score(X, T, W, w0, V, v0):
    U = np.matmul(X, V) + v0
    H = sigmoid(U)
    Z = np.matmul(H, W) + w0
    return np.sum(T == np.argmax(Z, axis=1)) / len(T)


def get_cross_entropy(X, T, W, w0, V, v0):
    num_samples = X.shape[0]
    num_labels = T.max() + 1
    one_hot_t = np.zeros((num_samples, num_labels))
    one_hot_t[np.arange(num_samples), T] = 1
    U = np.matmul(X, V) + v0
    H = sigmoid(U)
    Z = np.matmul(H, W) + w0
    exp_Z = np.exp(Z)
    O = exp_Z / np.sum(exp_Z, axis=1).reshape(-1, 1)
    C = -np.sum(np.log(O) * one_hot_t)
    return C


def my_bgd(train_data, test_data, num_units=30, learning_rate=10, iters=100, verbose=False):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    num_samples = train_dps.shape[0]
    num_features = train_dps.shape[1]
    num_labels = train_t.max() + 1
    T = np.zeros((num_samples, num_labels))  # matrix: 10000 * 10
    T[np.arange(num_samples), train_t] = 1
    X = train_dps  # matrix: 10000 * 784
    V = np.random.normal(0, 1, (num_features, num_units))  # matrix: 784 * 30
    v0 = np.zeros(num_units)  # vector: 30 * 1
    U = np.empty((num_samples, num_units))  # matrix: 10000 * 30
    H = np.empty((num_samples, num_units))  # matrix: 10000 * 30
    W = np.random.normal(0, 1, (num_units, num_labels))  # matrix: 30 * 10
    w0 = np.zeros(num_labels)  # vector: 10 * 1
    Z = np.empty((num_samples, num_labels))  # matrix: 10000 * 10
    O = np.empty((num_samples, num_labels))  # matrix: 10000 * 10
    C = np.inf  # scala: infinity

    for epoch in range(iters):
        # forward pass
        U = np.matmul(X, V) + v0
        H = sigmoid(U)
        Z = np.matmul(H, W) + w0
        exp_Z = np.exp(Z)
        O = exp_Z / np.sum(exp_Z, axis=1).reshape(-1, 1)
        # back propagation
        C = -np.sum(np.log(O) * T)
        dCdZ = O - T
        dCdW = np.matmul(H.T, dCdZ)
        dCdw0 = np.sum(dCdZ)
        dCdH = np.matmul(dCdZ, W.T)
        dCdU = H * (1 - H) * dCdH
        dCdV = np.matmul(X.T, dCdU)
        dCdv0 = np.sum(dCdU)
        # here the batch_size == training_set_size
        W -= learning_rate * dCdW / num_samples
        w0 -= learning_rate * dCdw0 / num_samples
        V -= learning_rate * dCdV / num_samples
        v0 -= learning_rate * dCdv0 / num_samples
        if verbose and epoch % 10 == 0:
            test_accuracy = score(test_dps, test_t, W, w0, V, v0)
            print('epoch {}, test data accuracy is {}'.format(
                epoch+1, test_accuracy))

    return W, w0, V, v0


def q3d(val_data, train_data, test_data):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    val_dps, val_t = val_data
    best_weights = None
    best_accuracy = 0
    for _ in range(10):
        W, w0, V, v0 = my_bgd(train_data, test_data,
                              num_units=30, learning_rate=10, iters=100)
        accuracy = score(val_dps, val_t, W, w0, V, v0)
        print('valdation accuracy of my bgd neural network is: {}'.format(accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = (W, w0, V, v0)

    W, w0, V, v0 = best_weights
    val_accuracy = score(val_dps, val_t, W, w0, V, v0)
    test_accuracy = score(test_dps, test_t, W, w0, V, v0)
    print('valdation accuracy of my best bgd neural network is: {}'.format(val_accuracy))
    print('test accuracy of my best bgd neural network is: {}'.format(test_accuracy))
    cross_entropy = get_cross_entropy(train_dps, train_t, W, w0, V, v0)
    print('cross entropy of my best bgd neural net is: {}'.format(cross_entropy))
    print('best learning rate of the neural net is: 10')


def my_sgd(train_data, test_data, batch_size=100, num_units=30, learning_rate=10, iters=100, verbose=False):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    num_samples = train_dps.shape[0]
    num_features = train_dps.shape[1]
    num_labels = train_t.max() + 1
    X = train_dps  # matrix: 10000 * 784
    V = np.random.normal(0, 1, (num_features, num_units))  # matrix: 784 * 30
    v0 = np.zeros(num_units)  # vector: 30 * 1
    U = np.empty((batch_size, num_units))  # matrix: 100 * 30
    H = np.empty((batch_size, num_units))  # matrix: 100 * 30
    W = np.random.normal(0, 1, (num_units, num_labels))  # matrix: 30 * 10
    w0 = np.zeros(num_labels)  # vector: 10 * 1
    Z = np.empty((batch_size, num_labels))  # matrix: 100 * 10
    O = np.empty((batch_size, num_labels))  # matrix: 100 * 10
    C = np.inf  # scala: infinity

    batch_iter = int(math.ceil(num_samples / batch_size))
    for epoch in range(iters):
        # epoch start
        X, T = sklearn.utils.shuffle(train_dps, train_t)
        for curr_batch_iter in range(batch_iter):
            # mini-batch created
            curr_X = X[curr_batch_iter * batch_size:
                       (curr_batch_iter+1)*batch_size]
            curr_T = T[curr_batch_iter * batch_size:
                       (curr_batch_iter+1)*batch_size]
            curr_batch_size = len(curr_T)
            one_hot_t = np.zeros((curr_batch_size, num_labels))
            one_hot_t[np.arange(curr_batch_size), curr_T] = 1
            # forward pass
            U = np.matmul(curr_X, V) + v0
            H = sigmoid(U)
            Z = np.matmul(H, W) + w0
            exp_Z = np.exp(Z)
            O = exp_Z / np.sum(exp_Z, axis=1).reshape(-1, 1)
            # back propagation
            C = -np.sum(np.log(O) * one_hot_t)
            dCdZ = O - one_hot_t
            dCdW = np.matmul(H.T, dCdZ)
            dCdw0 = np.sum(dCdZ)
            dCdH = np.matmul(dCdZ, W.T)
            dCdU = H * (1 - H) * dCdH
            dCdV = np.matmul(curr_X.T, dCdU)
            dCdv0 = np.sum(dCdU)
            # here the batch_size == training_set_size
            W -= learning_rate * dCdW / curr_batch_size
            w0 -= learning_rate * dCdw0 / curr_batch_size
            V -= learning_rate * dCdV / curr_batch_size
            v0 -= learning_rate * dCdv0 / curr_batch_size
        if verbose and epoch % 10 == 0:
            test_accuracy = score(test_dps, test_t, W, w0, V, v0)
            print('epoch {}, test data accuracy is {}'.format(
                epoch+1, test_accuracy))

    return W, w0, V, v0


def q3e(val_data, train_data, test_data):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    val_dps, val_t = val_data
    best_weights = None
    best_accuracy = 0
    for _ in range(10):
        W, w0, V, v0 = my_sgd(train_data, test_data, batch_size=100,
                              num_units=30, learning_rate=10, iters=100)
        accuracy = score(val_dps, val_t, W, w0, V, v0)
        print('valdation accuracy of my sgd neural network is: {}'.format(accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = (W, w0, V, v0)

    W, w0, V, v0 = best_weights
    val_accuracy = score(val_dps, val_t, W, w0, V, v0)
    test_accuracy = score(test_dps, test_t, W, w0, V, v0)
    print('valdation accuracy of my best sgd neural network is: {}'.format(val_accuracy))
    print('test accuracy of my best sgd neural network is: {}'.format(test_accuracy))
    cross_entropy = get_cross_entropy(train_dps, train_t, W, w0, V, v0)
    print('cross entropy of my best sgd neural net is: {}'.format(cross_entropy))
    print('best learning rate of the neural net is: 10')


def q3f(train_data, test_data):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    W, w0, V, v0 = my_sgd(train_data, test_data, batch_size=100,
                          num_units=100, learning_rate=10, iters=100, verbose=True)
    train_accuracy = score(train_dps, train_t, W, w0, V, v0)
    test_accuracy = score(test_dps, test_t, W, w0, V, v0)
    print('final training accuracy of my sgd neural network is: {}'.format(
        train_accuracy))
    print('final testing accuracy of my sgd neural network is: {}'.format(test_accuracy))
    cross_entropy = get_cross_entropy(train_dps, train_t, W, w0, V, v0)
    print('final cross entropy of my sgd neural net is: {}'.format(cross_entropy))


def q3g(train_data, test_data):
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    W, w0, V, v0 = my_bgd(train_data, test_data, num_units=100,
                          learning_rate=10, iters=100, verbose=True)
    train_accuracy = score(train_dps, train_t, W, w0, V, v0)
    test_accuracy = score(test_dps, test_t, W, w0, V, v0)
    print('final training accuracy of my bgd neural network is: {}'.format(
        train_accuracy))
    print('final testing accuracy of my bgd neural network is: {}'.format(test_accuracy))
    cross_entropy = get_cross_entropy(train_dps, train_t, W, w0, V, v0)
    print('final cross entropy of my bgd neural net is: {}'.format(cross_entropy))


if __name__ == "__main__":
    # q1
    print('\n')
    print('Question 1')
    print('----------')
    # q1a
    print('\nQuestion 1(a)')
    train_data = gen_data((1, 1), (2, 2), 0, 0.9, 1000, 500)
    test_data = gen_data((1, 1), (2, 2), 0, 0.9, 10000, 5000)
    train_dps, train_t = train_data
    test_dps, test_t = test_data
    # q1b
    print('\nQuestion 1(b)')
    q1b(train_data, test_data)
    # q1c
    print('\nQuestion 1(c)')
    plt.figure()
    plt.suptitle("Question 1(c): Neural net with 2 hidden units")
    q1c_clf = multi_nn_train(train_data, test_data, (2,))
    plt.figure()
    plt.title("Question 1(c): Best neural net with 2 hidden units")
    show_nn_result(q1c_clf, train_data, test_data)
    # q1d
    print('\nQuestion 1(d)')
    plt.figure()
    plt.suptitle("Question 1(d): Neural net with 3 hidden units")
    q1d_clf = multi_nn_train(train_data, test_data, (3,))
    plt.figure()
    plt.title("Question 1(d): Best neural net with 3 hidden units")
    show_nn_result(q1d_clf, train_data, test_data)
    # q1e
    print('\nQuestion 1(e)')
    plt.figure()
    plt.suptitle("Question 1(e): Neural net with 4 hidden units")
    q1e_clf = multi_nn_train(train_data, test_data, (4,))
    plt.figure()
    plt.title("Question 1(e): Best neural net with 4 hidden units")
    show_nn_result(q1e_clf, train_data, test_data)

    # q3
    print('\n')
    print('Question 3')
    print('----------')
    with open('mnist.pickle', 'rb') as f:
        Xtrain, Ttrain, Xtest, Ttest = pickle.load(f)
    val_data = (Xtrain[:10000], Ttrain[:10000])
    train_data = (Xtrain[10000:20000], Ttrain[10000:20000])
    test_data = (Xtest, Ttest)
    # q3a
    print('\nQuestion 3(a)')
    stochastic_gradient_descent(val_data, train_data, test_data)
    # q3b
    print('\nQuestion 3(b)')
    batch_gradient_descent(val_data, train_data, test_data)
    # q3c
    print('\nQuestion 3(c)')
    q3c(val_data, train_data, test_data)
    # q3d
    print('\nQuestion 3(d)')
    q3d(val_data, train_data, test_data)
    # q3e
    print('\nQuestion 3(e)')
    q3e(val_data, train_data, test_data)
    train_data = (Xtrain, Ttrain)
    test_data = (Xtest, Ttest)
    # q3f
    print('\nQuestion 3(f)')
    q3f(train_data, test_data)
    # q3g
    print('\nQuestion 3(g)')
    q3g(train_data, test_data)
