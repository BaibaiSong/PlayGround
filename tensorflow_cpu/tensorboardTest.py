__author__ = 'Song'
# from utils import mnist_reader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# def get_fashion_data():
#     root = 'fashion'
#     X_train, Y_train = mnist_reader.load_mnist(root, kind='train')
#     X_test, Y_test = mnist_reader.load_mnist(root, kind='t10k')
#     print(X_train.shape)
#     print(Y_train.shape)
#     print(X_test.shape)
#     print(Y_test.shape)

def get_data():
    N = 100  # sample
    D = 2  # dimensionality
    K = 3  # type
    var = 5
    X = np.zeros((N*K, D))  # sample input
    Y = np.zeros((N*K), dtype=int)
    for i in range(K):
        ix = range(N*i, N*(i+1))
        r = np.linspace(0, 1, N)  # radius
        t = np.linspace(i*var, (i+1)*var, N) + np.random.randn(N)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = int(i)

    # print(Y)
    # plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    return X, Y, K


def add_layer(inputs, d, neural_num, n_layer, activate_function=None):
    # d = inputs.shape[1]
    # print('n,d:', d)
    layer_name = 'layer %d' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weight = tf.Variable(tf.random_normal([d, neural_num]), name='w')
            tf.summary.histogram(layer_name+'/weights', weight)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, neural_num])+0.1, name='b')
            tf.summary.histogram(layer_name+'/bias', bias)

        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weight)+bias

        if activate_function is None:
            output = wx_plus_b
        else:
            output = activate_function(wx_plus_b)
        tf.summary.histogram(layer_name+'/output', output)
    return output, weight, bias

def display(X, Y, W, b, W2, b2):

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, marker='o', cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #fig.savefig('spiral_linear.png')
    plt.show()

def taskgo():
    X, Y, ktype = get_data()
    n, d = X.shape

    yy = np.zeros([n, ktype])
    yy[range(n), Y] = 1
    print('=============')
    print(yy[95:105])
    print('=============')
    print(yy[195:205])

    hidden = 128
    # xxx = tf.placeholder(tf.float32, [xxn, xxd])
    # yyy = tf.placeholder(tf.float32, [yyn, yyd])
    with tf.name_scope('inputs'):
        xtrain = tf.placeholder(tf.float32, [None, d], name='x_input')
        ytrain = tf.placeholder(tf.float32, [None, 3], name='y_input')

    l1, W1, b1 = add_layer(xtrain, 2, hidden, 1, activate_function=tf.nn.relu)

    predection, W2, b2 = add_layer(l1, 128, ktype, 2, activate_function=None)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predection, labels=ytrain))
        tf.summary.scalar('loss', loss)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    iterate_time = 501

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(init)
        for i in range(iterate_time):
            sess.run(train_op, feed_dict={xtrain:X, ytrain:yy})
            if i % 50 == 0:
                print('iterate %d loss %f:' % (i, sess.run(loss, feed_dict={xtrain:X, ytrain:yy})))
                res = sess.run(merged, feed_dict={xtrain:X, ytrain:yy})
                writer.add_summary(res, i)
        l1, W1, b1 = sess.run((l1, W1, b1), feed_dict={xtrain:X})
        predection, W2, b2 = sess.run((predection, W2, b2), feed_dict={xtrain:X})
        # print(type(W1), W1.shape)

    display(X, Y, W1, b1, W2, b2)


if __name__ == '__main__':
    taskgo()