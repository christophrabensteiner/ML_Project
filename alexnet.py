import tensorflow as tf

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def alex_net(_X, _dropout):
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64]), name='weights1'),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128]), name='weights2'),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256]), name='weights3'),
        'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024]), name='weights4'),
        'wd2': tf.Variable(tf.random_normal([1024, 1024]), name='weights5'),
        'out': tf.Variable(tf.random_normal([1024, 10]), name='weights_out')
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64]), name='biases1'),
        'bc2': tf.Variable(tf.random_normal([128]), name='biases2'),
        'bc3': tf.Variable(tf.random_normal([256]), name='biases3'),
        'bd1': tf.Variable(tf.random_normal([1024]), name='biases4'),
        'bd2': tf.Variable(tf.random_normal([1024]), name='biases5'),
        'out': tf.Variable(tf.random_normal([10]), name='biases_out')
    }

    # Reshape input picture
    # _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(norm3,
                        [-1, weights['wd1'].get_shape().as_list()[0]])  # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1')  # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2')  # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, weights['out']) + biases['out']
    return out
