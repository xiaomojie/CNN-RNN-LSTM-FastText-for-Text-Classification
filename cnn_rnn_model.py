# coding=utf-8
import tensorflow as tf


class TCNN_RNNConfig(object):
    # embedding
    embedding_dim = 100
    seq_length = 100
    num_classes = 2
    vocab_size = 10000
    bitch_size = 64
    learning_rate = 1e-3
    dropout_keep_prob = 0.5
    num_epochs = 5

    # cnn parameter
    num_filters = 256  # 卷积核个数
    kernel_size = 2  # 卷积核大小


    # fully connected
    hidden_dim = 128  # 隐藏层神经元数

    print_per_batch = 20  # 每多少轮输出一次结果
    save_per_batch = 20  # 每多少轮存入tensorboard

    # rnn
    rnn_layers = 2  # rnn 层数
    lstm_hidden = 256  # 每层lstm单元个数



class TextCNN_RNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn_rnn()

    def cnn_rnn(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cnn'):
            # convolutional layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')

            # global max pooling layer
            # gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        # with tf.name_scope('fc'):
        #     fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
        #     fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        #     fc = tf.nn.relu(fc)

        with tf.name_scope('rnn'):
            lstm_cells = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_hidden, state_is_tuple=True)
            lstm = tf.contrib.rnn.DropoutWrapper(lstm_cells, output_keep_prob=self.keep_prob)
            cells = [lstm for _ in range(self.config.rnn_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _output, _ = tf.nn.dynamic_rnn(rnn_cell, inputs=conv, dtype=tf.float32)

            last = _output[:, -1, :]

        with tf.name_scope('score'):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc2')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc3')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


