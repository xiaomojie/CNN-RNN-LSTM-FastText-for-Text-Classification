# coding: utf-8

# tensorflow 1.7.0
import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 200  # 词向量维度
    seq_length_word = 30  # 序列长度
    seq_length_char = 60  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 4  # 卷积核尺寸
    vocab_size = 600000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 256  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 60  # 每多少轮输出一次结果
    save_per_batch = 60  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 4个待输入的数据
        self.input_x_word = tf.placeholder(tf.int32, [None, self.config.seq_length_word], name='input_x_word')
        self.input_x_char = tf.placeholder(tf.int32, [None, self.config.seq_length_char], name='input_x_char')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding_word = tf.get_variable('embedding_word', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs_word = tf.nn.embedding_lookup(embedding_word, self.input_x_word)

            embedding_char = tf.get_variable('embedding_char', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs_char = tf.nn.embedding_lookup(embedding_char, self.input_x_char)

        with tf.name_scope("word_cnn"):
            # CNN layer
            conv_word = tf.layers.conv1d(embedding_inputs_word, self.config.num_filters, self.config.kernel_size, name='conv_word')
            # global max pooling layer
            gmp_word = tf.reduce_mean(conv_word, reduction_indices=[1], name='gmp_word')

        with tf.name_scope('char_cnn'):
            # CNN layer
            conv_char = tf.layers.conv1d(embedding_inputs_char, self.config.num_filters, self.config.kernel_size,
                                    name='conv_char')
            # global max pooling layer
            gmp_char = tf.reduce_mean(conv_char, reduction_indices=[1], name='gmp_char')

        with tf.name_scope("score"):
            gmps = [gmp_word, gmp_char]
            gmp = tf.concat(gmps, 1)

            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            self.prob = tf.nn.softmax(self.logits, name='prob')

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
