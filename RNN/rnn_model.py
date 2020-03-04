#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100      # 词向量维度
    seq_length = 150        # 序列长度
    num_classes = 2        # 类别数
    vocab_size = 10000       # 词汇表达小

    num_layers= 2           # LSTM层数
    hidden_dim = 128        # LSTM单元内部的门的隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 256         # 每批训练大小
    num_epochs = 5          # 总迭代轮次

    print_per_batch = 50    # 每多少轮输出一次结果
    save_per_batch = 50      # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            # num_units参数（self.config.hidden_dim）并不是指这一层有多少个相互独立的时序lstm，而是lstm单元内部的几个门
            # 的参数，这几个门其实内部是一个神经网络，num_units表示的是这个神经网络隐藏层的单元数
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)  # 层与层之间加上dropout，而不是时序与时序之间

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            """
            返回值：
                一个(outputs, output_states)的元组
                其中，
                1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。假设
                time_major=false,tensor的shape为[batch_size, max_time, depth]。实验中使用tf.concat(outputs, 2)将其拼接。
                2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
                output_state_fw和output_state_bw的类型为LSTMStateTuple。
                LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
            """
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            # print(_outputs)
            # print(_[0],_[1])
            # last = _[1][1]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

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
