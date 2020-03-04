#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class BiLSTMConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100      # 词向量维度
    seq_length = 150        # 序列长度
    num_classes = 2        # 类别数
    vocab_size = 10000       # 词汇表达小

    num_layers = 2           # 隐藏层层数
    hidden_dim = 100        # 隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 5          # 总迭代轮次

    print_per_batch = 20    # 每多少轮输出一次结果
    save_per_batch = 20      # 每多少轮存入tensorboard


class TextBiLSTM(object):
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

        # def lstm_cell():   # lstm核
        #     return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        #
        # def gru_cell():  # gru核
        #     return tf.contrib.rnn.GRUCell(self.config.hidden_dim)
        #
        # def dropout():  # 为每一个rnn核后面加一个dropout层
        #     if (self.config.rnn == 'lstm'):
        #         cell = lstm_cell()
        #     else:
        #         cell = gru_cell()
        #     return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_dim, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_dim, state_is_tuple=True)

            _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=embedding_inputs, dtype=tf.float32)
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
            _, ((_, output_fw), (_, output_bw)) = _output   # 获取的是hidden state
            output = tf.concat([output_fw, output_bw], axis=-1) # 前向和后向最后一次输出的隐藏状态hidden state，其实就是最后一个time的输出

            # 多层rnn网络
            # cells = [dropout() for _ in range(self.config.num_layers)]
            # rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            #
            # _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            # last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            # print(_outputs)
            # print(_[0],_[1])
            # last = _[1][1]  # 取最后一个时序输出作为结果


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(output, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
