字节跳动垃圾文本过滤：CNN, RNN, LSTM, FastText模型及其组合
目标：过滤发文、评论、私信中的垃圾文本

采用了CNN, RNN, LSTM, FastText等多种模型进行实验

1. CNN：基于CNN的文本分类实验
2. RNN：使用rnn和lstm进行文本分类
3. fast_text: 使用fast text模型进行分类实验
4. word_char_cnn: 将基于word和基于character的cnn进行结合
5. cnn_rnn_model.py：将cnn和rnn进行结合
6. 将tensorflow训练出来的模型转化成java可调用的版本，进行上线


相关知识：

交叉熵损失函数的理解与计算：https://blog.csdn.net/chaipp0607/article/details/73392175


网络的设置，层数


一、双向lstm的输出问题：
1. 函数定义：
def bidirectional_dynamic_rnn(
cell_fw, # 前向RNN
cell_bw, # 后向RNN
inputs, # 输入
sequence_length=None,# 输入序列的实际长度（可选，默认为输入序列的最大长度）
initial_state_fw=None,  # 前向的初始化状态（可选）
initial_state_bw=None,  # 后向的初始化状态（可选）
dtype=None, # 初始化和输出的数据类型（可选）
parallel_iterations=None,
swap_memory=False,
time_major=False,# 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`.# 如果为false, tensor的形状必须为`[batch_size, max_time, depth]`.
scope=None
)

返回值：
    一个(outputs, output_states)的元组
    其中，
    1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。假设
    time_major=false,tensor的shape为[batch_size, max_time, depth]。实验中使用tf.concat(outputs, 2)将其拼接。
    2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
    output_state_fw和output_state_bw的类型为LSTMStateTuple。
    LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。

2. 使用最后一个隐藏层输出
_output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=embedding_inputs, dtype=tf.float32)
_, ((_, output_fw), (_, output_bw)) = _output   # 获取的是hidden state
output = tf.concat([output_fw, output_bw], axis=-1) # 前向和后向最后一次输出的隐藏状态hidden state，其实就是最后一个time的输出


3. 使用所有中间状态
(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
output = tf.concat([output_fw, output_bw], axis=-1)  # 所有的中间状态的输出


二、两种不同的LSTM：
tf.contrib.rnn.BasicLSTMCell(经典结构)  就是常见的那种带有遗忘门、输入门、输出门的LSTM结构
tf.contrib.rnn.LSTMCell  带有窥视孔的  没有输入门，代替输入门的是： 1-遗忘门
参考网址：http://www.php-master.com/post/326590.html