# coding: utf-8

from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from CNN.cnn_model import TCNNConfig, TextCNN
from data_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/'
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'model/cnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_demo = ['小哥哥可以给我你的微信吗',
                 '亲，我们家发布了新款的精品珠宝翡翠玉石，不仅价格美丽，产品更美丽，各种款式的都有，可以加我Wx：pkzx187，快来联系我哦~']
    for i in test_demo:
        print(cnn_model.predict(i))

