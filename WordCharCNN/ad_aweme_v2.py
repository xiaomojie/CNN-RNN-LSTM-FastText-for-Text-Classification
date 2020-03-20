# coding: utf-8

from __future__ import print_function
import os
import jieba
import tensorflow as tf
import tensorflow.contrib.keras as kr
from CNN.cnn_model import TCNNConfig, TextCNN
from data_loader import read_category, read_vocab

# jieba.load_userdict("data/words.txt")
base_dir = 'data/'
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'model/cnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class ADModel(object):
    __species = None
    __first_init = True

    # 单例
    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self):
        if self.__first_init:
            self.config = TCNNConfig()
            self.categories, self.cat_to_id = read_category()
            self.words, self.word_to_id = read_vocab(vocab_dir)
            self.config.vocab_size = len(self.words)
            self.model = TextCNN(self.config)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
            self.__class__.__first_init = False

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行

        stopwords = [u' ', u'？', u'。', u'！', u'；', u'-', u'.', u'\"', u'“', u'\'', u'!', u'?', u'‼', u';', u'，',
                     u'~', u'【', u'〖', u'〗', u'】', u'『', u'』', u'《', u'》', u'丨', u'丿', u'（', u'）',
                     u'～', u'丶', u'、']

        segs = list(jieba.cut(message))

        words = []
        for seg in segs:
            if seg not in stopwords:
                words.append(seg)
        message = ''.join(words)

        words = list(jieba.cut(message))

        # print(' '.join(words))
        data = [self.word_to_id[x] for x in words if x in self.word_to_id]

        # print(data)
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        prob = self.session.run(self.model.prob, feed_dict=feed_dict)

        return prob[0][1]


if __name__ == '__main__':

    # for i in range(10000):
    #     pm1 = ADModel()
    #     print(pm1.predict('东北菜五花肉酸菜炖土豆，下个视频出成品。'))
    #     print(id(pm1))

    cnn_model = ADModel()
    f = open('data/spam.txt', 'r')
    lines = f.readlines()
    f.close()

    text = []
    label = []
    from_uid = []
    to_uid = []
    t = 0
    f = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for line in lines:

        segs = line.strip().split('\t')
        # print(line)
        text.append(segs[0])
        if len(segs) == 2:
            label.append(segs[1])
        else:
            label.append('spam')
        # print(segs[1] + '*')

        if label[-1] == 'spam':
            t += 1
        else:
            f += 1


    pm1 = ADModel()

    scores = []
    for i in range(len(text)):
        pred = pm1.predict(text[i])
        scores.append([text[i], pred])

        if pred >= 0.9:
            if label[i] == 'spam':
                tp += 1

            else:
                fp += 1  # 误伤
                # print(pred)
                # print(from_uid[i] + '\t' + to_uid[i] + '\t' + text[i] + '\t' + label[i])

        else:
            if label[i] == 'spam':  # missing
                tn += 1
                # print(pred)
                print(text[i])
                # print(text[i])
            if label[i] == 'normal':
                fn += 1

    # print('fn', fn, 'tp:', tp, 'fp:', fp, 'tn:', tn, 't:', t, 'f', f)
    # r = float(tp) / (tp + tn)
    # p = float(tp) / (tp + fp)
    # f = 2 * p * r / (p + r)
    # print('r:', r)
    # print('p:', p)
    # print('f:', f)

    sorted_scores = sorted(scores, key=lambda d: d[1], reverse=True)
    f = open('data/1-result.txt', 'w')

    for item in sorted_scores:
        f.write(item[0].encode('utf-8') + '\n' + str(item[1]) + '\n')

    f.close()
