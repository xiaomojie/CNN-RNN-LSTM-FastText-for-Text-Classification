# coding: utf-8

import sys
from collections import Counter
import jieba
import numpy as np
import tensorflow.contrib.keras as kr
import re
reload(sys)
sys.setdefaultencoding("utf-8")


def read_file(filename):
    data = []
    stopwords = [' ', '？', '。', '！', '；', '-', '\.', '“', '!', ';', '，',
                 '~', '【', '〖', '〗', '】', '『', '』', '《', '》', '丨', '丿', '（', '）',
                 '～', '丶', '、']
    pattern = re.compile('|'.join(stopwords))
    with open(filename, 'r') as f:
        for line in f:
            data.append(re.sub(pattern, '', line).decode('utf-8'))
    return data


def read_data(filename):
    contents, label = [], []

    with open(filename, 'r') as f:
        for line in f:
            # print(line)
            segs = line.strip().split('\t')
            contents.append(segs[0].decode('utf-8'))
            label.append(segs[1])

    return contents, label


def segment(data):
    """读取文件数据"""
    contents = []
    stopwords = [u' ', u'？', u'。', u'！', u'；', u'-', u'.', u'\"', u'“', u'\'', u'!', u'?', u'‼', u';', u'，',
                 u'~', u'【', u'〖', u'〗', u'】', u'『', u'』', u'《', u'》', u'丨', u'丿', u'（', u'）',
                 u'～', u'丶', u'、']

    for line in data:
        try:
            segs = list(jieba.cut(line.decode('utf-8')))

            words = []
            for seg in segs:
                if seg not in stopwords:
                    words.append(seg)
            message = ''.join(words).encode('utf-8')

            segs = list(jieba.cut(message))

            contents.append(segs)
        except:
            pass

    return contents


def build_vocab(spam_dir, normal_dir, vocab_dir, vocab_size):
    """根据训练集构建词汇表，存储"""
    spam_data = read_file(spam_dir)
    normal_data = read_file(normal_dir)
    data_origin = spam_data + normal_data

    data_segment = segment(data_origin)

    all_data = []
    for content in data_segment:
        all_data.extend(content)

    for content in data_origin:
        all_data.extend(list(content))

    counter = Counter(all_data)  # 每个元素出现的次数
    count_pairs = counter.most_common(vocab_size)
    words, _ = list(zip(*count_pairs))  # 解压，获取第一维存入words

    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir, 'r') as fp:
        words = [(_.strip().decode('utf-8')) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['normal', 'spam']

    categories = [(x.decode('utf-8')) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字，将一条由id表示的数据重新转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, seq_length_word, seq_length_char):
    """将文件转换为id表示"""
    content, label = read_data(filename)
    contents_segs = segment(content)

    print(label[0] + '\t' + ' '.join(contents_segs[0]))
    print(label[0] + '\t' + ' '.join(content[0]))

    data_id, label_id = [], []
    for i in range(len(contents_segs)):
        data_id.append([word_to_id[x] for x in list(contents_segs)[i] if x.decode('utf-8') in word_to_id])
        label_id.append(cat_to_id[label[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad_word = kr.preprocessing.sequence.pad_sequences(data_id, seq_length_word)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    data_id, label_id = [], []
    for i in range(len(content)):
        data_id.append([word_to_id[x] for x in list(content[i]) if x in word_to_id])
        label_id.append(cat_to_id[label[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad_char = kr.preprocessing.sequence.pad_sequences(data_id, seq_length_char)

    return x_pad_word, x_pad_char, y_pad


def batch_iter(x_train_word, x_train_char, y_train, batch_size):
    """生成批次数据"""
    data_len = len(x_train_word)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))  # 把括号中的东西进行打乱,是一个array
    x_shuffle_word = x_train_word[indices]  # 把x中的元素按照indices中的顺序重排
    x_shuffle_char = x_train_char[indices]  # 把x中的元素按照indices中的顺序重排
    y_shuffle = y_train[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle_word[start_id:end_id], x_shuffle_char[start_id:end_id], y_shuffle[start_id:end_id]
