# coding: utf-8

import sys
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr

reload(sys)
sys.setdefaultencoding("utf-8")


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content.decode('utf-8')))
                    labels.append(label.decode('utf-8'))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)  # 每个元素出现的次数
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))  # 解压，获取第一维存入words
    # 添加一个 <PAD> 来将所有文本pad为同一长度

    # words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir, 'r') as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [(_.strip().decode('utf-8')) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['normal', 'spam']

    categories = [(x.decode('utf-8')) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    print(cat_to_id)

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字，将一条由id表示的数据重新转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=100):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))  # 把括号中的东西进行打乱,是一个array
    x_shuffle = x[indices]  # 把x中的元素按照indices中的顺序重排
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
