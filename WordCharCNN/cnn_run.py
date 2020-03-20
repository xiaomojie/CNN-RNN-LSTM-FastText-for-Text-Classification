#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
from CNN.cnn_model import TCNNConfig, TextCNN
from data_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from create_data import create_dataset

base_dir = 'data'
spam_dir = os.path.join(base_dir, 'spam.txt')
# spam_dir = os.path.join(base_dir, '1.txt')
normal_dir = os.path.join(base_dir, 'normal_data.txt')
# normal_dir = os.path.join(base_dir, '2.txt')
train_dir = os.path.join(base_dir, 'train.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
log_dir = 'log/cnn.log'

save_dir = 'model/cnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch_word, x_batch_char, y_batch, keep_prob):
    feed_dict = {
        model.input_x_word: x_batch_word,
        model.input_x_char: x_batch_char,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
    }
    return feed_dict


def evaluate(sess, x_word, x_char, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_word)
    batch_eval = batch_iter(x_word, x_char, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch_word, x_batch_char, y_batch in batch_eval:
        batch_len = len(x_batch_word)
        feed_dict = feed_data(x_batch_word, x_batch_char, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    mylog = open(log_dir, 'a')

    print("Configuring TensorBoard and Saver...")
    mylog.write("Configuring TensorBoard and Saver...\n")

    tensorboard_dir = 'tensorboard/cnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    mylog.write('Training and evaluating...\n')

    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 800  # 如果超过100轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        mylog.write('Epoch:' + str(epoch + 1)+'\n')

        create_dataset()  # 对normal样本重新随机采样

        x_train_word, x_train_char, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length_word, config.seq_length_char)
        x_val_word, x_val_char, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length_word, config.seq_length_char)

        batch_train = batch_iter(x_train_word, x_train_char, y_train, config.batch_size)
        for x_batch_word, x_batch_char, y_batch in batch_train:
            feed_dict = feed_data(x_batch_word, x_batch_char, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

                tf.train.write_graph(session.graph_def, './graph/', "nn_model.pbtxt", as_text=True)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val_word, x_val_char, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                mylog.write(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str)+'\n')

            mylog.flush()
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement or best_acc_val == 1:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                mylog.write("No optimization for a long time, auto-stopping...\n")
                mylog.flush()

                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

    mylog.close()


def test():
    mylog = open(log_dir, 'a')

    print("Loading test data...")
    mylog.write("Loading test data...\n")

    start_time = time.time()
    x_test_word, x_test_char, y_test = process_file(val_dir, word_to_id, cat_to_id, config.seq_length_word, config.seq_length_char)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    mylog.write('Testing...\n')

    loss_test, acc_test = evaluate(session, x_test_word, x_test_char, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    mylog.write(msg.format(loss_test, acc_test))
    mylog.write('\n')

    batch_size = 128
    data_len = len(x_test_word)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test_word), dtype=np.int32)  # 保存预测结果
    y_pred_prob = np.zeros(shape=len(x_test_word), dtype=np.float32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x_word: x_test_word[start_id:end_id],
            model.input_x_char: x_test_char[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
        y_pred_prob[start_id:end_id] = session.run(model.prob, feed_dict=feed_dict)[:,1]

    # 评估
    print("Precision, Recall and F1-Score...")
    mylog.write("Precision, Recall and F1-Score...\n")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    mylog.write(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    mylog.write('\n')

    # 混淆矩阵
    print("Confusion Matrix...")
    mylog.write("Confusion Matrix...\n")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    mylog.write(str(cm[0][0]) + '   ' + str(cm[0][1]) + '\n' + str(cm[1][0]) + '   ' + str(cm[1][1]) )

    print('AUC: ')
    mylog.write('AUC:\n')
    auc = metrics.roc_auc_score(y_test_cls, y_pred_prob)
    print(auc)
    mylog.write(str(auc) + '\n')

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    mylog.write("Time usage:" + str(time_dif) + '\n\n\n')
    mylog.flush()
    mylog.close()


if __name__ == '__main__':

    mylog = open(log_dir, 'a')
    print('Configuring CNN model...\n')
    mylog.write('Configuring CNN model...\n')

    config = TCNNConfig()
    print('kernel_size:' + str(config.kernel_size) + '\t num_filters:' + str(config.num_filters) + '\n')
    mylog.write('kernel_size:' + str(config.kernel_size) + '\t num_filters:' + str(config.num_filters) + '\n')

    mylog.flush()
    mylog.close()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(spam_dir, normal_dir, vocab_dir, config.vocab_size)

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    train()
    test()


#python freeze_graph.py --input_graph=./graph/nn_model.pbtxt --input_checkpoint=./model/cnn/best_validation --output_graph=./model/nn_model_frozen.pb --output_node_names="score/Softmax"
