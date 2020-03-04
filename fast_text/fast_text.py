# coding=utf-8
import fasttext

clf = fasttext.supervised('train_data.txt', 'fasttext.model', label_prefix='__label__')
result = clf.test('test_data.txt')

print(result.precision)
print(result.recall)

classifier = fasttext.load_model('fasttext.model.bin', label_prefix='__label__')
labels_right = []
texts = []
with open("test_data.txt") as fr:
    for line in fr:
        line = line.decode("utf-8").rstrip()
        labels_right.append(line.split("\t")[1].replace("__label__", ""))
        texts.append(line.split("\t")[0])
    #     print labels
    #     print texts
#     break
labels_predict = [e[0] for e in classifier.predict(texts)]  # 预测输出结果为二维形式
# print labels_predict

text_labels = list(set(labels_right))
text_predict_labels = list(set(labels_predict))
print text_predict_labels
print text_labels

A = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
B = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
C = dict.fromkeys(text_predict_labels, 0)  # 预测结果中各个类的数目
for i in range(0, len(labels_right)):
    B[labels_right[i]] += 1
    C[labels_predict[i]] += 1
    if labels_right[i] == labels_predict[i]:
        A[labels_right[i]] += 1

print A
print B
print C
# 计算准确率，召回率，F值
for key in B:
    try:
        r = float(A[key]) / float(B[key])
        p = float(A[key]) / float(C[key])
        f = p * r * 2 / (p + r)
        print "%s:\t p:%f\t r:%f\t f:%f" % (key, p, r, f)
    except:
        print "error:", key, "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key, 0)
