import numpy as np

i = 0
with open('../data/labeled_data', 'r') as f:
    data = []
    for line in f:
        i += 1
        print(line)
        label, content = line.split('\t')
        data.append(' '.join(list(content.strip().decode('utf-8'))) + "\t__label__" + label + '\n')

train = open('train_data.txt', 'w')
test = open('test_data.txt', 'w')
for i in range(len(data)):
    if i < len(data)*0.8:
        train.write(data[i].encode('utf-8'))
    else:
        test.write(data[i].encode('utf-8'))
train.close()
test.close()

# train = open('test_data.txt', 'w')
# for i in range(len(data)):
#     train.write(data[i].encode('utf-8'))
#
# train.close()