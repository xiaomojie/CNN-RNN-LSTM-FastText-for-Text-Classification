import random


def create_dataset():
    f1 = open('./data/spam.txt', 'r')
    f2 = open('./data/normal_data.txt', 'r')
    lines = []
    lines1 = f1.readlines()
    random.shuffle(lines1)
    for line in lines1:
        if line.strip() != '':
            lines.append(line.strip() + '\tspam\n')

    lines2 = f2.readlines()
    random.shuffle(lines2)
    for i in range(len(lines1)):
        if lines2[i].strip() != '':
            lines.append(lines2[i].strip() + '\tnormal\n')

    f1.close()
    f2.close()

    random.shuffle(lines)

    f1 = open('./data/train.txt', 'w')
    for i in range(len(lines)):
        segs = lines[i].split('\t')
        if len(segs) < 2 or segs[1] == '\n':
            continue

        f1.write(lines[i])
        
        # print(lines[i])

    f1.close()

    f2 = open('./data/val.txt', 'w')
    random.shuffle(lines1)
    random.shuffle(lines2)

    for i in range(int(len(lines1)*0.1)):
        f2.write(lines1[i].strip() + '\tspam\n')

        f2.write(lines2[i].strip() + '\tnormal\n')

    f2.close()
