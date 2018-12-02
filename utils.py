import os
import random


def split_train_test(source_file, target_dir, seed):
    print('======== spliting dataset =======')
    with open(source_file, 'r') as f:
        lines = f.readlines()[1:]

    train_id = []
    test_id = []
    random.seed(seed)

    # split train-test as 9-1
    for cls in range(12):
        sample_range = list(range(cls * 650, (cls + 1) * 650))
        random.shuffle(sample_range)
        test_id.extend(sample_range[:65])
        train_id.extend(sample_range[65:])

    # write the results
    with open(os.path.join(target_dir, 'train.csv'), 'w') as f:
        for i in train_id:
            f.write(lines[i])

    with open(os.path.join(target_dir, 'test.csv'), 'w') as f:
        for i in test_id:
            f.write(lines[i])


def compare(f1, f2):
    with open(f1, 'r') as f1:
        with open(f2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

    for l1, l2 in zip(lines1, lines2):
        if l1 != l2:
            print(l1[:-1], '  ', l2[:-1])


if __name__ == '__main__':
    compare('results.csv', 'nn/results-98675.csv')