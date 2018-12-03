import os
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier


def main(args, feats, lbls, configs):
    model = MLPClassifier(max_iter=configs['max_iter'], random_state=args.seed,
                          activation=configs['activation'], solver=configs['solver'],
                          learning_rate_init=configs['lr'])
    kfolds = StratifiedKFold(n_splits=10, random_state=args.seed, shuffle=True)

    i = 0
    for tr, t in kfolds.split(feats, lbls):
        i += 1
        print('======= training fold %d ========' % i)
        model.fit(feats[tr], lbls[tr])
        print(model.score(feats[t], lbls[t]))

    # read evaluation data
    feats_e = []
    with open('origin_data/test.csv', 'r') as f:
        lines = f.readlines()

    for line in lines:
        items = line.rstrip().split(',')
        feat = items[1:]
        feats_e.append(feat)

    feats_e = np.array(feats_e).astype(np.float32)
    pred = model.predict(feats_e)
    print(pred.shape)

    # write results
    result = open(os.path.join(args.method, 'results.csv'), 'w')
    result.write('id,categories\n')
    for i in range(pred.shape[0]):
        result.write('%d,%d\n' % (i, pred[i]))
    result.close()


if __name__ == '__main__':
    pass