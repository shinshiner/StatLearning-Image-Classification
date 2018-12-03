import os
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def main(args, feats, lbls, configs):
    feats_tr, feat_t, lbls_tr, lbls_t = train_test_split(feats, lbls,
                            test_size=0.33, random_state=args.seed, shuffle=True)
    model = SVC(C=configs['C'], kernel=configs['kernel'])
    model.fit(feats_tr, lbls_tr)
    print(model.score(feat_t, lbls_t))

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