import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate


def main(args, feats, lbls, configs):
    if args.method == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif args.method == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif args.method == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion='gini')
    elif args.method == 'nb':
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
        model = GaussianNB()
    elif args.method == 'lda':
        from sklearn.lda import LDA
        model = LDA()

    cv_results = cross_validate(model, feats, lbls, cv=5, return_train_score=False)
    print(cv_results['test_score'])
    print(cv_results['test_score'].mean())

    # read evaluation data
    feats_e = []
    with open('origin_data/test.csv', 'r') as f:
        lines = f.readlines()[1:]

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