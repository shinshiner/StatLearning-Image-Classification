import os
import argparse

from utils import config_parser


def init_parser():
    parser = argparse.ArgumentParser(description='Cifar-10 Classification')
    parser.add_argument('--method', type=str, default='nn',
                        help='training method')
    parser.add_argument('--seed', type=int, default=666,
                        help='random seed')
    parser.add_argument('--data-dir', type=str, default='splited_data',
                        help='path to data files')
    parser.add_argument('--eval-dir', type=str, default='origin_data',
                        help='path to eval data files')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='path to configuration files')
    parser.add_argument('--model-dir', type=str, default='trained_models',
                        help='path to trained models')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='path to logs')
    parser.add_argument('--mode', type=str, default='train',
                        help='train / test')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_parser()
    if not os.path.exists('splited_data/train.csv'):
        from utils import split_train_test
        split_train_test('origin_data/train.csv', 'splited_data', args.seed)

    configs = config_parser(os.path.join(args.config_dir, '%s.json' % args.method))
    if args.method == 'nn':
        from nn.main import *
        main(args, configs)
    elif args.method == 'sknn':
        from utils import sk_read
        feats, lbls = sk_read('origin_data/train.csv')
        from sknn.main import *
        main(args, feats, lbls, configs)
    elif args.method == 'svm':
        from utils import sk_read
        feats, lbls = sk_read('origin_data/train.csv')
        from svm.main import *
        main(args, feats, lbls, configs)
    else:
        raise NotImplementedError('Method %s not implemented yet !' % args.method)