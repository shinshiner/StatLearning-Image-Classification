import os
import argparse


def init_parser():
    parser = argparse.ArgumentParser(description='Cifar-10 Classification')
    parser.add_argument('--method', type=str, default='nn',
                        help='training method')
    parser.add_argument('--seed', type=int, default=666,
                        help='random seed')
    parser.add_argument('--data-dir', type=str, default='splited_data',
                        help='path to data files')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='path to configuration files')
    parser.add_argument('--model-dir', type=str, default='trained_models',
                        help='path to trained models')
    parser.add_argument('--mode', type=str, default='train',
                        help='train / test')

    return parser.parse_args()


if __name__ == '__main__':
    args = init_parser()
    if not os.path.exists('splited_data/train.csv'):
        from utils import split_train_test
        split_train_test('origin_data/train.csv', 'splited_data', args.seed)

    if args.method == 'nn':
        from nn.main import *
        main(args)
    else:
        print('Not implemented !')