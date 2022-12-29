import argparse
from data_loader import load_data
from train import train
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def print_setting(args):
    assert args.use_neighbor or args.use_analogy
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('epoch: ' + str(args.epoch))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('use_sample: ' + str(args.use_sample))

    print('use_neighbor: ' + str(args.use_neighbor))
    if args.use_neighbor:
        print('neighbor_hops: ' + str(args.neighbor_hops))
        print('neighbor_samples: ' + str(args.neighbor_samples))
        print('neighbor_agg: ' + args.neighbor_agg)

    print('use_analogy: ' + str(args.use_analogy))
    if args.use_analogy:
        print('max_path_len: ' + str(args.max_path_len))
        print('path_type: ' + args.path_type)
    if args.use_sample:
        print('path_samples: ' + str(args.path_samples))
        print('path_agg: ' + args.path_agg)
    print('=============================================')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, help='use gpu', action='store_true')

    '''
    # ===== FB15k-237 ===== #
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--use_sample', type=bool, default=True, help='whether use sampling')


    # settings for subgraph context
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use subgraph context')
    parser.add_argument('--neighbor_hops', type=int, default=2, help='number of neighbor hops')
    parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for analogous subgraphs
    parser.add_argument('--use_analogy', type=bool, default=True, help='whether use analogy subgraphs')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='mlp', help='path representation type: mlp, att, mean')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using sampling')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator: mean, att')
    '''

    # ===== wn18rr ===== #
    parser.add_argument('--dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--use_sample', type=bool, default=False, help='whether use sampling')

    # settings for subgraph context
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use subgraph context')
    parser.add_argument('--neighbor_hops', type=int, default=3, help='number of neighbor hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for analogous subgraphs
    parser.add_argument('--use_analogy', type=bool, default=True, help='whether use analogy subgraphs')
    parser.add_argument('--max_path_len', type=int, default=4, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='mlp', help='path representation type: mlp, att, mean')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using sampling')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using mean, att')

    args = parser.parse_args()
    print_setting(args)
    data = load_data(args)
    train(args, data)


if __name__ == '__main__':
    main()
