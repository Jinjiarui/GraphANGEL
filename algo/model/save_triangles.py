import argparse
import os
import pickle
import random
import sys

import dgl
import numpy as np
import torch


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(args):
    # load from file
    triangles_by_type = dict()
    with open(f"output/{args.dataset}_triangles.txt", "r") as f:
        n_types = int(f.readline())
        for i in range(n_types):
            type0, type1, type2, count = [int(x) for x in f.readline().split()]
            triangle_list = []
            triangles_by_type[(type0, type1, type2)] = triangle_list
            for j in range(count):
                id0, id1, id2 = [int(x) for x in f.readline().split()]
                triangle_list.append((id0, id1, id2))

    neg_triangles_by_type = dict()
    with open(f"output/{args.dataset}_neg_triangles.txt", "r") as f:
        n_types = int(f.readline())
        for i in range(n_types):
            type0, type1, type2, count = [int(x) for x in f.readline().split()]
            triangle_list = []
            neg_triangles_by_type[(type0, type1, type2)] = triangle_list
            for j in range(count):
                id0, id1, id2 = [int(x) for x in f.readline().split()]
                triangle_list.append((id0, id1, id2))

    quadrangles_by_type = dict()
    with open(f"output/{args.dataset}_quadrangles.txt", "r") as f:
        n_types = int(f.readline())
        for i in range(n_types):
            type0, type1, type2, type3, count = [int(x) for x in f.readline().split()]
            quadrangle_list = []
            quadrangles_by_type[(type0, type1, type2, type3)] = quadrangle_list
            for j in range(count):
                id0, id1, id2, id3 = [int(x) for x in f.readline().split()]
                quadrangle_list.append((id0, id1, id2, id3))

    neg_quadrangles_by_type = dict()
    with open(f"output/{args.dataset}_neg_quadrangles.txt", "r") as f:
        n_types = int(f.readline())
        for i in range(n_types):
            type0, type1, type2, type3, count = [int(x) for x in f.readline().split()]
            quadrangle_list = []
            neg_quadrangles_by_type[(type0, type1, type2, type3)] = quadrangle_list
            for j in range(count):
                id0, id1, id2, id3 = [int(x) for x in f.readline().split()]
                quadrangle_list.append((id0, id1, id2, id3))

    return triangles_by_type, neg_triangles_by_type, quadrangles_by_type, neg_quadrangles_by_type


def process(args, triangles_by_type, neg_triangles_by_type, quadrangles_by_type, neg_quadrangles_by_type):
    with open(f"output/{args.dataset}_triangles.pkl", "wb") as f:
        pickle.dump(triangles_by_type, f)
    with open(f"output/{args.dataset}_neg_triangles.pkl", "wb") as f:
        pickle.dump(neg_triangles_by_type, f)
    with open(f"output/{args.dataset}_quadrangles.pkl", "wb") as f:
        pickle.dump(quadrangles_by_type, f)
    with open(f"output/{args.dataset}_neg_quadrangles.pkl", "wb") as f:
        pickle.dump(neg_quadrangles_by_type, f)


def main():
    argparser = argparse.ArgumentParser("GraphANGEL preprocess", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=["lastfm", "douban_book", "douban_movie", "amazon", "yelp", "fb15k237", "wn18", "AIFB", "AM", "acm"],
        default="lastfm",
        help="dataset",
    )
    args = argparser.parse_args()
    # seed(args.seed)

    triangles_by_type, neg_triangles_by_type, quadrangles_by_type, neg_quadrangles_by_type = load_data(args)
    process(args, triangles_by_type, neg_triangles_by_type, quadrangles_by_type, neg_quadrangles_by_type)

    # save_data(args, graph, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges)


if __name__ == "__main__":
    main()
