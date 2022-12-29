import argparse
import os
import pickle
import random
import sys

import dgl
import numpy as np
import torch

from config import ntypes


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
    if args.data_type in ["kg", "ng"]:
        with open(f"output/{args.dataset}_split.pt", "rb") as f:
            graph, train_pos_edges, val_edges, test_edges = torch.load(f)
    else:
        with open(f"datasets/processed_{args.dataset}_hg.pkl", "rb") as f:
            graph, *_ = pickle.load(f)

    return graph  # , train_pos_edges, val_edges, test_edges


def process(args, graph):
    graph.create_formats_()

    ntype0, ntype1 = ntypes[args.dataset]

    if args.data_type in ["hg", "nc"]:
        graph_homo = dgl.to_homogeneous(graph)
    elif args.data_type == "kg":
        graph_homo = graph
    else:
        assert False

    graph_homo = graph_homo.remove_self_loop()

    print(graph.ntypes)

    original_stdout = sys.stdout

    with open(f"output/{args.dataset}_graph.txt", "w") as f:
        sys.stdout = f

        if args.data_type == "hg":
            task = 0
        elif args.data_type == "kg":
            task = 1
        else:
            task = 2

        print(graph_homo.number_of_nodes(), graph_homo.number_of_edges(), task)
        if task != 2:
            print(graph.ntypes.index(ntype0), graph.ntypes.index(ntype1))
        else:
            print(0, 0)

        type_id = 0
        type_ids = []
        for ntype in graph.ntypes:
            type_ids += [type_id] * graph.number_of_nodes(ntype)
            if task != 2:
                type_id += 1

        print(" ".join(map(str, type_ids)))

        us, vs = graph_homo.all_edges()
        if args.data_type in ["hg", "nc"]:
            for u, v in zip(us, vs):
                assert u != v
                print(u.item(), v.item())
        else:
            types = graph_homo.edata["etype"]
            for u, v, t in zip(us, vs, types):
                print(u.item(), v.item(), t.item())

        sys.stdout = original_stdout

    return graph


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
    if args.dataset in ["fb15k237", "wn18"]:
        args.data_type = "kg"
    elif args.dataset in ["AIFB", "AM", "acm"]:
        args.data_type = "nc"
    else:
        args.data_type = "hg"

    graph = load_data(args)
    # graph = preprocess(graph)

    graph = process(args, graph)

    # save_data(args, graph, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges)


if __name__ == "__main__":
    main()
