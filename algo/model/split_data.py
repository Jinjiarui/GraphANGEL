import argparse
import os
import pickle
import random

import dgl
import numpy as np
import torch
from dgl.data import FB15k237Dataset, WN18Dataset

from config import etypes, ntypes


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
    if args.data_type == "kg":
        dataset = FB15k237Dataset() if args.dataset == "fb15k237" else WN18Dataset()
        graph = dataset[0]
        train_mask, val_mask, test_mask = graph.edata["train_mask"], graph.edata["val_mask"], graph.edata["test_mask"]
        train_idx, val_idx, test_idx = (
            torch.nonzero(train_mask).squeeze(),
            torch.nonzero(val_mask).squeeze(),
            torch.nonzero(test_mask).squeeze(),
        )
        src, dst = graph.edges()
        train_src, train_dst = src[train_idx], dst[train_idx]
        val_src, val_dst = src[val_idx], dst[val_idx]
        test_src, test_dst = src[test_idx], dst[test_idx]
        train_rel = graph.edata["etype"][train_idx]
        val_rel = graph.edata["etype"][val_idx]
        test_rel = graph.edata["etype"][test_idx]

        graph.remove_edges(torch.cat([val_idx, test_idx]))

        return graph, (train_src, train_dst), (val_src, val_dst), (test_src, test_dst), train_rel, val_rel, test_rel
    elif args.data_type == "hg":
        # load from file
        with open(f"datasets/{args.dataset}.pkl", "rb") as f:
            graph = pickle.load(f)

        return (graph,)
    else:
        with open(f"datasets/processed_{args.dataset}_hg.pkl", "rb") as f:
            graph, category, category_id, labels, train_idx, val_idx, test_idx = pickle.load(f)
            # for i, e in enumerate([graph, category, category_id, labels, train_idx, val_idx, test_idx]):
            #     print(i, e)

        # print(graph.ndata[category])

        return graph, category, category_id, labels, train_idx, val_idx, test_idx


def preprocess(graph):
    id_start = 0
    for ntype in graph.ntypes:
        n_nodes = graph.number_of_nodes(ntype)
        print(f"#{ntype}: {n_nodes}")
        # print(graph.ndata[ntype])
        if ntype == "_N":
            graph.ndata["_GLOBAL_ID"] = torch.arange(start=id_start, end=id_start + n_nodes)
        else:
            graph.ndata["_GLOBAL_ID"] = {ntype: torch.arange(start=id_start, end=id_start + n_nodes)}
        id_start += n_nodes

    graph.create_formats_()
    print(graph)


def process_hg(args, graph):
    graph = graph[0]
    etype0, etype1 = etypes[args.dataset]
    ntype0, ntype1 = ntypes[args.dataset]

    n_core_edges = graph.number_of_edges(etype0)

    edges0 = torch.stack(graph.all_edges(etype=etype0)).transpose(0, 1)  # n_core_edges x 2
    # edges1 = torch.vstack(graph.all_edges(etype=etype1)).transpose(0, 1)

    # split edges
    perm = torch.randperm(n_core_edges)
    train_bound = int(args.train_rate * n_core_edges)
    val_bound = int((args.train_rate + args.val_rate) * n_core_edges)

    train_idx = perm[:train_bound]
    val_idx = perm[train_bound:val_bound]
    test_idx = perm[val_bound:]

    train_pos_edges = edges0[train_idx]
    val_pos_edges = edges0[val_idx]
    test_pos_edges = edges0[test_idx]

    # remove positive edges
    graph.remove_edges(torch.cat([val_idx, test_idx]), etype0)
    if etype0 != etype1:
        graph.remove_edges(torch.cat([val_idx, test_idx]), etype1)

    # sample negative edges
    def sample_non_existing_edges(existing_edges, n_nodes0, n_nodes1, n):
        s = set()

        for i in range(len(existing_edges)):
            u, v = existing_edges[i, 0].item(), existing_edges[i, 1].item()
            s.add((u, v))

        if ntype0 == ntype1:
            for i in range(graph.number_of_nodes()):
                s.add((i, i))

        res = []
        while len(res) < n:
            u = random.randint(0, n_nodes0 - 1)
            v = random.randint(0, n_nodes1 - 1)
            if (u, v) in s:
                continue
            s.add((u, v))  # avoid duplicate edges
            res.append((u, v))

        return torch.tensor(res)

    non_existing_edges = sample_non_existing_edges(
        edges0, graph.number_of_nodes(ntype0), graph.number_of_nodes(ntype1), len(val_pos_edges) + len(test_pos_edges),
    )

    val_neg_edges = non_existing_edges[: len(val_pos_edges)]
    test_neg_edges = non_existing_edges[len(val_pos_edges) :]

    # re-index
    for edges in [train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges]:
        if ntype0 != "_N":
            edges[:, 0] = graph.ndata["_GLOBAL_ID"][ntype0][edges[:, 0]]
            edges[:, 1] = graph.ndata["_GLOBAL_ID"][ntype1][edges[:, 1]]
        else:
            edges[:, 0] = graph.ndata["_GLOBAL_ID"][edges[:, 0]]
            edges[:, 1] = graph.ndata["_GLOBAL_ID"][edges[:, 1]]

    print("#train positive edges:", len(train_pos_edges))
    print("#val positive edges:", len(val_pos_edges))
    print("#val negative edges:", len(val_neg_edges))
    print("#test positive edges:", len(test_pos_edges))
    print("#test negative edges:", len(test_neg_edges))

    train_edges = torch.cat([train_pos_edges, torch.ones(train_pos_edges.shape[0], 1, dtype=torch.long)], dim=1)
    val_edges = torch.cat(
        [
            torch.cat([val_pos_edges, torch.ones(val_pos_edges.shape[0], 1, dtype=torch.long)], dim=1),
            torch.cat([val_neg_edges, torch.zeros(val_neg_edges.shape[0], 1, dtype=torch.long)], dim=1),
        ],
        dim=0,
    )
    test_edges = torch.cat(
        [
            torch.cat([test_pos_edges, torch.ones(test_pos_edges.shape[0], 1, dtype=torch.long)], dim=1),
            torch.cat([test_neg_edges, torch.zeros(test_neg_edges.shape[0], 1, dtype=torch.long)], dim=1),
        ],
        dim=0,
    )

    return graph, train_edges, val_edges, test_edges


def process_kg(args, data):
    graph, (train_src, train_dst), (val_src, val_dst), (test_src, test_dst), train_rel, val_rel, test_rel = data
    n_rels = train_rel.max().item() + 1

    n_edges = graph.number_of_edges()

    # print(train_src)
    # print(train_dst)
    # print(torch.vstack([train_src, train_dst]).transpose(0, 1))
    edges0 = train_pos_edges = torch.cat(
        [
            torch.vstack([train_src, train_dst]).transpose(0, 1),
            torch.vstack([val_src, val_dst]).transpose(0, 1),
            torch.vstack([test_src, test_dst]).transpose(0, 1),
        ],
        dim=0,
    )  # n_edges x 2
    # edges1 = torch.vstack(graph.all_edges(etype=etype1)).transpose(0, 1)

    # split edges
    train_pos_edges = torch.vstack([train_src, train_rel, train_dst]).transpose(0, 1)
    val_pos_edges = torch.vstack([val_src, val_rel, val_dst]).transpose(0, 1)
    test_pos_edges = torch.vstack([test_src, test_rel, test_dst]).transpose(0, 1)

    # sample negative edges
    def sample_non_existing_edges(existing_edges, n_nodes0, n_nodes1, n, n_rels):
        s = set()

        for i in range(len(existing_edges)):
            u, v = existing_edges[i, 0].item(), existing_edges[i, 1].item()
            s.add((u, v))

        for i in range(graph.number_of_nodes()):
            s.add((i, i))

        res = []
        while len(res) < n:
            u = random.randint(0, n_nodes0 - 1)
            v = random.randint(0, n_nodes1 - 1)
            if (u, v) in s:
                continue
            rel = random.randint(0, n_rels)
            s.add((u, v))  # avoid duplicate edges
            res.append((u, rel, v))

        return torch.tensor(res)

    non_existing_edges = sample_non_existing_edges(
        edges0, graph.number_of_nodes(), graph.number_of_nodes(), len(val_pos_edges) + len(test_pos_edges), n_rels
    )

    val_neg_edges = non_existing_edges[: len(val_pos_edges)]
    test_neg_edges = non_existing_edges[len(val_pos_edges) :]

    # re-index
    for edges in [train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges]:
        edges[:, 0] = graph.ndata["_GLOBAL_ID"][edges[:, 0]]
        edges[:, -1] = graph.ndata["_GLOBAL_ID"][edges[:, -1]]

    print("#train positive edges:", len(train_pos_edges))
    print("#val positive edges:", len(val_pos_edges))
    print("#val negative edges:", len(val_neg_edges))
    print("#test positive edges:", len(test_pos_edges))
    print("#test negative edges:", len(test_neg_edges))

    train_edges = torch.cat([train_pos_edges, torch.ones(train_pos_edges.shape[0], 1, dtype=torch.long)], dim=1)
    val_edges = torch.cat(
        [
            torch.cat([val_pos_edges, torch.ones(val_pos_edges.shape[0], 1, dtype=torch.long)], dim=1),
            torch.cat([val_neg_edges, torch.zeros(val_neg_edges.shape[0], 1, dtype=torch.long)], dim=1),
        ],
        dim=0,
    )
    test_edges = torch.cat(
        [
            torch.cat([test_pos_edges, torch.ones(test_pos_edges.shape[0], 1, dtype=torch.long)], dim=1),
            torch.cat([test_neg_edges, torch.zeros(test_neg_edges.shape[0], 1, dtype=torch.long)], dim=1),
        ],
        dim=0,
    )

    return graph, train_edges, val_edges, test_edges


def process_nc(args, data):
    graph, categories, num_classes, labels, train_idx, val_idx, test_idx = data

    train_data = torch.cat([train_idx.reshape(-1, 1), labels[train_idx].reshape(-1, 1)], dim=1)
    val_data = torch.cat([val_idx.reshape(-1, 1), labels[val_idx].reshape(-1, 1)], dim=1)
    test_data = torch.cat([test_idx.reshape(-1, 1), labels[test_idx].reshape(-1, 1)], dim=1)
    # print(train_data)
    print(categories)
    return graph, categories, train_data, val_data, test_data


def save_data(args, data):
    os.makedirs("./output", exist_ok=True)
    torch.save(
        data, f"./output/{args.dataset}_split.pt",
    )


def main():
    argparser = argparse.ArgumentParser("GraphANGEL preprocess", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=["lastfm", "douban_book", "douban_movie", "amazon", "yelp", "fb15k237", "wn18", "AIFB", "AM", "acm"],
        default="lastfm",
        help="dataset",
    )
    argparser.add_argument("--train-rate", type=float, default=0.6, help="train rate")
    argparser.add_argument("--val-rate", type=float, default=0.2, help="val rate")
    args = argparser.parse_args()

    if not 0 <= args.train_rate + args.val_rate <= 1:
        raise ValueError("please make sure 0 <= train_rate + val_rate <= 1")

    if args.dataset in ["fb15k237", "wn18"]:
        args.data_type = "kg"
    elif args.dataset in ["lastfm", "douban_book", "douban_movie", "amazon", "yelp"]:
        args.data_type = "hg"
    else:
        args.data_type = "nc"

    seed(args.seed)

    data = load_data(args)
    preprocess(data[0])

    if args.data_type == "hg":
        data = process_hg(args, data)
    elif args.data_type == "kg":
        data = process_kg(args, data)
    elif args.data_type == "nc":
        data = process_nc(args, data)
    else:
        assert False

    save_data(args, data)


if __name__ == "__main__":
    main()
