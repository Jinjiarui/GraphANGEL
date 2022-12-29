import argparse
import math
import os
import random
import time

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from config import etypes, ntypes
from models import GAT, GCN

device = None


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
    with open(f"output/{args.dataset}_split.pt", "rb") as f:
        graph, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges = torch.load(f)

    return graph, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges


def preprocess(graph, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges):
    graph.create_formats_()
    print("number of nodes:", graph.number_of_nodes())
    print("node types:", graph.ntypes)
    graph_homo = dgl.to_homogeneous(graph)
    return graph, graph_homo, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges


def gen_model(args):
    # model = GCN(
    #     in_feats=args.embedding_size,
    #     n_hidden=args.n_hidden,
    #     n_classes=args.embedding_size,
    #     n_layers=args.n_layers,
    #     activation=F.relu,
    #     input_drop=args.input_drop,
    #     dropout=args.dropout,
    #     use_linear=True,
    # )
    model = GAT(
        in_feats=args.embedding_size,
        n_classes=args.embedding_size,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        edge_drop=args.edge_drop,
        use_attention=False,
        use_symmetric_norm=True,
        allow_zero_in_degree=True,
    )

    return model


def sample_non_existing_edges(excluded_edges, n_nodes0, n_nodes1, n):
    res = []
    while True:
        rand_cnt = n - len(res) + 100
        us = np.random.randint(0, n_nodes0 - 1, size=rand_cnt)
        vs = np.random.randint(0, n_nodes1 - 1, size=rand_cnt)
        for u, v in zip(us, vs):
            # u = random.randint(0, n_nodes0 - 1)
            # v = random.randint(0, n_nodes1 - 1)
            if (u, v) in excluded_edges:
                continue
            # s.add((u, v))  # avoid duplicate edges
            res.append((u, v))

            if len(res) >= n:
                return torch.tensor(res)

    # return torch.tensor(res)


def train(args, model, graph, graph_homo, embedding, train_pos_edges, excluded_edges_set, optimizer, evaluator):
    model.train()
    ntype0, ntype1 = ntypes[args.dataset]

    # sample negative train edges
    # tic = time.time()
    train_neg_edges = sample_non_existing_edges(
        excluded_edges_set, graph.number_of_nodes(ntype0), graph.number_of_nodes(ntype1), len(train_pos_edges)
    )
    # print(time.time() - tic)

    # re-index
    # global_idx_0 = graph.ndata["_GLOBAL_ID"][ntype0]
    # global_idx_1 = graph.ndata["_GLOBAL_ID"][ntype1]
    # for edges in [train_neg_edges]:
    #     edges[:, 0] = global_idx_0[edges[:, 0]]
    #     edges[:, 1] = global_idx_1[edges[:, 1]]
    for edges in [train_neg_edges]:
        # print(edges[:, 0].max())
        # print(edges[:, 1].max())
        # print(graph.ndata["_GLOBAL_ID"][ntype1].shape)
        edges[:, 0] = graph.ndata["_GLOBAL_ID"][ntype0][edges[:, 0]]
        edges[:, 1] = graph.ndata["_GLOBAL_ID"][ntype1][edges[:, 1]]

    # shuffle
    # train_edges = torch.cat([train_pos_edges, train_neg_edges])
    # y = torch.cat([torch.ones(len(train_pos_edges)), torch.zeros(len(train_neg_edges))])
    # perm = torch.randperm(len(train_edges))
    # train_edges = train_edges[perm]
    # y = y[perm]

    pred = model(graph_homo, embedding)
    loss, acc = evaluator(pred, train_pos_edges, train_neg_edges)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc


@torch.no_grad()
def evaluate(
    args, model, graph_homo, embedding, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges, evaluator
):
    model.eval()

    pred = model(graph_homo, embedding)
    val_loss, val_acc = evaluator(pred, val_pos_edges, val_neg_edges)
    test_loss, test_acc = evaluator(pred, test_pos_edges, test_neg_edges)

    return val_loss.item(), test_loss.item(), val_acc, test_acc, pred


def plot_learning_curves(args, arrays, labels, filename, yticks=None, y_major_locator=0.1):
    fig = plt.figure(figsize=(24, 24))
    ax = fig.gca()
    ax.set_xticks(np.arange(0, args.n_epochs, 100))
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(labeltop=True, labelright=True)
    for y, label in zip(arrays, labels):
        plt.plot(range(1, args.n_epochs + 1), y, label=label, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which="major", color="red", linestyle="dotted")
    plt.grid(which="minor", color="orange", linestyle="dotted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)


def run(
    args, graph, graph_homo, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges, n_running
):
    def evaluator(y, pos_edges, neg_edges):
        m = y.shape[-1] // 2
        pos_u_y0 = y[pos_edges[:, 0]][:, :m]
        pos_u_y1 = y[pos_edges[:, 0]][:, m:]
        pos_v_y0 = y[pos_edges[:, 1]][:, :m]
        pos_v_y1 = y[pos_edges[:, 1]][:, m:]
        neg_u_y0 = y[neg_edges[:, 0]][:, :m]
        neg_u_y1 = y[neg_edges[:, 0]][:, m:]
        neg_v_y0 = y[neg_edges[:, 1]][:, :m]
        neg_v_y1 = y[neg_edges[:, 1]][:, m:]

        # 1: complex number, for undirected graph
        # pos_logits = torch.mean(pos_u_y0 * pos_v_y0 - pos_u_y1 * pos_v_y1, dim=-1)
        # neg_logits = torch.mean(neg_u_y0 * neg_v_y0 - neg_u_y1 * neg_v_y1, dim=-1)
        # 2: real number
        pos_logits = torch.mean(pos_u_y0 * pos_v_y0 + pos_u_y1 * pos_v_y1, dim=-1)
        neg_logits = torch.mean(neg_u_y0 * neg_v_y0 + neg_u_y1 * neg_v_y1, dim=-1)
        # 3: SVD-like, for directed graph
        # pos_logits = torch.mean(pos_u_y0 * pos_v_y1, dim=-1)
        # neg_logits = torch.mean(neg_u_y0 * neg_v_y1, dim=-1)
        # 4: SVD-like, for undirected graph
        # pos_logits = torch.mean(pos_u_y0 * pos_v_y1 + pos_u_y1 * pos_v_y0, dim=-1)
        # neg_logits = torch.mean(neg_u_y0 * neg_v_y1 + neg_u_y1 * neg_v_y0, dim=-1)

        pos_loss = torch.mean(F.softplus(-pos_logits))
        neg_loss = torch.mean(F.softplus(neg_logits))
        loss = (pos_loss + neg_loss) / 2

        pos_acc = torch.mean((pos_logits.detach() >= 0).to(torch.float32)).item()
        neg_acc = torch.mean((neg_logits.detach() < 0).to(torch.float32)).item()
        acc = (pos_acc + neg_acc) / 2
        # print("acc", pos_acc, neg_acc, acc)

        return loss, acc

    # preprocess
    excluded_edges = torch.cat([train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges])
    excluded_edges_set = set()
    for i in range(len(excluded_edges)):
        u, v = excluded_edges[i, 0].item(), excluded_edges[i, 1].item()
        excluded_edges_set.add((u, v))

    # define embedding, model and optimizer
    embedding = (
        (torch.rand(size=(graph.number_of_nodes(), args.embedding_size), device=device) * 2 - 1) * math.sqrt(3)
    ).requires_grad_()  # [-sqrt(3), sqrt(3))
    model = gen_model(args).to(device)
    optimizer = optim.AdamW(set(model.parameters()) | {embedding}, lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    final_pred = None
    best_val_acc, final_test_acc = 0, 0

    accs, val_accs, test_accs = [], [], []
    losses, val_losses, test_losses = [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        loss, acc = train(
            args, model, graph, graph_homo, embedding, train_pos_edges, excluded_edges_set, optimizer, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        val_loss, test_loss, val_acc, test_acc, pred = evaluate(
            args, model, graph_homo, embedding, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges, evaluator,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            print(
                f"Runs: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, val_accs, test_accs, losses, val_losses, test_losses],
            [acc, val_acc, test_acc, loss, val_loss, test_loss],
        ):
            l.append(e)

    os.makedirs("./output", exist_ok=True)
    torch.save(F.softmax(final_pred, dim=1), f"./output/{args.dataset}_embedding_{n_running}.pt")

    if args.plot_curves:
        plot_learning_curves(
            args,
            [accs, val_accs, test_accs],
            ["train acc", "val acc", "test acc"],
            f"pretrain_{args.dataset}_acc_{n_running}.png",
            yticks=np.linspace(0, 1.0, 101),
            y_major_locator=0.01,
        )
        plot_learning_curves(
            args,
            [losses, val_losses, test_losses],
            ["loss", "val loss", "test loss"],
            f"pretrain_{args.dataset}_loss_{n_running}.png",
            y_major_locator=0.1,
        )


def main():
    global device

    argparser = argparse.ArgumentParser("GraphANGEL pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=["lastfm", "douban_book", "douban_movie", "amazon", "yelp"],
        default="lastfm",
        help="dataset",
    )
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--embedding-size", type=int, default=192, help="number of embedding size")
    argparser.add_argument("--n-hidden", type=int, default=192, help="number of hidden units")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=1, help="number of heads")
    argparser.add_argument("--input_drop", type=float, default=0.6, help="dropout rate")
    argparser.add_argument("--edge_drop", type=float, default=0.3, help="edge drop rate")
    argparser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
    argparser.add_argument("--log-every", type=int, default=25, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", action="store_true", help="plot learning curves")
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    graph, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges = load_data(args)
    graph, graph_homo, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges = preprocess(
        graph, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges
    )

    graph, graph_homo, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges = map(
        lambda x: x.to(device),
        (graph, graph_homo, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges),
    )

    # for i, param in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    # args.input_drop = param
    seed(args.seed)
    run(args, graph, graph_homo, train_pos_edges, val_pos_edges, val_neg_edges, test_pos_edges, test_neg_edges, 0)


if __name__ == "__main__":
    main()
