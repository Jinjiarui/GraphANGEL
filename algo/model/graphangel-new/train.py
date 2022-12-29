import torch
import numpy as np
from collections import defaultdict
from model import Angel
from utils import sparse_to_tuple


args = None


def train(model_args, data):
    global args, model, sess
    args = model_args

    # extract data
    triplets, paths, n_relations, neighbor_params, path_params = data

    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32))
    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))

    train_paths, valid_paths, test_paths = paths

    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    # define the model
    model = Angel(args, n_relations, neighbor_params, path_params)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_neighbor:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit10

    print('=== START TRAINING ===')

    for step in range(args.epoch):

        # shuffle training data
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_neighbor:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
        if args.use_analogy:
            train_paths = train_paths[index]
        train_labels = train_labels[index]

        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
            s += args.batch_size

        # evaluation
        print('EPOCH %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
        current_res = 'ACC: %.4f' % test_acc
        print('TRAIN ACC: %.4f   VALID ACC: %.4f   TEST ACC: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit10 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   MRR: %.4f   MR: %.4f   H1: %.4f   H3: %.4f   H10: %.4f' % (mrr, mr, hit1, hit3, hit10)
        print('           MRR: %.4f   MR: %.4f   H1: %.4f   H3: %.4f   H10: %.4f' % (mrr, mr, hit1, hit3, hit10))
        print()

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('=== FINAL RESULTS \n%s ===' % final_res)


def get_feed_dict(entity_pairs, train_edges, paths, labels, start, end):
    feed_dict = {}

    if args.use_neighbor:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
        else:
            # for evaluation no edges should be masked out
            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                        else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    if args.use_analogy:
        if args.use_sample:
            feed_dict["path_ids"] = torch.LongTensor(paths[start:end]).cuda() if args.cuda \
                    else torch.LongTensor(paths[start:end])
        else:
            indices, values, shape = sparse_to_tuple(paths[start:end])
            indices = torch.LongTensor(indices).cuda() if args.cuda else torch.LongTensor(indices)
            values = torch.Tensor(values).cuda() if args.cuda else torch.Tensor(values)
            feed_dict["path_features"] = torch.sparse.FloatTensor(indices.t(), values, torch.Size(shape)).to_dense()            

    feed_dict["labels"] = labels[start:end]

    return feed_dict


def evaluate(entity_pairs, paths, labels):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, paths, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        s += args.batch_size

    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit10 = float(np.mean(rankings <= 10))

    return mrr, mr, hit1, hit3, hit10
