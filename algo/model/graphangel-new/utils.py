import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from collections import defaultdict
from abc import abstractmethod

# paths: pair, triangle, square: determined by max_path_len
def retrieve_all_paths_with_mp(e2re, max_path_len, head2tails):
    n_cores, pool, range_list = get_params_for_mp(len(head2tails))
    results = pool.map(count_all_paths, zip([e2re] * n_cores,
                                            [max_path_len] * n_cores,
                                            [head2tails[i[0]:i[1]] for i in range_list],
                                            range(n_cores)))
    res = defaultdict(set)
    for ht2paths in results:
        res.update(ht2paths)

    return res


def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list


# input: [(h1, {t1, t2 ...}), (h2, {t3 ...}), ...]
# output: {(h1, t1): paths, (h1, t2): paths, (h2, t3): paths, ...}
def retrieve_all_paths(inputs):
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
    #     print('pid %d:  %d / %d' % (pid, i, len(head2tails)))
    # print('pid %d  done' % pid)
    return ht2paths


def bfs(head, tails, e2re, max_path_len):
    # put length-1 paths into all_paths
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head]]

    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            last_entity_in_path = path[-1][1]
            entities_in_path = set([head] + [i[1] for i in path])
            for edge in e2re[last_entity_in_path]:
                # append (relation, entity) to the path if the new entity does not appear in this path before
                if edge[1] not in entities_in_path:
                    all_paths.append(path + [edge])
            p += 1

    ht2paths = defaultdict(set)
    for path in all_paths:
        tail = path[-1][1]
        if tail in tails:  # if this path ends at tail
            ht2paths[(head, tail)].add(tuple([i[0] for i in path]))

    return ht2paths


def retrieve_paths(triplets, ht2paths, train_set):
    res = []

    for head, tail, relation in triplets:
        path_set = ht2paths[(head, tail)]
        if (tail, head, relation) in train_set:
            path_list = list(path_set)
        else:
            path_list = list(path_set - {tuple([relation])})
        res.append([list(i) for i in path_list])

    return res


def get_path_dict_and_length(train_paths, valid_paths, test_paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    for paths_of_triplet in train_paths + valid_paths + test_paths:
        for path in paths_of_triplet:
            path_tuple = tuple(path)
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths
                id2length.append(len(path))
                id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                n_paths += 1
    return path2id, id2path, id2length


def one_hot_path_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    for data in (train_paths, valid_paths, test_paths):
        bop_list = []  # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)

    return [get_sparse_feature_matrix(bop_list, len(path_dict)) for bop_list in res]


def sample_paths(train_paths, valid_paths, test_paths, path_dict, path_samples):
    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = [path_dict[tuple(path)] for path in paths]
            sampled_path_ids_for_triplet = np.random.choice(
                path_ids_for_triplet, size=path_samples, replace=len(path_ids_for_triplet) < path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data, dtype=np.int32)
        res.append(path_ids_for_data)
    return res


def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape


class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act, self_included):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included

    def forward(self, self_vectors, neighbor_vectors, masks):
        # self_vectors: [batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        entity_vectors = torch.mean(neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 2, input_dim]
        outputs = self._call(self_vectors, entity_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        pass


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        self.layer = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = torch.mean(entity_vectors, dim=-2)  # [batch_size, -1, input_dim]
        if self.self_included:
            output += self_vectors
        output = output.view([-1, self.input_dim])  # [-1, input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        multiplier = 3 if self_included else 2

        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = entity_vectors.view([-1, self.input_dim * 2])  # [-1, input_dim * 2]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * 3]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class CrossAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(CrossAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        addition = self.input_dim if self.self_included else 0

        self.layer = nn.Linear(self.input_dim * self.input_dim + addition, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        # [batch_size, -1, 1, input_dim]
        entity_vectors_a, entity_vectors_b = torch.chunk(entity_vectors, 2, dim=-2)
        entity_vectors_a = entity_vectors_a.view([-1, self.input_dim, 1])
        entity_vectors_b = entity_vectors_b.view([-1, 1, self.input_dim])
        output = torch.matmul(entity_vectors_a, entity_vectors_b)  # [-1, input_dim, input_dim]
        output = output.view([-1, self.input_dim * self.input_dim])  # [-1, input_dim * input_dim]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * input_dim + input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)
