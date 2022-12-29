import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MeanAggregator, ConcatAggregator, CrossAggregator

class Angel(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, params_for_paths):
        super(Angel, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, params_for_paths)
        self._build_model()

    def _parse_args(self, args, n_relations, params_for_neighbors, params_for_paths):
        self.n_relations = n_relations
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim

        # for each subgraph, we aggregate the node embeddings
        self.use_neighbor = args.use_neighbor
        if self.use_neighbor:
            self.entity2edges = torch.LongTensor(params_for_neighbors[0]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[0])
            self.edge2entities = torch.LongTensor(params_for_neighbors[1]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[1])
            self.edge2relation = torch.LongTensor(params_for_neighbors[2]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[2])
            self.neighbor_samples = args.neighbor_samples
            self.neighbor_hops = args.neighbor_hops
            if args.neighbor_agg == 'mean':
                self.neighbor_agg = MeanAggregator
            elif args.neighbor_agg == 'concat':
                self.neighbor_agg = ConcatAggregator
            elif args.neighbor_agg == 'cross':
                self.neighbor_agg = CrossAggregator
            else:
                print("=== Please Use: mean, concat, cross ===")
                raise NotImplementedError

        # for each analogy subgraph, we use MLP or Attention to aggregate
        self.use_analogy = args.use_analogy
        if self.use_analogy:
            self.path_type = args.path_type
            self.use_sample = args.use_sample
            if self.path_type == 'mlp' or 'att' or 'mean':
                if self.use_sample:
                    self.max_path_len = args.max_path_len
                    self.path_samples = args.path_samples
                    self.path_agg = args.path_agg
                    self.id2path = torch.LongTensor(params_for_paths[0]).cuda() if args.cuda \
                            else torch.LongTensor(params_for_paths[0])
                    self.id2length = torch.LongTensor(params_for_paths[1]).cuda() if args.cuda \
                            else torch.LongTensor(params_for_paths[1])
                else:            
                    self.n_paths = params_for_paths[0]
            else:
                print("=== Please Use: mlp, att, mean")
                raise NotImplementedError

    def _build_model(self):
        # define initial relation features
        if self.use_neighbor or self.use_analogy:
            self._build_relation_feature()

        self.scores = 0.0

        if self.use_neighbor:
            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())  # define aggregators for each layer

        if self.use_analogy:
            if self.use_sample:
                self.layer = nn.Linear(self.n_relations, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)
            else:
                self.layer = nn.Linear(self.n_paths, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch):
        if self.use_neighbor:
            self.entity_pairs = batch['entity_pairs']
            self.train_edges = batch['train_edges']

        if self.use_analogy:
            if self.use_sample:
                self.path_ids = batch['path_ids']
            else:
                self.path_features = batch['path_features']

        self.labels = batch['labels']

        self._call_model()

    def _call_model(self):
        self.scores = 0.

        if self.use_neighbor:
            edge_list, mask_list = self._get_neighbors_and_masks(self.labels, self.entity_pairs, self.train_edges)
            self.aggregated_neighbors = self._aggregate_neighbors(edge_list, mask_list) # [batch_size, n_relations]
            self.scores += self.aggregated_neighbors

        if self.use_analogy:
            path_features = self._get_sampled_path_feature(self.path_ids) if self.use_sample else self.path_features 
            if self.path_type == 'mlp':
                self.scores += self.layer(path_features) # [batch_size, n_paths] -> [batch_size, n_relations]
            elif self.path_type == 'att' or 'mean':
                self.scores += self._att(path_features) # [batch_size, path_samples * max_path_len, n_relations] -> [batch_size, n_relations]

        self.scores_normalized = F.sigmoid(self.scores)

    def _build_relation_feature(self):
        self.relation_dim = self.n_relations
        self.relation_features = torch.eye(self.n_relations).cuda() if self.use_gpu \
                    else torch.eye(self.n_relations)
        # the feature of the last relation (the null relation) is a zero vector
        self.relation_features = torch.cat([self.relation_features, 
                        torch.zeros([1, self.relation_dim]).cuda() if self.use_gpu \
                            else torch.zeros([1, self.relation_dim])], dim=0)

    def _get_neighbors_and_masks(self, relations, entity_pairs, train_edges):
        edges_list = [relations]
        masks = []
        train_edges = torch.unsqueeze(train_edges, -1)  # [batch_size, 1]

        for i in range(self.neighbor_hops):
            if i == 0:
                neighbor_entities = entity_pairs
            else:
                neighbor_entities = torch.index_select(self.edge2entities, 0, 
                            edges_list[-1].view(-1)).view([self.batch_size, -1])
            neighbor_edges = torch.index_select(self.entity2edges, 0, 
                            neighbor_entities.view(-1)).view([self.batch_size, -1])
            edges_list.append(neighbor_edges)

            mask = neighbor_edges - train_edges  # [batch_size, -1]
            mask = (mask != 0).float()
            masks.append(mask)
        return edges_list, masks

    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators

        if self.neighbor_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=F.relu))
            # middle layers
            for i in range(self.neighbor_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=F.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        return aggregators

    def _aggregate_neighbors(self, edge_list, mask_list):
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [torch.index_select(self.relation_features, 0, edge_list[0])]
        for edges in edge_list[1:]:
            relations = torch.index_select(self.edge2relation, 0, 
                            edges.view(-1)).view(list(edges.shape)+[-1])
            edge_vectors.append(torch.index_select(self.relation_features, 0, 
                            relations.view(-1)).view(list(relations.shape)+[-1]))

        # shape of edge vectors:
        # [[batch_size, relation_dim],
        #  [batch_size, 2 * neighbor_samples, relation_dim],
        #  [batch_size, (2 * neighbor_samples) ^ 2, relation_dim],
        #  ...]

        for i in range(self.neighbor_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            neighbors_shape = [self.batch_size, -1, 2, self.neighbor_samples, aggregator.input_dim]
            masks_shape = [self.batch_size, -1, 2, self.neighbor_samples, 1]

            for hop in range(self.neighbor_hops - i):
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=edge_vectors[hop + 1].view(neighbors_shape),
                                    masks=mask_list[hop].view(masks_shape))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
        res = edge_vectors[0].view([self.batch_size, self.n_relations])
        return res
    
    def _get_sampled_path_feature(self, path_ids):
        path_ids = path_ids.view([self.batch_size * self.path_samples])  # [batch_size * path_samples]
        paths = torch.index_select(self.id2path, 0, 
                path_ids.view(-1)).view(list(path_ids.shape)+[-1])  # [batch_size * path_samples, max_path_len]
        # [batch_size * path_samples, max_path_len, relation]
        path_features = torch.index_select(self.relation_features, 0, 
                paths.view(-1)).view(list(paths.shape)+[-1])
        lengths = torch.index_select(self.id2length, 0, path_ids)  # [batch_size * path_samples]
        return path_features.view(self.batch_size, self.path_samples*self.max_path_len, self.n_relations) # [batch_size, path_samples * max_path_len, n_relations]

    def _att(self, path_features):
        # path_features: [batch_size, path_samples * max_path_len, n_relations]
        if self.path_type == 'mean':
            output = torch.mean(path_features, dim=1) # [batch_size, n_relations]
        else:
            aggregated_neighbors = self.aggregated_neighbors.unsqueeze(1)  # [batch_size, 1, n_relations]
            attention_weights = torch.sum(aggregated_neighbors * path_features, dim=-1)  # [batch_size, path_samples * max_path_len]
            attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, path_samples * max_path_len]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, path_samples * max_path_len, 1]
            path_features = self.layer(path_features)  # [batch_size, path_samples * max_path_len, n_relations]
            output = torch.sum(attention_weights * path_features, dim=1)  # [batch_size, n_relations]

        return output # [batch_size, n_relations]

    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores, model.labels))
        loss.backward()
        optimizer.step()

        return loss.item()
    
    @staticmethod
    def test_step(model, batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            acc = (model.labels == model.scores.argmax(dim=1)).float().tolist()
        return acc, model.scores_normalized.tolist()