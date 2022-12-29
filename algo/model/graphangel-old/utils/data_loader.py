import argparse
import math
import os
import pickle as pkl
import random
import sys
import time

import dgl
import numpy as np
import torch as th
from dgl.data import FB15k237Dataset, WN18Dataset

# for heterogeneous graph recommendation, we select 2 triangle and 4 square logics
# for knowledge graph recommendation, we select 3 triangle and 6 square logics


def lower_bound(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l


class MyDataset:
    def __init__(self):
        # source, edge, target for link predication on knowledge graph
        # source, target for link prediction heterogeneous graph
        # source for node classification
        self._datainput = []  # dataset_size x [1|2|3]
        self._label = []  # dataset_size x label_num
        # logic for each shape
        self._trianglelogic = []  # dataset_size x logic_num x 3
        self._squarelogic = []  # dataset_size x logic_num x 4
        # neighborhood store the ids for triangles
        self._triangleid = []  # dataset_size x logic_num x neighbor_graphlet_num x 3
        # mask to show whether the logic exists
        self._trianglemask = []  # dataset_size x logic_num x graphlet_num x 1
        # neighborhood store the ids for squares
        self._squareid = []  # dataset_size x logic_num x neighbor_graphlet_num x 4
        # mask to show whether the logic exists
        self._squaremask = []  # dataset_size x logic_num x graphlet_num x 1
        # analogy graphlets store the ids
        self._trianglelistid = []  # dataset_size x logic_num x graphlet_num x 3
        self._notrianglelistid = []  # dataset_size x logic_num x graphlet_num x 3
        self._squarelistid = []  # dataset_size x logic_num x graphlet_num x 4
        self._nosquarelistid = []  # dataset_size x logic_num x graphlet_num x 4
        self._len = 0

    def sample(self, index):
        # index: list
        assert max(index) < self._len - 1
        sample_datainput, sample_label = [], []
        sample_trianglelogic, sample_squarelogic = [], []
        sample_triangleid, sample_trianglemask = [], []
        sample_squareid, sample_squaremask = [], []
        sample_trianglelistid, sample_notrianglelistid = [], []
        sample_squarelistid, sample_nosquarelistid = [], []
        for _index in index:
            sample_datainput.append(self._datainput[_index])
            sample_label.append(self._label[_index])
            sample_trianglelogic.append(self._trianglelogic[_index])
            sample_triangleid.append(self._triangleid[_index])
            sample_trianglemask.append(self._trianglemask[_index])
            sample_squarelogic.append(self._squarelogic[_index])
            sample_squareid.append(self._squareid[_index])
            sample_squaremask.append(self._squaremask[_index])
            sample_trianglelistid.append(self._trianglelistid[_index])
            sample_notrianglelistid.append(self._notrianglelistid[_index])
            sample_squarelistid.append(self._squarelistid[_index])
            sample_nosquarelistid.append(self._nosquarelistid[_index])
        return (
            sample_datainput,
            sample_label,
            sample_trianglelogic,
            sample_triangleid,
            sample_trianglemask,
            sample_squarelogic,
            sample_squareid,
            sample_squaremask,
            sample_trianglelistid,
            sample_notrianglelistid,
            sample_squarelistid,
            sample_nosquarelistid,
        )

    def append(
        self,
        data_input,
        label,
        triangle_logic,
        triangle_id,
        triangle_mask,
        square_logic,
        square_id,
        square_mask,
        triangle_listid,
        notriangle_listid,
        square_listid,
        nosquare_listid,
    ):
        self._datainput.append(data_input)
        self._label.append(label)
        self._trianglelogic.append(triangle_logic)
        self._triangleid.append(triangle_id)
        self._trianglemask.append(triangle_mask)
        self._squarelogic.append(square_logic)
        self._squareid.append(square_id)
        self._squaremask.append(square_mask)
        self._trianglelistid.append(triangle_listid)
        self._notrianglelistid.append(notriangle_listid)
        self._squarelistid.append(square_listid)
        self._nosquarelistid.append(nosquare_listid)
        self._len += 1
        assert self._len == len(self._datainput) == len(self._label)

    def __len__(self):
        return self._len


class DataLoader:
    def __init__(
        self,
        name,
        data_dir,
        triangle_num,
        square_num,
        trianglelogic_num,
        squarelogic_num,
        triangle_neighbornum,
        square_neighbornum,
        embed_size,
    ):
        self._name = name
        self._path = data_dir
        # number of logic
        self._trianglelogicnum = trianglelogic_num
        self._squarelogicnum = squarelogic_num
        # graphlets in each logic
        self._trianglenum = triangle_num
        self._squarenum = square_num
        # neighborhood logic
        self._triangleneighbornum = triangle_neighbornum
        self._squareneighbornum = square_neighbornum
        # size of embedding
        self._embedsize = embed_size

    def _generate_neighborhoodtriangleid(self, sourceid, targetid, triangle, notriangle):
        # search to find neighbors of source and target nodes
        # for node classification, targetid = sourceid
        tic = time.time()
        triangleneighbor = []
        trianglemask = []

        # if targetid is not None:
        for _graphlet in triangle[lower_bound(triangle, (sourceid, targetid)) :]:
            if len(triangleneighbor) < self._triangleneighbornum:
                if sourceid == _graphlet[0] and targetid == _graphlet[2]:
                    # assert _graphlet not in triangleneighbor
                    # if _graphlet not in triangleneighbor:
                    triangleneighbor.append(_graphlet)
                    trianglemask.append([1])
                else:
                    break
            else:
                break

        for _graphlet in notriangle[lower_bound(notriangle, (sourceid, targetid)) :]:
            if len(triangleneighbor) < self._triangleneighbornum:
                if sourceid == _graphlet[0] and targetid == _graphlet[2]:
                    # assert _graphlet not in triangleneighbor
                    # if _graphlet not in triangleneighbor:
                    triangleneighbor.append(_graphlet)
                    trianglemask.append([1])
                else:
                    break
            else:
                break

        # if there is no such logic in neighbors, then add <source, source, target>, <source, source, target, target>
        # set islogic = 0 to mask the result
        if len(triangleneighbor) != self._triangleneighbornum:
            for _ in range(len(triangleneighbor), self._triangleneighbornum):
                triangleneighbor.append([sourceid, sourceid, targetid])
                trianglemask.append([0])

        assert len(trianglemask) == len(triangleneighbor) == self._triangleneighbornum

        return triangleneighbor, trianglemask

    def _generate_neighborhoodsquareid(self, sourceid, targetid, square, nosquare):
        # search to find neighbors of source and target nodes
        squareneighbor = []
        squaremask = []

        for _graphlet in square[lower_bound(square, (sourceid, targetid)) :]:
            if len(squareneighbor) < self._squareneighbornum:
                if sourceid in _graphlet and targetid in _graphlet:
                    # if _graphlet not in squareneighbor:
                    squareneighbor.append(_graphlet)
                    squaremask.append([1])
                else:
                    break
            else:
                break

        for _graphlet in nosquare[lower_bound(nosquare, (sourceid, targetid)) :]:
            if len(squaremask) < self._squareneighbornum:
                if sourceid in _graphlet and targetid in _graphlet:
                    # if _graphlet not in squareneighbor:
                    squareneighbor.append(_graphlet)
                    squaremask.append([1])
                else:
                    break
            else:
                break

        # if there is no such logic in neighbors, then add <source, source, target>, <source, source, target, target>
        # set islogic = False to mask the result
        if len(squareneighbor) < self._squareneighbornum:
            for _ in range(len(squareneighbor), self._squareneighbornum):
                squareneighbor.append([sourceid, sourceid, targetid, targetid])
                squaremask.append([0])

        assert len(squareneighbor) == self._squareneighbornum == len(squaremask)

        return squareneighbor, squaremask

    def _generate_graphlettriangleid(self, triangle, notriangle):
        trianglegraphletid, notrianglegraphletid = [], []
        # for those dataset which is no each logic
        if len(triangle) == 0:
            for _ in range(self._trianglenum):
                trianglegraphletid.append([0, 0, 0])
        else:
            index_arr = list(np.random.choice(len(triangle), self._trianglenum - len(trianglegraphletid)))
            for index in index_arr:
                trianglegraphletid.append(triangle[index])
        if len(notriangle) == 0:
            for _ in range(self._trianglenum):
                notrianglegraphletid.append([0, 0, 0])
        else:
            index_arr = list(np.random.choice(len(notriangle), self._trianglenum - len(notrianglegraphletid)))
            for index in index_arr:
                notrianglegraphletid.append(notriangle[index])
        assert len(trianglegraphletid) == len(notrianglegraphletid) == self._trianglenum
        return trianglegraphletid, notrianglegraphletid

    def _generate_graphletsquareid(self, square, nosquare):
        squaregraphletid, nosquaregraphletid = [], []
        # for those dataset which is no each logic
        if len(square) == 0:
            for _ in range(self._squarenum):
                squaregraphletid.append([0, 0, 0, 0])
        else:
            index_arr = list(np.random.choice(len(square), self._squarenum - len(squaregraphletid)))
            for index in index_arr:
                squaregraphletid.append(square[index])
        if len(nosquare) == 0:
            for _ in range(self._squarenum):
                nosquaregraphletid.append([0, 0, 0, 0])
        else:
            index_arr = list(np.random.choice(len(nosquare), self._trianglenum - len(nosquaregraphletid)))
            for index in index_arr:
                nosquaregraphletid.append(nosquare[index])
        assert len(squaregraphletid) == len(nosquaregraphletid) == self._squarenum
        return squaregraphletid, nosquaregraphletid

    def load_data(self):
        if self._name in ["lastfm", "amazon", "yelp", "douban_book"]:
            return self._load_hg(self._name)
        elif self._name in ["fb15k237", "wn18"]:
            return self._load_kg(self._name)
        elif self._name in ["AIFB", "AM", "acm"]:
            return self._load_nc(self._name)
        elif self._name == "demo":
            return self._load_demo()
        else:
            raise NotImplementedError

    def _load_demo(self):
        # nodedata: node classification, input: sourceid, label
        # hgdata: link predication on heterogeneous graph, input: sourceid, targetid, label
        # kgdata: link predication on knowledge graph, input: sourceid, relation, targetid, label
        nodedata_loader = MyDataset()
        hgdata_loader = MyDataset()
        kgdata_loader = MyDataset()
        # graph is on the document
        # heterogeneous graph
        graph_data = {
            ("player", "pc", "club"): (np.array([0, 0, 1, 3]), np.array([2, 3, 2, 4])),
            ("club", "cp", "player"): (np.array([0, 1, 2, 3]), np.array([0, 0, 4, 2])),
            ("player", "pp", "player"): (np.array([2, 3, 2, 4]), np.array([0, 0, 1, 3])),
            ("player", "pt", "city"): (np.array([0, 1, 2, 3, 4]), np.array([0, 0, 2, 1, 1])),
            ("city", "tp", "player"): (np.array([0, 0, 2, 1, 1]), np.array([0, 1, 2, 3, 4])),
            ("club", "ct", "city"): (np.array([0, 1, 2, 3]), np.array([0, 1, 1, 2])),
            ("city", "tc", "club"): (np.array([0, 1, 1, 2]), np.array([0, 1, 2, 3])),
        }
        graph_data = dgl.heterograph(graph_data)
        # graph embedding
        node_num = graph_data.num_nodes()
        node_embedding = (th.rand((node_num, self._embedsize)) * 2 - 1) * math.sqrt(3)
        # node type:
        # 0: player, 1: city, 2: club
        # graph_data.ndata['type'] = [2, 2, 2, 1, 0, 1, 0, 0, 1, 0, 0, 2]
        print("===== GRAPH SUCCESSFULLY LOADED =====")
        # logic 1: player-club-city 0-2-1
        # logic 2: player-player-city 0-0-1
        # for heterogeneous graph
        triangle_logic = [(0, 2, 1), (0, 0, 1)]
        triangle1_data = [(0, 0, 0), (2, 3, 2)]
        notriangle1_data = [(0, 1, 1), (4, 2, 1)]
        triangle2_data = []
        notriangle2_data = [(0, 3, 1), (3, 0, 0), (0, 2, 2), (2, 1, 3), (2, 0, 3)]
        _triangle_data = []
        _triangle_data.append(triangle1_data)
        _triangle_data.append(triangle2_data)
        triangle_data = dict(zip(triangle_logic, _triangle_data))
        _notriangle_data = []
        _notriangle_data.append(notriangle1_data)
        _notriangle_data.append(notriangle2_data)
        notriangle_data = dict(zip(triangle_logic, _notriangle_data))
        # for knowledge graph
        _triangle_logic = [(0, 2, 1), (0, 0, 1), (0, 1, 1)]
        _triangle3_data = [(0, 0, 0), (2, 3, 2)]
        _notriangle3_data = [(0, 1, 1), (4, 2, 1)]
        _triangle_data.append(_triangle3_data)
        _notriangle_data.append(_notriangle3_data)
        _triangle_data = dict(zip(_triangle_logic, _triangle_data))
        _notriangle_data = dict(zip(_triangle_logic, _notriangle_data))
        print("===== TRIANGLE SUCCESSFULLY LOADED =====")
        # logic 1: player-player-club-city 0-0-2-1
        # logic 2: player-city-player-city 0-1-0-1
        # logic 3: player-club-player-city 0-2-0-1
        # logic 4: player-player-player-city 0-0-0-1
        square_logic = [(0, 0, 2, 1), (0, 1, 0, 1), (0, 2, 0, 1), (0, 0, 0, 1)]
        square1_data = [(3, 4, 2, 1), (3, 0, 1, 1)]
        nosquare1_data = [(0, 2, 3, 2), (2, 0, 0, 0), (3, 0, 0, 0), (1, 2, 3, 2)]
        square2_data = [(1, 0, 1, 0), (3, 1, 3, 1), (1, 0, 0, 0), (2, 2, 2, 2), (3, 4, 2, 4), (3, 4, 3, 4)]
        nosquare2_data = []
        square3_data = [(2, 3, 2, 2), (0, 0, 0, 0)]
        nosquare3_data = [(0, 1, 0, 1)]
        square4_data = [(3, 4, 3, 1), (3, 4, 3, 1), (1, 2, 1, 0), (2, 1, 2, 2), (1, 2, 0, 0)]
        nosquare4_data = [(4, 9, 10, 5)]
        _square_data = []
        _square_data.append(square1_data)
        _square_data.append(square2_data)
        _square_data.append(square3_data)
        _square_data.append(square4_data)
        square_data = dict(zip(square_logic, _square_data))
        _nosquare_data = []
        _nosquare_data.append(nosquare1_data)
        _nosquare_data.append(nosquare2_data)
        _nosquare_data.append(nosquare3_data)
        _nosquare_data.append(nosquare4_data)
        nosquare_data = dict(zip(square_logic, _nosquare_data))
        # for knowledge graph
        _square_logic = [(0, 0, 2, 1), (0, 1, 0, 1), (0, 2, 0, 1), (0, 0, 0, 1), (0, 1, 2, 1), (0, 1, 1, 1)]
        _square5_data = [(3, 4, 2, 1), (3, 0, 1, 1)]
        _nosquare5_data = [(0, 2, 3, 2), (2, 0, 0, 0), (3, 0, 0, 0), (1, 2, 3, 2)]
        _square6_data = [(1, 0, 1, 0), (3, 1, 3, 1), (1, 0, 0, 0), (2, 2, 2, 2), (3, 4, 2, 4), (3, 4, 3, 4)]
        _nosquare6_data = []
        _square_data.append(_square5_data)
        _square_data.append(_square6_data)
        _nosquare_data.append(_nosquare5_data)
        _nosquare_data.append(_nosquare6_data)
        _square_data = dict(zip(_square_logic, _square_data))
        _nosquare_data = dict(zip(_square_logic, _nosquare_data))
        print("===== SQUARE SUCCESSFULLY LOADED =====")
        hg_data = [[0, 1, 1], [1, 0, 1], [1, 4, 1], [4, 1, 1]]
        kg_data = [[0, 0, 1, 1], [1, 0, 1, 1], [1, 0, 4, 1], [4, 0, 1, 1]]
        node_data = [[0, 1], [1, 0], [2, 0], [4, 0]]
        trianglelogic, triangleneighbor, trianglemask, squarelogic, squareneighbor, squaremask = [], [], [], [], [], []
        trianglelist, notrianglelist, squarelist, nosquarelist = [], [], [], []
        triangle_key = list(triangle_data.keys())
        notriangle_key = list(notriangle_data.keys())

        assert len(triangle_key) == len(notriangle_key)

        square_key = list(square_data.keys())
        nosquare_key = list(nosquare_data.keys())

        assert len(square_key) == len(nosquare_key)

        if len(triangle_key) < self._trianglelogicnum or len(square_key) < self._squarelogicnum:
            print("===== DECREASE LOGIC NUMBER =====")
            raise NotImplementedError

        for _data in hg_data:
            # _data: source, target, label
            source, target, label = _data
            for _logic in range(self._trianglelogicnum):
                _triangleneighbor, _trianglemask = self._generate_neighborhoodtriangleid(
                    source,
                    target,
                    triangle_data.get(triangle_key[_logic]),
                    notriangle_data.get(notriangle_key[_logic]),
                )
                triangleneighbor.append(_triangleneighbor)
                trianglemask.append(_trianglemask)
                # _trianglelist, _notrianglelist = self._generate_graphlettriangleid(graph_data, source, target, triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic]))
                _trianglelist, _notrianglelist = self._generate_graphlettriangleid(
                    triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic])
                )
                trianglelist.append(_trianglelist)
                notrianglelist.append(_notrianglelist)
                trianglelogic.append(list(triangle_key[_logic]))
            for _logic in range(self._squarelogicnum):
                _squareneighbor, _squaremask = self._generate_neighborhoodsquareid(
                    source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic])
                )
                squareneighbor.append(_squareneighbor)
                squaremask.append(_squaremask)
                # _squarelist, _nosquarelist = self._generate_graphletsquareid(graph_data, source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic]))
                _squarelist, _nosquarelist = self._generate_graphletsquareid(
                    square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic])
                )
                squarelist.append(_squarelist)
                nosquarelist.append(_nosquarelist)
                squarelogic.append(list(square_key[_logic]))
            # use list for multi-label task
            if type(label) == list:
                label = label
            elif type(label) == int:
                label = [label]
            else:
                raise NotImplementedError
            hgdata_loader.append(
                [source, target],
                label,
                trianglelogic,
                triangleneighbor,
                trianglemask,
                squarelogic,
                squareneighbor,
                squaremask,
                trianglelist,
                notrianglelist,
                squarelist,
                nosquarelist,
            )
        for _data in kg_data:
            # _data: source, relation, target, label
            source, relation, target, label = _data
            for _logic in range(self._trianglelogicnum):
                _triangleneighbor, _trianglemask = self._generate_neighborhoodtriangleid(
                    source,
                    target,
                    triangle_data.get(triangle_key[_logic]),
                    notriangle_data.get(notriangle_key[_logic]),
                )
                triangleneighbor.append(_triangleneighbor)
                trianglemask.append(_trianglemask)
                # _trianglelist, _notrianglelist = self._generate_graphlettriangleid(graph_data, source, target, triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic]))
                _trianglelist, _notrianglelist = self._generate_graphlettriangleid(
                    triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic])
                )
                trianglelist.append(_trianglelist)
                notrianglelist.append(_notrianglelist)
                trianglelogic.append(list(triangle_key[_logic]))
            for _logic in range(self._squarelogicnum):
                _squareneighbor, _squaremask = self._generate_neighborhoodsquareid(
                    source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic])
                )
                squareneighbor.append(_squareneighbor)
                squaremask.append(_squaremask)
                # _squarelist, _nosquarelist = self._generate_graphletsquareid(graph_data, source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic]))
                _squarelist, _nosquarelist = self._generate_graphletsquareid(
                    square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic])
                )
                squarelist.append(_squarelist)
                nosquarelist.append(_nosquarelist)
                squarelogic.append(list(square_key[_logic]))
            # use list for multi-label task
            if type(label) == list:
                label = label
            elif type(label) == int:
                label = [label]
            else:
                raise NotImplementedError
            kgdata_loader.append(
                [source, relation, target],
                label,
                trianglelogic,
                triangleneighbor,
                trianglemask,
                squarelogic,
                squareneighbor,
                squaremask,
                trianglelist,
                notrianglelist,
                squarelist,
                nosquarelist,
            )
        for _data in node_data:
            # _data: source, label
            source, label = _data
            for _logic in range(self._trianglelogicnum):
                _triangleneighbor, _trianglemask = self._generate_neighborhoodtriangleid(
                    source,
                    source,
                    triangle_data.get(triangle_key[_logic]),
                    notriangle_data.get(notriangle_key[_logic]),
                )
                triangleneighbor.append(_triangleneighbor)
                trianglemask.append(_trianglemask)
                # _trianglelist, _notrianglelist = self._generate_graphlettriangleid(graph_data, source, source, triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic]))
                _trianglelist, _notrianglelist = self._generate_graphlettriangleid(
                    triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic])
                )
                trianglelist.append(_trianglelist)
                notrianglelist.append(_notrianglelist)
                trianglelogic.append(list(triangle_key[_logic]))
            for _logic in range(self._squarelogicnum):
                _squareneighbor, _squaremask = self._generate_neighborhoodsquareid(
                    source, source, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic])
                )
                squareneighbor.append(_squareneighbor)
                squaremask.append(_squaremask)
                # _squarelist, _nosquarelist = self._generate_graphletsquareid(graph_data, source, source, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic]))
                _squarelist, _nosquarelist = self._generate_graphletsquareid(
                    square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic])
                )
                squarelist.append(_squarelist)
                nosquarelist.append(_nosquarelist)
                squarelogic.append(list(square_key[_logic]))
            # use list for multi-label task
            if type(label) == list:
                label = label
            elif type(label) == int:
                label = [label]
            else:
                raise NotImplementedError
            nodedata_loader.append(
                [source],
                label,
                trianglelogic,
                triangleneighbor,
                trianglemask,
                squarelogic,
                squareneighbor,
                squaremask,
                trianglelist,
                notrianglelist,
                squarelist,
                nosquarelist,
            )
        # train, test graph is not exactly same, save every time.
        with open(os.path.join(self._path, self._name + "_graph.pkl"), "wb") as graphfile:
            pkl.dump(
                [
                    self._trianglelogicnum,
                    self._squarelogicnum,
                    self._trianglenum,
                    self._triangleneighbornum,
                    self._squarenum,
                    self._squareneighbornum,
                    node_embedding.shape[1],
                ],
                graphfile,
            )
            pkl.dump(graph_data, graphfile)
            pkl.dump(node_embedding, graphfile)
        with open(os.path.join(self._path, self._name + "_hgdata.pkl"), "wb") as foutfile:
            pkl.dump(hgdata_loader, foutfile)
        with open(os.path.join(self._path, self._name + "_kgdata.pkl"), "wb") as foutfile:
            pkl.dump(kgdata_loader, foutfile)
        with open(os.path.join(self._path, self._name + "_nodedata.pkl"), "wb") as foutfile:
            pkl.dump(nodedata_loader, foutfile)
        return [hgdata_loader, kgdata_loader, nodedata_loader]

    def _load_hg(self, hg_name):
        train_loader = MyDataset()
        train_name = "_train.pkl"
        test_loader = MyDataset()
        test_name = "_test.pkl"
        eval_loader = MyDataset()
        eval_name = "_eval.pkl"
        data_loader = [train_loader, test_loader, eval_loader]
        data_name = [train_name, test_name, eval_name]
        graph_file = open(os.path.join(self._path, f"{hg_name}_hg.pkl"), "rb")
        graph_data = pkl.load(graph_file)
        graph_file.close()
        # embedding_file = open(os.path.join(self._path, "lastfm_embedding.pkl"),'rb')
        # embedding_data = pkl.load(embedding_file)
        # embedding_file.close()
        # embedding_data = th.load(os.path.join(self._path, f"{hg_name}_embedding.pt"), map_location=th.device("cpu"))
        embedding_data = (th.rand((graph_data.number_of_nodes(), self._embedsize)) * 2 - 1) * math.sqrt(3)
        print("===== GRAPH SUCCESSFULLY LOADED =====")
        triangle_file = open(os.path.join(self._path, f"{hg_name}_triangles.pkl"), "rb")
        triangle_data = pkl.load(triangle_file)
        triangle_file.close()
        notriangle_file = open(os.path.join(self._path, f"{hg_name}_neg_triangles.pkl"), "rb")
        notriangle_data = pkl.load(notriangle_file)
        notriangle_file.close()
        print("===== TRIANGLE SUCCESSFULLY LOADED =====")
        square_file = open(os.path.join(self._path, f"{hg_name}_quadrangles.pkl"), "rb")
        square_data = pkl.load(square_file)
        square_file.close()
        nosquare_file = open(os.path.join(self._path, f"{hg_name}_quadrangles.pkl"), "rb")
        nosquare_data = pkl.load(nosquare_file)
        nosquare_file.close()
        print("===== SQUARE SUCCESSFULLY LOADED =====")
        # for value in triangle_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in notriangle_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in square_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in nosquare_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        print("===== FINISGH SORTING =====")
        data_name = f"{hg_name}_split.pt"
        # data_file = open(os.path.join(self._path, data_name), "rb")
        # data = pkl.load(data_file)
        # data_file.close()
        data = th.load(os.path.join(self._path, data_name))
        triangle_key = [_key for _key in list(triangle_data.keys()) if _key in list(notriangle_data.keys())]
        # triangle_key = [_key for _key in list(triangle_data.keys())]
        # print(len(square_data.keys()), len(nosquare_data.keys()))
        square_key = [_key for _key in list(square_data.keys()) if _key in list(nosquare_data.keys())]
        # square_key = [_key for _key in list(square_data.keys())]
        #  assert len(square_key) == len(nosquare_key)
        if len(triangle_key) < self._trianglelogicnum or len(square_key) < self._squarelogicnum:
            print("===== DECREASE LOGIC NUMBER =====")
            print(len(triangle_key), len(square_key), self._trianglelogicnum, self._squarelogicnum)
            raise NotImplementedError
        data_name_list = ["train", "eval", "test"]
        for _index, _dataset in enumerate(data[1:]):
            count = 0
            for _data in _dataset:
                print(count)
                count += 1
                # initial
                trianglelogic, triangleneighbor, trianglemask, squarelogic, squareneighbor, squaremask = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                trianglelist, notrianglelist, squarelist, nosquarelist = [], [], [], []
                # _data: source, target, label
                source, target, label = _data
                source, target, label = map(lambda x: x.item(), (source, target, label))
                # label = int(label)
                for _logic in range(self._trianglelogicnum):
                    _triangleneighbor, _trianglemask = self._generate_neighborhoodtriangleid(
                        source,
                        target,
                        triangle_data.get(triangle_key[_logic]),
                        notriangle_data.get(triangle_key[_logic]),
                    )
                    triangleneighbor.append(_triangleneighbor)
                    trianglemask.append(_trianglemask)
                    # _trianglelist, _notrianglelist = self._generate_graphlettriangleid(graph_data, source, target, triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic]))
                    _trianglelist, _notrianglelist = self._generate_graphlettriangleid(
                        triangle_data.get(triangle_key[_logic]), notriangle_data.get(triangle_key[_logic])
                    )
                    trianglelist.append(_trianglelist)
                    notrianglelist.append(_notrianglelist)
                    # trianglelogic.append(list(_logic))
                    trianglelogic.append(list(triangle_key[_logic]))
                for _logic in range(self._squarelogicnum):
                    _squareneighbor, _squaremask = self._generate_neighborhoodsquareid(
                        source, target, square_data.get(square_key[_logic]), nosquare_data.get(square_key[_logic])
                    )
                    squareneighbor.append(_squareneighbor)
                    squaremask.append(_squaremask)
                    # _squarelist, _nosquarelist = self._generate_graphletsquareid(graph_data, source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic]))
                    _squarelist, _nosquarelist = self._generate_graphletsquareid(
                        square_data.get(square_key[_logic]), nosquare_data.get(square_key[_logic])
                    )
                    squarelist.append(_squarelist)
                    nosquarelist.append(_nosquarelist)
                    squarelogic.append(list(square_key[_logic]))
                # use list for multi-label task
                if type(label) == list:
                    label = label
                elif type(label) == int or type(label) == float:
                    label = [int(label)]
                else:
                    print(label)
                    raise NotImplementedError
                data_loader[_index].append(
                    [source, target],
                    label,
                    trianglelogic,
                    triangleneighbor,
                    trianglemask,
                    squarelogic,
                    squareneighbor,
                    squaremask,
                    trianglelist,
                    notrianglelist,
                    squarelist,
                    nosquarelist,
                )
            with open(os.path.join(self._path, self._name + "_" + data_name_list[_index] + ".pkl"), "wb") as foutfile:
                pkl.dump(data_loader[_index], foutfile)
        print("===== FINISGH LOADING DATA =====")
        with open(os.path.join(self._path, self._name + "_graph.pkl"), "wb") as graphfile:
            pkl.dump(
                [
                    self._trianglelogicnum,
                    self._squarelogicnum,
                    self._trianglenum,
                    self._triangleneighbornum,
                    self._squarenum,
                    self._squareneighbornum,
                    embedding_data.shape[1],
                ],
                graphfile,
            )
            pkl.dump(graph_data, graphfile)
            pkl.dump(embedding_data, graphfile)
        return data_loader

    def _load_kg(self, kg_name):
        train_loader = MyDataset()
        train_name = "_train.pkl"
        test_loader = MyDataset()
        test_name = "_test.pkl"
        eval_loader = MyDataset()
        eval_name = "_eval.pkl"
        data_loader = [train_loader, test_loader, eval_loader]
        data_name = [train_name, test_name, eval_name]
        # graph_file = open(os.path.join(self._path, f"{kg_name}_hg.pkl"), "rb")
        # graph_data = pkl.load(graph_file)
        # graph_file.close()
        if kg_name == "fb15k237":
            graph_data = FB15k237Dataset()[0]
        else:
            graph_data = WN18Dataset()[0]
        # embedding_file = open(os.path.join(self._path, "lastfm_embedding.pkl"),'rb')
        # embedding_data = pkl.load(embedding_file)
        # embedding_file.close()
        # embedding_data = th.load(os.path.join(self._path, f"{kg_name}_embedding.pt"), map_location=th.device("cpu"))
        embedding_data = (th.rand((graph_data.number_of_nodes(), self._embedsize)) * 2 - 1) * math.sqrt(3)
        print("===== GRAPH SUCCESSFULLY LOADED =====")
        triangle_file = open(os.path.join(self._path, f"{kg_name}_triangles.pkl"), "rb")
        triangle_data = pkl.load(triangle_file)
        triangle_file.close()
        notriangle_file = open(os.path.join(self._path, f"{kg_name}_neg_triangles.pkl"), "rb")
        notriangle_data = pkl.load(notriangle_file)
        notriangle_file.close()
        print("===== TRIANGLE SUCCESSFULLY LOADED =====")
        square_file = open(os.path.join(self._path, f"{kg_name}_quadrangles.pkl"), "rb")
        square_data = pkl.load(square_file)
        square_file.close()
        nosquare_file = open(os.path.join(self._path, f"{kg_name}_quadrangles.pkl"), "rb")
        nosquare_data = pkl.load(nosquare_file)
        nosquare_file.close()
        print("===== SQUARE SUCCESSFULLY LOADED =====")
        # for value in triangle_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in notriangle_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in square_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in nosquare_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        print("===== FINISGH SORTING =====")
        data_name = f"{kg_name}_split.pt"
        # data_file = open(os.path.join(self._path, data_name), "rb")
        # data = pkl.load(data_file)
        # data_file.close()
        data = th.load(os.path.join(self._path, data_name))
        triangle_key = [_key for _key in list(triangle_data.keys()) if _key in list(notriangle_data.keys())]
        print(len(square_data.keys()), len(nosquare_data.keys()))
        square_key = [_key for _key in list(square_data.keys()) if _key in list(nosquare_data.keys())]
        #  assert len(square_key) == len(nosquare_key)
        if len(triangle_key) < self._trianglelogicnum or len(square_key) < self._squarelogicnum:
            print("===== DECREASE LOGIC NUMBER =====")
            print(len(triangle_key), len(square_key), self._trianglelogicnum, self._squarelogicnum)
            raise NotImplementedError
        data_name_list = ["train", "eval", "test"]
        for _index, _dataset in enumerate(data[1:]):
            count = 0
            for _data in _dataset:
                print(count)
                count += 1
                # initialize
                (
                    trianglelogic,
                    triangleneighbor,
                    trianglemask,
                    squarelogic,
                    squareneighbor,
                    squaremask,
                    trianglelist,
                    notrianglelist,
                    squarelist,
                    nosquarelist,
                ) = ([], [], [], [], [], [], [], [], [], [])
                trianglelogic, triangleneighbor, trianglemask, squarelogic, squareneighbor, squaremask = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                trianglelist, notrianglelist, squarelist, nosquarelist = [], [], [], []
                # _data: source, relation, target, label
                source, relation, target, label = _data
                source, relation, target, label = map(lambda x: x.item(), (source, relation, target, label))
                # label = int(label)
                for _logic in range(self._trianglelogicnum):
                    _triangleneighbor, _trianglemask = self._generate_neighborhoodtriangleid(
                        source,
                        target,
                        triangle_data.get(triangle_key[_logic]),
                        notriangle_data.get(triangle_key[_logic]),
                    )
                    triangleneighbor.append(_triangleneighbor)
                    trianglemask.append(_trianglemask)
                    # _trianglelist, _notrianglelist = self._generate_graphlettriangleid(graph_data, source, target, triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic]))
                    _trianglelist, _notrianglelist = self._generate_graphlettriangleid(
                        triangle_data.get(triangle_key[_logic]), notriangle_data.get(triangle_key[_logic])
                    )
                    trianglelist.append(_trianglelist)
                    notrianglelist.append(_notrianglelist)
                    trianglelogic.append(list(triangle_key[_logic]))
                for _logic in range(self._squarelogicnum):
                    _squareneighbor, _squaremask = self._generate_neighborhoodsquareid(
                        source, target, square_data.get(square_key[_logic]), nosquare_data.get(square_key[_logic])
                    )
                    squareneighbor.append(_squareneighbor)
                    squaremask.append(_squaremask)
                    # _squarelist, _nosquarelist = self._generate_graphletsquareid(graph_data, source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic]))
                    _squarelist, _nosquarelist = self._generate_graphletsquareid(
                        square_data.get(square_key[_logic]), nosquare_data.get(square_key[_logic])
                    )
                    squarelist.append(_squarelist)
                    nosquarelist.append(_nosquarelist)
                    squarelogic.append(list(square_key[_logic]))
                # use list for multi-label task
                if type(label) == list:
                    label = label
                elif type(label) == int or type(label) == float:
                    label = [int(label)]
                else:
                    print(label)
                    raise NotImplementedError
                data_loader[_index].append(
                    [source, relation, target],
                    label,
                    trianglelogic,
                    triangleneighbor,
                    trianglemask,
                    squarelogic,
                    squareneighbor,
                    squaremask,
                    trianglelist,
                    notrianglelist,
                    squarelist,
                    nosquarelist,
                )
            with open(os.path.join(self._path, self._name + "_" + data_name_list[_index] + ".pkl"), "wb") as foutfile:
                pkl.dump(data_loader[_index], foutfile)
        print("===== FINISGH LOADING DATA =====")
        with open(os.path.join(self._path, self._name + "_graph.pkl"), "wb") as graphfile:
            pkl.dump(
                [
                    self._trianglelogicnum,
                    self._squarelogicnum,
                    self._trianglenum,
                    self._triangleneighbornum,
                    self._squarenum,
                    self._squareneighbornum,
                    embedding_data.shape[1],
                ],
                graphfile,
            )
            pkl.dump(graph_data, graphfile)
            pkl.dump(embedding_data, graphfile)
        return data_loader

    def _load_nc(self, nc_name):
        train_loader = MyDataset()
        train_name = "_train.pkl"
        test_loader = MyDataset()
        test_name = "_test.pkl"
        eval_loader = MyDataset()
        eval_name = "_eval.pkl"
        data_loader = [train_loader, test_loader, eval_loader]
        data_name = [train_name, test_name, eval_name]
        # graph_file = open(os.path.join(self._path, f"{nc_name}_nc.pkl"), "rb")
        # graph_data = pkl.load(graph_file)
        # graph_file.close()
        with open(os.path.join(self._path, f"{nc_name}_data.pkl"), "rb") as f:
            graph_data, *_ = pkl.load(f)
        # embedding_file = open(os.path.join(self._path, "lastfm_embedding.pkl"),'rb')
        # embedding_data = pkl.load(embedding_file)
        # embedding_file.close()
        # embedding_data = th.load(os.path.join(self._path, f"{hg_name}_embedding.pt"), map_location=th.device("cpu"))
        embedding_data = (th.rand((graph_data.number_of_nodes(), self._embedsize)) * 2 - 1) * math.sqrt(3)
        print("===== GRAPH SUCCESSFULLY LOADED =====")
        triangle_file = open(os.path.join(self._path, f"{nc_name}_triangles.pkl"), "rb")
        triangle_data = pkl.load(triangle_file)
        triangle_file.close()
        notriangle_file = open(os.path.join(self._path, f"{nc_name}_neg_triangles.pkl"), "rb")
        notriangle_data = pkl.load(notriangle_file)
        notriangle_file.close()
        print("===== TRIANGLE SUCCESSFULLY LOADED =====")
        square_file = open(os.path.join(self._path, f"{nc_name}_quadrangles.pkl"), "rb")
        square_data = pkl.load(square_file)
        square_file.close()
        nosquare_file = open(os.path.join(self._path, f"{nc_name}_quadrangles.pkl"), "rb")
        nosquare_data = pkl.load(nosquare_file)
        nosquare_file.close()
        print("===== SQUARE SUCCESSFULLY LOADED =====")
        # for value in triangle_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in notriangle_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in square_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        # for value in nosquare_data.values():
        #     value.sort(key=lambda x: (x[0], x[-1]))
        print("===== FINISGH SORTING =====")
        data_name = f"{nc_name}_split.pt"
        # data_file = open(os.path.join(self._path, data_name), "rb")
        # data = pkl.load(data_file)
        # data_file.close()
        data = th.load(os.path.join(self._path, data_name))
        triangle_key = [_key for _key in list(triangle_data.keys()) if _key in list(notriangle_data.keys())]
        # triangle_key = [_key for _key in list(triangle_data.keys())]
        print(len(square_data.keys()), len(nosquare_data.keys()))
        square_key = [_key for _key in list(square_data.keys()) if _key in list(nosquare_data.keys())]
        # square_key = [_key for _key in list(square_data.keys())]
        #  assert len(square_key) == len(nosquare_key)
        if len(triangle_key) < self._trianglelogicnum or len(square_key) < self._squarelogicnum:
            print("===== DECREASE LOGIC NUMBER =====")
            print(len(triangle_key), len(square_key), self._trianglelogicnum, self._squarelogicnum)
            raise NotImplementedError
        data_name_list = ["train", "eval", "test"]
        for _index, _dataset in enumerate(data[2:]):
            count = 0
            for _data in _dataset:
                print(count)
                count += 1
                # initial
                trianglelogic, triangleneighbor, trianglemask, squarelogic, squareneighbor, squaremask = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                trianglelist, notrianglelist, squarelist, nosquarelist = [], [], [], []
                # _data: node, label
                node, label = _data
                node, label = map(lambda x: x.item(), (node, label))
                # label = int(label)
                for _logic in range(self._trianglelogicnum):
                    _triangleneighbor, _trianglemask = self._generate_neighborhoodtriangleid(
                        node, node, triangle_data.get(triangle_key[_logic]), notriangle_data.get(triangle_key[_logic]),
                    )
                    triangleneighbor.append(_triangleneighbor)
                    trianglemask.append(_trianglemask)
                    # _trianglelist, _notrianglelist = self._generate_graphlettriangleid(graph_data, source, target, triangle_data.get(triangle_key[_logic]), notriangle_data.get(notriangle_key[_logic]))
                    _trianglelist, _notrianglelist = self._generate_graphlettriangleid(
                        triangle_data.get(triangle_key[_logic]), notriangle_data.get(triangle_key[_logic])
                    )
                    trianglelist.append(_trianglelist)
                    notrianglelist.append(_notrianglelist)
                    # trianglelogic.append(list(_logic))
                    trianglelogic.append(list(triangle_key[_logic]))
                for _logic in range(self._squarelogicnum):
                    _squareneighbor, _squaremask = self._generate_neighborhoodsquareid(
                        node, node, square_data.get(square_key[_logic]), nosquare_data.get(square_key[_logic])
                    )
                    squareneighbor.append(_squareneighbor)
                    squaremask.append(_squaremask)
                    # _squarelist, _nosquarelist = self._generate_graphletsquareid(graph_data, source, target, square_data.get(square_key[_logic]), nosquare_data.get(nosquare_key[_logic]))
                    _squarelist, _nosquarelist = self._generate_graphletsquareid(
                        square_data.get(square_key[_logic]), nosquare_data.get(square_key[_logic])
                    )
                    squarelist.append(_squarelist)
                    nosquarelist.append(_nosquarelist)
                    squarelogic.append(list(square_key[_logic]))
                # use list for multi-label task
                if type(label) == list:
                    label = label
                elif type(label) == int or type(label) == float:
                    label = [int(label)]
                else:
                    print(label)
                    raise NotImplementedError
                data_loader[_index].append(
                    [node],
                    label,
                    trianglelogic,
                    triangleneighbor,
                    trianglemask,
                    squarelogic,
                    squareneighbor,
                    squaremask,
                    trianglelist,
                    notrianglelist,
                    squarelist,
                    nosquarelist,
                )
            with open(os.path.join(self._path, self._name + "_" + data_name_list[_index] + ".pkl"), "wb") as foutfile:
                pkl.dump(data_loader[_index], foutfile)
        print("===== FINISGH LOADING DATA =====")
        with open(os.path.join(self._path, self._name + "_graph.pkl"), "wb") as graphfile:
            pkl.dump(
                [
                    self._trianglelogicnum,
                    self._squarelogicnum,
                    self._trianglenum,
                    self._triangleneighbornum,
                    self._squarenum,
                    self._squareneighbornum,
                    embedding_data.shape[1],
                ],
                graphfile,
            )
            pkl.dump(graph_data, graphfile)
            pkl.dump(embedding_data, graphfile)
        return data_loader

    # def _generate_graphlettriangleid(self, graph, sourceid, targetid, triangle, notriangle):
    #     # use random walk to get neighbors of source and target nodes
    #     # triangle, notriangle: list
    #     nodeset = []
    #     trianglegraphlet, notrianglegraphlet = [], []
    #     # record id for graphlet sampled for following padding
    #     triangleids, notriangleids = [_id for _id in range(len(triangle))], [_id for _id in range(len(notriangle))]
    #     _trianglegraphletid, _notrianglegraphletid = [], []
    #     trianglegraphletid, notrianglegraphletid = [], []
    #     nodes = th.Tensor([sourceid]*self._walknum+[targetid]*self._walknum).cpu().to(th.int64)
    #     traces, _ = dgl.sampling.random_walk_with_restart(graph, nodes, metapath=graph.etypes*2, length=self._walklen, restart_prob=0.2)
    #     # length = self._walklen+1
    #     traces = traces.reshape([2*self._walknum*(self._walklen+1)]).tolist()
    #     for _node in traces:
    #         if _node not in nodeset:
    #             nodeset.append(_node)
    #         else:
    #             continue
    #         if len(trianglegraphlet) < self._trianglenum:
    #             for _graphlet, _id in zip(triangle, triangleids):
    #                 if _node in _graphlet:
    #                     if _graphlet not in trianglegraphlet:
    #                         trianglegraphlet.append(_graphlet)
    #                         _trianglegraphletid.append(_id)
    #         if len(notrianglegraphlet) < self._trianglenum:
    #             for _graphlet, _id in zip(notriangle, notriangleids):
    #                 if _node in _graphlet:
    #                     if _graphlet not in notrianglegraphlet:
    #                         notrianglegraphlet.append(_graphlet)
    #                         _notrianglegraphletid.append(_id)
    #     # if there are not enough graphlet, randomly sample from buffer
    #     if len(trianglegraphlet) < self._trianglenum:
    #         print("===== RAISE RANDOM WALK NUMBER =====")
    #         print("===== TRIANGLENUM: %d =====" %(self._trianglenum-len(trianglegraphlet)))
    #         for _id in triangleids:
    #             if _id not in _trianglegraphletid:
    #                 trianglegraphletid.append(_id)
    #         assert len(trianglegraphletid) == len(triangle)-len(trianglegraphlet)
    #         if len(trianglegraphletid) > 0 and len(trianglegraphlet) < self._trianglenum:
    #             index_arr = list(np.random.choice(len(trianglegraphletid), self._trianglenum-len(trianglegraphlet)))
    #             for index in index_arr:
    #                 trianglegraphlet.append(triangle[trianglegraphletid[index]])
    #     # comment following code for running real dataset
    #     if len(trianglegraphlet) < self._trianglenum:
    #         print("===== COMMENT AFTER THIS RUN!!! =====")
    #         for _ in range(len(trianglegraphlet), self._trianglenum):
    #             trianglegraphlet.append([0, 0, 0])
    #     if len(notrianglegraphlet) < self._trianglenum:
    #         print("===== RAISE RANDOM WALK NUMBER =====")
    #         print("===== NOTRIANGLENUM: %d =====" %(self._trianglenum-len(notrianglegraphlet)))
    #         for _id in notriangleids:
    #             if _id not in _notrianglegraphletid:
    #                 notrianglegraphletid.append(_id)
    #         assert len(notrianglegraphletid) == len(notriangle)-len(notrianglegraphlet)
    #         if len(notrianglegraphletid) > 0 and len(notrianglegraphlet) < self._trianglenum:
    #             index_arr = list(np.random.choice(len(notrianglegraphletid), self._trianglenum-len(notrianglegraphlet)))
    #             for index in index_arr:
    #                 notrianglegraphlet.append(notriangle[notrianglegraphletid[index]])
    #     # comment following code for running real dataset
    #     if len(notrianglegraphlet) < self._trianglenum:
    #         print("===== COMMENT AFTER THIS RUN!!! =====")
    #         for _ in range(len(notrianglegraphlet), self._trianglenum):
    #             notrianglegraphlet.append([0, 0, 0])
    #     assert len(trianglegraphlet) == self._trianglenum == len(notrianglegraphlet)
    #     return trianglegraphlet, notrianglegraphlet

    # def _generate_graphletsquareid(self, graph, sourceid, targetid, square, nosquare):
    #     # use random walk to get neighbors of source and target nodes
    #     # square, nosquare: list
    #     nodeset = []
    #     squaregraphlet, nosquaregraphlet = [], []
    #     # record id for graphlet sampled for following padding
    #     squareids, nosquareids = [_id for _id in range(len(square))], [_id for _id in range(len(nosquare))]
    #     _squaregraphletid, _nosquaregraphletid = [], []
    #     squaregraphletid, nosquaregraphletid = [], []
    #     nodes = th.Tensor([sourceid]*self._walknum+[targetid]*self._walknum).cpu().to(th.int64)
    #     traces, _ = dgl.sampling.random_walk(graph, nodes, length=self._walklen)
    #     traces = traces.reshape([2*self._walknum*(self._walklen+1)]).tolist()
    #     for _node in traces:
    #         if _node not in nodeset:
    #             nodeset.append(_node)
    #         else:
    #             continue
    #         if len(squaregraphlet) < self._squarenum:
    #             for _graphlet, _id in zip(square, squareids):
    #                 if _node in _graphlet:
    #                     if _graphlet not in squaregraphlet:
    #                         squaregraphlet.append(_graphlet)
    #                         _squaregraphletid.append(_id)
    #         if len(nosquaregraphlet) < self._squarenum:
    #             for _graphlet, _id in zip(nosquare, nosquareids):
    #                 if _node in _graphlet:
    #                     if _graphlet not in nosquaregraphlet:
    #                         nosquaregraphlet.append(_graphlet)
    #                         _nosquaregraphletid.append(_id)
    #     # if there are not enough graphlet, randomly sample from buffer
    #     if len(squaregraphlet) < self._squarenum:
    #         print("===== RAISE RANDOM WALK NUMBER =====")
    #         print("===== SQUARENUM: %d =====" %(self._squarenum-len(squaregraphlet)))
    #         for _id in squareids:
    #             if _id not in _squaregraphletid:
    #                 squaregraphletid.append(_id)
    #         assert len(squaregraphletid) == len(square)-len(squaregraphlet)
    #         if len(squaregraphletid) > 0 and len(squaregraphlet) < self._squarenum:
    #             index_arr = list(np.random.choice(len(squaregraphletid), self._squarenum-len(squaregraphlet)))
    #             for index in index_arr:
    #                 squaregraphlet.append(square[squaregraphletid[index]])
    #     # comment following code for running real dataset
    #     if len(squaregraphlet) < self._squarenum:
    #         print("===== COMMENT AFTER THIS RUN!!! =====")
    #         for _ in range(len(squaregraphlet), self._squarenum):
    #             squaregraphlet.append([0, 0, 0, 0])
    #     if len(nosquaregraphlet) < self._squarenum:
    #         print("===== RAISE RANDOM WALK NUMBER =====")
    #         print("===== NOSQUARENUM: %d =====" %(self._squarenum-len(nosquaregraphlet)))
    #         for _id in nosquareids:
    #             if _id not in _nosquaregraphletid:
    #                 nosquaregraphletid.append(_id)
    #         assert len(nosquaregraphletid) == len(nosquare)-len(nosquaregraphlet)
    #         if len(nosquaregraphletid) > 0 and len(nosquaregraphlet) < self._squarenum:
    #             index_arr = list(np.random.choice(len(nosquaregraphletid), self._squarenum-len(nosquaregraphlet)))
    #             for index in index_arr:
    #                 nosquaregraphlet.append(nosquare[nosquaregraphletid[index]])
    #     # comment following code for running real dataset
    #     if len(nosquaregraphlet) < self._squarenum:
    #         print("===== COMMENT AFTER THIS RUN!!! =====")
    #         for _ in range(len(nosquaregraphlet), self._squarenum):
    #             nosquaregraphlet.append([0, 0, 0, 0])
    #     assert len(squaregraphlet) == self._squarenum == len(nosquaregraphlet)
    #     return squaregraphlet, nosquaregraphlet
