import torch as th
import torch.nn as nn
import os
import sys
import math
import torch.nn.functional as F
sys.path.append("../")
from utils.base import glorot

# Neural Message Passing for Quantum Chemistry
class GraphletMessagePass(nn.Module):
    def __init__(self, pass_type, readout_type, update_type, graphlet_type, graphlet_num, pass_dim, in_dim, out_dim, pass_num, is_neighbor):
        super(GraphletMessagePass, self).__init__()
        # graphlet_type: 3 for triangle, 4 for square
        # pass_type: mean, project, concat
        # update_type: gru, concat
        # readout_type: softmax, product, concat
        # is_neighbor: true or false
        self.embed_size = pass_dim
        self.pass_num = pass_num
        self.pass_type = pass_type
        self.update_type = update_type
        self.readout_type = readout_type
        self.graphlet_type = graphlet_type
        self.graphlet_num = graphlet_num
        self.is_neighbor = is_neighbor
        self._Fpass1 = nn.Sequential(nn.Linear(in_dim*self.graphlet_type, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Wpass1 = nn.Parameter(glorot((in_dim, self.embed_size)))
        self._Fupdate1 = nn.Sequential(nn.Linear(self.embed_size*2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        # default layer of GRU is 1
        assert self.graphlet_num % 2 == 0
        self._Fupdate2 = nn.GRU(input_size=self.embed_size, hidden_size=self.embed_size, num_layers=int(self.graphlet_num/2), bidirectional=True)
        self._Fupdate3 = nn.Sequential(nn.Linear(self.embed_size*2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fread1 = nn.Softmax(dim=2)
        self._Wread1 = nn.Parameter(glorot((self.embed_size, self.embed_size)))
        self._Fread2 = nn.Sequential(nn.Linear(self.embed_size, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fread3 = nn.Sequential(nn.Linear(self.embed_size*self.graphlet_type, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fread4 = nn.Sequential(nn.Linear(self.embed_size*2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fread5 = nn.Sequential(nn.Linear(self.embed_size, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        # output for [source, target, source*target]
        self._Fout1 = nn.Sequential(nn.Linear(self.embed_size*3, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_dim))
        # output for [graphlet]
        self._Fout2 = nn.Sequential(nn.Linear(self.embed_size, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_dim))

    def forward(self, graphlet_embed):
        # graphlet_embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        _init_embed = graphlet_embed
        for _ in range(self.pass_num):
            # message passing
            if self.pass_type == "mean":
                # _message: batch_size, graphlet_num, 1, embed_size
                _message = th.mean(graphlet_embed, dim=2, keepdim=True)
                # _message: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
                _message = th.repeat_interleave(_message, self.graphlet_type, dim=2)
                # _message = _message.repeat(1, 1, self.graphlet_type, 1)
            elif self.pass_type == "concat":
                # _message: batch_size, graphlet_num, 3 for triangle (4 for square) * embed_size
                _message = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[2], graphlet_embed.shape[3]*graphlet_embed.shape[4])
                # _message: batch_size, graphlet_num, embed_size
                _message = self._Fpass1(_message)
                # _message: batch_size, graphlet_num, 1, embed_size
                _message = _message.unsqueeze(2)
                # _message: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
                _message = th.repeat_interleave(_message, self.graphlet_type, dim=1)
                # _message = _message.repeat(1, 1, self.graphlet_type, 1)
            elif self.pass_type == "project":
                # _message: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
                _message = th.matmul(self._Wpass1, graphlet_embed)
            else:
                raise NotImplementedError
            # node updating
            _graphlet_embed = []
            for _node in range(self.graphlet_type):
                if self.update_type == "concat":
                    # _embed: batch_size, graphlet_num, 2embed_size
                    _embed = th.cat((_init_embed[:,:,_node,:], _message[:,:,_node,:]), dim=2)
                    # batch_size, graphlet_num, embed_size
                    _embed = self._Fupdate1(_embed)
                elif self.update_type == "gru":
                    # _embed: batch_size, graphlet_num, 2embed_size
                    _embed = th.cat((_init_embed[:,:,_node,:], _message[:,:,_node,:]), dim=2)
                    # batch_size, graphlet_num, embed_size
                    _embed = self._Fupdate1(_embed)
                    _embed = _embed.reshape(_embed.shape[1], _embed.shape[0], _embed.shape[2])
                    # input of GRU: seq_len, batch_size, input_size
                    # _embed: graphlet_num, batch_size, embed_size -> graphlet_num, batch_size, 2embed_size
                    _embed, _ = self._Fupdate2(_embed, _embed)
                    _embed = _embed.reshape(_embed.shape[1], _embed.shape[0], _embed.shape[2])
                    # _embed: batch_size, graphlet_num, embed_size
                    _embed = self._Fupdate3(_embed)
                else:
                    raise NotImplementedError
                _graphlet_embed.append(_embed)
            # graphlet_embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
            graphlet_embed = th.stack(_graphlet_embed, dim=2)
            # graph readout
            if self.readout_type == "softmax":
                _readout_embed = []
                for _node in range(self.graphlet_type):
                    # _embed: batch_size, graphlet_num, embed_size
                    _embed = th.matmul(self._Wread1, graphlet_embed[:,:,_node,:])
                    # _embed: batch_size, graphlet_num, embed_size
                    _embed = self._Fread1(_embed)
                    _readout_embed.append(_embed)
                # _embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
                _embed = th.stack(_readout_embed, dim=2)
                # _embed: batch_size, graphlet_num, embed_size
                _embed = th.sum(_embed, dim=2, keepdim=False)
                # _embed: batch_size, graphlet_num, embed_size
                _embed = self._Fread2(_embed)
            elif self.readout_type == "concat":
                # _embed: batch_size, graphlet_num, 3 for triangle (4 for square) * embed_size
                _embed = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[1], graphlet_embed.shape[2]*graphlet_embed.shape[3])
                # _embed: batch_size, graphlet_num, embed_size
                _embed = self._Fread3(_embed)
            elif self.readout_type == "product":
                _readout_embed = []
                for _node in range(self.graphlet_type):
                    # _embed1: batch_size, graphlet_num, 2embed_size
                    _embed1 = th.cat((graphlet_embed[:,:,_node,:], _init_embed[:,:,_node,:]), dim=2)
                    # _embed1: batch_size, graphlet_num, embed_size
                    _embed1 = self._Fread4(_embed1)
                    # _embed2: batch_size, graphlet_num, embed_size
                    _embed2 = self._Fread2(graphlet_embed[:,:,_node,:])
                    # _embed: batch_size, graphlet_num, embed_size
                    _embed = _embed1 * _embed2
                    # _embed: batch_size, graphlet_num, embed_size
                    _embed = self._Fread5(_embed)
                    _readout_embed.append(_embed)
                # _embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
                _embed = th.stack(_readout_embed, dim=2)
                # _embed: batch_size, graphlet_num, embed_size
                _embed = th.sum(_embed, dim=2, keepdim=False)
        if self.is_neighbor:
            # _embed: batch_size, graphlet_num, 3, embed_size
            _embed = th.stack((graphlet_embed[:,:,0,:], graphlet_embed[:,:,self.graphlet_type-1,:], graphlet_embed[:,:,0,:]*graphlet_embed[:,:,self.graphlet_type-1,:]), dim=2)
            # _embed: batch_size, graphlet_num, 3 * embed_size
            _embed = _embed.reshape(_embed.shape[0], _embed.shape[1], _embed.shape[2]*_embed.shape[3])
            return self._Fout1(_embed)
        else:
            return self._Fout2(_embed)

class GraphletAggregate(nn.Module):
    def __init__(self, aggregate_type, graphlet_num, aggregate_dim, in_size, out_size):
        super(GraphletAggregate, self).__init__()
        # aggregate_type: mean, concat, gru, project
        self.embed_size = aggregate_dim
        self.aggregate_type = aggregate_type
        self.graphlet_num = graphlet_num
        self._Faggregate1 = nn.Sequential(nn.Linear(in_size*self.graphlet_num, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Faggregate2 = nn.Sequential(nn.Linear(in_size, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Waggregate1 = nn.Parameter(glorot((in_size*self.graphlet_num, self.embed_size)))
        assert self.graphlet_num % 2 == 0
        self._Faggregate3 = nn.GRU(input_size=in_size, hidden_size=self.embed_size, num_layers=int(self.graphlet_num/2), bidirectional=True)
        self._Faggregate4 = nn.Sequential(nn.Linear(self.embed_size*2*self.graphlet_num, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fout = nn.Sequential(nn.Linear(self.embed_size, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_size))

    def forward(self, graphlet_embed, graphlet_mask=None):
        # graphlet_embed: batch_size, graphlet_num, embed_size
        if graphlet_mask is not None:
            # graphlet_mask: batch_size, graphlet_num, 0/1
            graphlet_mask = th.repeat_interleave(graphlet_mask, self.embed_size, dim=1)
            graphlet_mask = graphlet_mask.reshape(graphlet_mask.shape[0], graphlet_embed.shape[1], self.embed_size)
            graphlet_embed = graphlet_embed * graphlet_mask
        if self.aggregate_type == "concat":
            # _embed: batch_size, graphlet_num * embed_size
            _embed = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[1]*graphlet_embed.shape[2])
            # _embed: batch_size, embed_size
            _embed = self._Faggregate1(_embed)
        elif self.aggregate_type == "mean":
            # _embed: batch_size, embed_size
            _embed = th.mean(graphlet_embed, dim=1)
            # _embed: batch_size, embed_size
            _embed = self._Faggregate2(_embed)
        elif self.aggregate_type == "project":
            # _embed: batch_size, graphlet_num * embed_size
            _embed = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[1]*graphlet_embed[2])
           # _embed: batch_size, embed_size
            _embed = th.matmul(_embed, self._Waggregate1)
        elif self.aggregate_type == "gru":
            # _embed: graphlet_num, batch_size, embed_size
            _embed = _embed.reshape(_embed.shape[1], _embed.shape[0], _embed.shape[2])
            # input of GRU: seq_len, batch_size, input_size
            # _embed: graphlet_num, batch_size, embed_size -> graphlet_num, batch_size, 2embed_size
            _embed, _ = self._Faggregate3(_embed, _embed)
            # _embed: batch_size, graphlet_num * 2embed_size
            _embed = _embed.reshape(_embed.shape[1], _embed.shape[0] * _embed.shape[2])
            # _embed: batch_size, embed_size
            _embed = self._Faggregate4(_embed)
        else:
            raise NotImplementedError
        return self._Fout(_embed)
        

class LogicCombine(nn.Module):
    def __init__(self, combine_type, combine_embed, triangle_signal, square_signal, in_size, out_size):
        super(LogicCombine, self).__init__()
        # combine_type: concat, product
        # for the final prediction, use sigmoid
        self.embed_size = combine_embed
        self.combine_type = combine_type
        self._istriangle = triangle_signal
        self._issquare = square_signal
        self._Fcombine1 = nn.Sequential(nn.Linear(in_size*4, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine2 = nn.Sequential(nn.Linear(in_size*2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine3 = nn.Sequential(nn.Linear(in_size*8, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine4 = nn.Sequential(nn.Linear(in_size*4, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine5 = nn.Sequential(nn.Linear(in_size*5, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine6 = nn.Sequential(nn.Linear(in_size*5, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine7 = nn.Sequential(nn.Linear(in_size*3, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fcombine8 = nn.Sequential(nn.Linear(in_size*3, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, self.embed_size))
        self._Fout1 = nn.Sequential(nn.Linear(self.embed_size*7, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_size))
        self._Fout2 = nn.Sequential(nn.Linear(self.embed_size*5, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_size))
        self._Fout3 = nn.Sequential(nn.Linear(self.embed_size*3, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_size))
        self._Fout4 = nn.Sequential(nn.Linear(self.embed_size*6, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, out_size))

    def forward(self, source_embed, target_embed, triangle_embed=None, triangleneighbor_embed=None, square_embed=None, squareneighbor_embed=None):
        if self.combine_type == "concat":
            # source_embed, source_embed: batch_size, embed_size
            _embed = [source_embed, target_embed, source_embed*target_embed]
            # _embed: batch_size, 3, embed_size
            _embed = th.stack(_embed, dim=1)
            # triangle_embed: batch_size, 4, embed_size
            # triangleneighbor_embed: batch_size, 2, embed_size
            if self._istriangle:
                # _triangleembed: batch_size, 4embed_size
                _triangleembed = triangle_embed.reshape(triangle_embed.shape[0], triangle_embed.shape[1]*triangle_embed.shape[2])
                # _triangleembed: batch_size, embed_size
                _triangleembed = self._Fcombine1(_triangleembed)
                # _triangleneighborembed: batch_size, 2embed_size
                _triangleneighborembed = triangleneighbor_embed.reshape(triangleneighbor_embed.shape[0], triangleneighbor_embed.shape[1]*triangleneighbor_embed.shape[2])
                # _triangleneighborembed: batch_size, embed_size
                _triangleneighborembed = self._Fcombine2(_triangleneighborembed)
            if self._issquare:
                # _squareembed: batch_size, 8embed_size
                _squareembed = square_embed.reshape(square_embed.shape[0], square_embed.shape[1]*square_embed.shape[2])
                # _squareembed: batch_size, embed_size
                _squareembed = self._Fcombine3(_squareembed)
                # _squareneighborembed: batch_size, 4embed_size
                _squareneighborembed = squareneighbor_embed.reshape(squareneighbor_embed.shape[0], squareneighbor_embed.shape[1]*squareneighbor_embed.shape[2])
                # _squareneighborembed: batch_size, embed_size
                _squareneighborembed = self._Fcombine4(_squareneighborembed)
            if self._istriangle and self._issquare:
                # _embed: batch_size, 7embed_size
                _embed = th.cat((source_embed, target_embed, source_embed*target_embed, _triangleembed, _triangleneighborembed, _squareembed, _squareneighborembed), dim=1)
                # _embed: batch_size, out_size
                return self._Fout1(_embed).softmax(1)
            elif self._istriangle and not self._issquare:
                # _embed: batch_size, 5embed_size
                _embed = th.cat((source_embed, target_embed, source_embed*target_embed, _triangleembed, _triangleneighborembed), dim=1)
                # _embed: batch_size, out_size
                return self._Fout2(_embed).softmax(1)
            elif self._issquare and not self._istriangle:
                # _embed:batch_size, 5embed_size
                _embed = th.cat((source_embed, target_embed, source_embed*target_embed, _squareembed, _squareneighborembed), dim=1)
                # _embed: batch_size, out_size
                return self._Fout2(_embed).softmax(1)
            elif not self._istriangle and not self._issquare:
                # _embed:batch_size, 3embed_size
                _embed = th.cat((source_embed, target_embed, source_embed*target_embed), dim=1)
                # _embed: batch_size, out_size
                return self._Fout3(_embed).softmax(1)
            else:
                raise NotImplementedError
        elif self.combine_type == "product":
            if self._istriangle:
                # _triangleembed: batch_size, 4embed_size
                _triangleembed = triangle_embed.reshape(triangle_embed.shape[0], triangle_embed.shape[1]*triangle_embed.shape[2])
                # _triangleembed: batch_size, embed_size
                _triangleembed = self._Fcombine1(_triangleembed)
                # _triangleneighborembed: batch_size, 2embed_size
                _triangleneighborembed = triangleneighbor_embed.reshape(triangleneighbor_embed.shape[0], triangleneighbor_embed.shape[1]*triangleneighbor_embed.shape[2])
                # _triangleneighborembed: batch_size, embed_size
                _triangleneighborembed = self._Fcombine2(_triangleneighborembed)
            if self._issquare:
                # _squareembed: batch_size, 8embed_size
                _squareembed = square_embed.reshape(square_embed.shape[0], square_embed.shape[1]*square_embed.shape[2])
                # _squareembed: batch_size, embed_size
                _squareembed = self._Fcombine3(_squareembed)
                # _squareneighborembed: batch_size, 4embed_size
                _squareneighborembed = squareneighbor_embed.reshape(squareneighbor_embed.shape[0], squareneighbor_embed[1]*squareneighbor_embed[2])
                # _squareneighborembed: batch_size, embed_size
                _squareneighborembed = self._Fcombine4(_squareneighborembed)
            if self._istriangle and self._issquare:
                # _sourceembed: batch_size, 5embed_size
                _sourceembed = th.cat((source_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), dim=1)
                _sourceembed = self._Fcombine5(_sourceembed)
                _targetembed = th.cat((target_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), dim=1)
                _targetembed = self._Fcombine6(_targetembed)
                # _embed: batch_size, 6embed_size
                _embed = th.cat((source_embed, target_embed, _sourceembed, _targetembed, source_embed*target_embed, _sourceembed*_targetembed), dim=1)
                return self._Fout4(_embed).softmax(1)
            elif self._istriangle and not self._issquare:
                # _sourceembed: batch_size, 3embed_size
                _sourceembed = th.cat((source_embed, triangle_embed, triangleneighbor_embed), dim=1)
                _sourceembed = self._Fcombine7(_sourceembed)
                _targetembed = th.cat((target_embed, triangle_embed, triangleneighbor_embed), dim=1)
                _targetembed = self._Fcombine8(_targetembed)
                # _embed: batch_size, 6embed_size
                _embed = th.cat((source_embed, target_embed, _sourceembed, _targetembed, source_embed*target_embed, _sourceembed*_targetembed), dim=1)
                return self._Fout4(_embed).softmax(1)
            elif self._issquare and not self._istriangle:
                # _sourceembed: batch_size, 3embed_size
                _sourceembed = th.cat((source_embed, square_embed, squareneighbor_embed), dim=1)
                _sourceembed = self._Fcombine7(_sourceembed)
                _targetembed = th.cat((target_embed, square_embed, squareneighbor_embed), dim=1)
                _targetembed = self._Fcombine8(_targetembed)
                # _embed: batch_size, 6embed_size
                _embed = th.cat((source_embed, target_embed, _sourceembed, _targetembed, source_embed*target_embed, _sourceembed*_targetembed), dim=1)
                return self._Fout4(_embed).softmax(1)
            else:
                raise NotImplementedError


class Angel(nn.Module):
    def __init__(self, node_num, nodetype_num, embed_size, triangle_signal, square_signal,
                    pass_type, readout_type, update_type, aggregate_type, combine_type,
                    pass_embed, aggregate_embed, combine_embed, pass_num,
                    triangleneighbor_num, squareneighbor_num, triangle_num, square_num,
                    node_embedding=None, label_num=1):
        super(Angel, self).__init__()
        # triangle signal, square signal: bool
        self._istriangle = triangle_signal
        self._issquare = square_signal
        self.nodetype_embedding = (th.rand((nodetype_num, embed_size))*2-1)*math.sqrt(3)
        if node_embedding is not None:
            node_emb0, node_emb1 = node_embedding.shape[0], node_embedding.shape[1]
            assert node_num == node_emb0
            # embedding_size is the same as the default
            assert embed_size == node_emb1
            self.node_embedding = node_embedding
        else:
            self.node_embedding = (th.rand((node_num, embed_size))*2-1)*math.sqrt(3)
        self._embedding = nn.Embedding.from_pretrained(self.node_embedding).requires_grad_()
        self._nodeembedding = nn.Embedding.from_pretrained(self.nodetype_embedding).requires_grad_()
        self.NodeEmbeddingGenerater = nn.Sequential(nn.Linear(2*embed_size, embed_size), nn.ReLU(), nn.Linear(embed_size, embed_size))
        if self._istriangle:
            # graphlet_type = ["triangle", "triangle", "notriangle", "notriangle"]
            self.TriangleMessagePass = nn.ModuleList()
            self.TriangleNeighborPass = nn.ModuleList()
            for _id in range(4):
                # triangle1, triangle2, notriangle1, notriangle2
                self.TriangleMessagePass.append(GraphletMessagePass(pass_type, readout_type, update_type, 3, triangle_num, pass_embed, embed_size, embed_size, pass_num, False))
            for _id in range(2):
                # triangle_neighbor1, triangle_neighbor2
                self.TriangleNeighborPass.append(GraphletMessagePass(pass_type, readout_type, update_type, 3, triangleneighbor_num, pass_embed, embed_size, embed_size, pass_num, True))
            self.TriangleAggregate = nn.ModuleList()
            self.TriangleNeighborAggregate = nn.ModuleList()
            for _ in range(4):
                self.TriangleAggregate.append(GraphletAggregate(aggregate_type, triangle_num, aggregate_embed, embed_size, embed_size))
            for _ in range(2):
                self.TriangleNeighborAggregate.append(GraphletAggregate(aggregate_type, triangleneighbor_num, aggregate_embed, embed_size, embed_size))
        if self._issquare:
            # graphlet_type = ["square", "square", "square", "square", "nosquare", "nosquare", "nosquare", "nosquare"]
            self.SquareMessagePass = nn.ModuleList()
            self.SquareNeighborPass = nn.ModuleList()
            for _id in range(8):
                # square1, square2, square3, square4, nosquare1, nosquare2, nosquare3, nosquare4, square_neighbor1, square_neighbor2, square_neighbor3, square_neighbor4
                self.SquareMessagePass.append(GraphletMessagePass(pass_type, readout_type, update_type, 4, square_num, pass_embed, embed_size, embed_size, pass_num, False))
            for _id in range(4):
                self.SquareNeighborPass.append(GraphletMessagePass(pass_type, readout_type, update_type, 4, squareneighbor_num, pass_embed, embed_size, embed_size, pass_num, True))
            self.SquareAggregate = nn.ModuleList()
            self.SquareNeighborAggregate = nn.ModuleList()
            for _ in range(8):
                self.SquareAggregate.append(GraphletAggregate(aggregate_type, square_num, aggregate_embed, embed_size, embed_size))
            for _ in range(4):
                self.SquareNeighborAggregate.append(GraphletAggregate(aggregate_type, squareneighbor_num, aggregate_embed, embed_size, embed_size))
        if self._istriangle or self._issquare:
            self.LogicCombine = LogicCombine(combine_type, combine_embed, triangle_signal, square_signal, embed_size, label_num)
        else:
            raise NotImplementedError
        
    def forward(self, source_id, target_id, 
                trianglelogic1, trianglelogic2, 
                squarelogic1, squarelogic2, squarelogic3, squarelogic4,
                triangle1, triangle2, notriangle1, notriangle2, 
                square1, square2, square3, square4, nosquare1, nosquare2, nosquare3, nosquare4, 
                triangle1_neighbor, triangle2_neighbor, triangle1_mask, triangle2_mask, square1_neighbor, square2_neighbor, square3_neighbor, square4_neighbor, square1_mask, square2_mask, square3_mask, square4_mask):
        source_embed = self._embedding(source_id)
        target_embed = self._embedding(target_id)
        triangle_embed, square_embed = [], []
        triangleneighbor_embed, squareneighbor_embed = [], []
        if self._istriangle:
            # graphlet_neighbor: batch_size, graphletneighbor_num, 3 for triangle (4 for square), embed_size
            _triangles = [triangle1, triangle2, notriangle1, notriangle2] 
            _triangleneighbors = [triangle1_neighbor, triangle2_neighbor]
            _trianglelogics = [trianglelogic1, trianglelogic2]
            trianglemasks = [triangle1_mask, triangle2_mask]
            # triangleneighbors = [triangle1_neighbor, triangle2_neighbor]
            triangleneighbors = []
            for _neighbor in range(2):
                triangleneighbor = []
                for _node in range(3):
                    _embed = self.NodeEmbeddingGenerater(th.cat((self._embedding(_triangleneighbors[_neighbor][:,:,_node]), self._nodeembedding(_trianglelogics[_neighbor][:,_node])), dim=2))
                    triangleneighbor.append(_embed)
                triangleneighbors.append(th.stack(triangleneighbor, dim=2))
            # triangles = [triangle1, triangle2, notriangle1, notriangle2] 
            triangles = []
            for _triangle in range(4):
                triangle = []
                for _node in range(3):
                    _embed = self.NodeEmbeddingGenerater(th.cat((self._embedding(_triangles[_triangle][:,:,_node]), self._nodeembedding(_trianglelogics[_triangle][:,_node])), dim=2))
                    triangle.append(_embed)
                triangles.append(th.stack(triangle, dim=2))
            for _t in range(4):
                # graphlet_embed: batch_size, graphlet_num, embed_size
                graphlet_embed = self.TriangleMessagePass[_t](triangles[_t])
                # graphlet_embed: batch_size, embed_size
                graphlet_embed = self.TriangleAggregate[_t](graphlet_embed)
                triangle_embed.append(graphlet_embed)
            for _t in range(2):
                # graphlet_embed: batch_size, graphletneighbor_num, embed_size
                graphlet_embed = self.TriangleNeighborPass[_t](triangleneighbors[_t])
                # graphlet_embed: batch_size, embed_size
                graphlet_embed = self.TriangleNeighborAggregate[_t](graphlet_embed, trianglemasks[_t])
                triangleneighbor_embed.append(graphlet_embed)
        if self._issquare:
            _squares = [square1, square2, square3, square4, nosquare1, nosquare2, nosquare3, nosquare4] 
            _squareneighbors = [square1_neighbor, square2_neighbor, square3_neighbor, square4_neighbor]
            _squarelogics = [squarelogic1, squarelogic2, squarelogic3, squarelogic4]
            squaremasks = [square1_mask, square2_mask, square3_mask, square4_mask]
            # squareneighbors = [square1_neighbor, square2_neighbor, square3_neighbor, square4_neighbor] 
            squareneighbors = []
            for _neighbor in range(4):
                squareneighbor = []
                for _node in range(4):
                    _embed = self.NodeEmbeddingGenerater(th.cat((self._embedding(_squareneighbors[_neighbor][:,:,_node]), self._nodeembedding(_squarelogics[_neighbor][:,_node])), dim=2))
                    squareneighbor.append(_embed)
                squareneighbors.append(th.stack(squareneighbor, dim=2))
            # squares = [square1, square2, square3, square4, nosquare1, nosquare2, nosquare3, nosquare4] 
            squares = []
            for _square in range(8):
                square = []
                for _node in range(4):
                    _embed = self.NodeEmbeddingGenerater(th.cat((self._embedding(_squares[_square][:,:,_node]), self._nodeembedding(_squarelogics[_square][:,_node])), dim=2))
                    square.append(_embed)
                squares.append(th.stack(square, dim=2))
            squaremasks = [square1_mask, square2_mask, square3_mask, square4_mask]
            for _s in range(8):
                graphlet_embed = self.SquareMessagePass[_s](squares[_s])
                graphlet_embed = self.SquareAggregate[_s](graphlet_embed)
                square_embed.append(graphlet_embed)
            for _s in range(4):
                graphlet_embed = self.SquareNeighborPass[_s](squareneighbors[_s])
                graphlet_embed = self.SquareNeighborAggregate[_s](graphlet_embed, squaremasks[_s])
                squareneighbor_embed.append(graphlet_embed)
        # triangle_embed: batch_size, 4, embed_size
        triangle_embed = th.stack(triangle_embed, dim=1) if self._istriangle else None
        # triangleneighbor_embed: batch_size, 2, embed_size
        triangleneighbor_embed = th.stack(triangleneighbor_embed, dim=1) if self._istriangle else None
        # square_embed: batch_size, 8, embed_size
        square_embed = th.stack(square_embed, dim=1) if self._issquare else None
        # squareneighbor_embed: batch_size, 4, embed_size
        squareneighbor_embed = th.stack(squareneighbor_embed, dim=1) if self._issquare else None
        return self.LogicCombine(source_embed, target_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed)