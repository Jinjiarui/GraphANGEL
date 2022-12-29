import math
import os
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from utils.base import glorot

# Neural Message Passing for Quantum Chemistry
class GraphletMessagePass(nn.Module):
    def __init__(
        self,
        pass_type,
        readout_type,
        update_type,
        graphlet_type,
        graphlet_num,
        pass_dim,
        in_dim,
        out_dim,
        pass_num,
        is_neighbor,
    ):
        super(GraphletMessagePass, self).__init__()
        # graphlet_type: 3 for triangle, 4 for square
        # pass_type: mean, project, concat
        # update_type: gru, concat
        # readout_type: softmax, product, concat
        # is_neighbor: true or false
        self.pass_num = pass_num
        self.pass_type = pass_type
        self.update_type = update_type
        self.readout_type = readout_type
        self.graphlet_type = graphlet_type
        self.is_neighbor = is_neighbor

        self._Fpass1 = nn.Sequential(
            nn.Linear(in_dim * self.graphlet_type, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim)
        )
        self._Wpass1 = nn.Parameter(glorot((in_dim, pass_dim)))

        self._Fupdate1 = nn.Sequential(nn.Linear(pass_dim * 2, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim))
        # default layer of GRU is 1
        assert graphlet_num % 2 == 0
        self._Fupdate2 = nn.GRU(
            input_size=pass_dim, hidden_size=pass_dim, num_layers=graphlet_num // 2, bidirectional=True
        )
        self._Fupdate3 = nn.Sequential(nn.Linear(pass_dim * 2, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim))

        self._Fread1 = nn.Softmax(dim=2)
        # self._Wread1 = nn.Parameter(glorot((pass_dim, pass_dim)))
        self._Fread2 = nn.Sequential(nn.Linear(pass_dim, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim))
        self._Fread3 = nn.Sequential(
            nn.Linear(pass_dim * self.graphlet_type, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim)
        )
        self._Fread4 = nn.Sequential(nn.Linear(pass_dim * 2, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim))
        self._Fread5 = nn.Sequential(nn.Linear(pass_dim, pass_dim), nn.ReLU(), nn.Linear(pass_dim, pass_dim))
        if is_neighbor:
            # output for [source, target, source*target]
            self._Fout = nn.Sequential(nn.Linear(pass_dim * 3, pass_dim), nn.ReLU(), nn.Linear(pass_dim, out_dim))
        else:
            # output for [graphlet]
            self._Fout = nn.Sequential(nn.Linear(pass_dim, pass_dim), nn.ReLU(), nn.Linear(pass_dim, out_dim))

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(pass_num):
            in_dims = graphlet_type * pass_dim
            out_dims = graphlet_type * pass_dim if i < pass_num - 1 else out_dim
            self.layers.append(nn.Linear(in_dims, out_dims))

            if i < pass_num - 1:
                self.norms.append(nn.BatchNorm1d(out_dims))

    def forward(self, graphlet_embed):
        # graphlet_embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        # _init_embed = graphlet_embed
        h = graphlet_embed.reshape(
            graphlet_embed.shape[0], graphlet_embed.shape[1], graphlet_embed.shaoe[2] * graphlet_embed.shape[3]
        )  # batch_size, graphlet_num,  for triangle (4 for square) * embed_size

        for i in range(self.pass_num):
            h = self.layers[i](h)

            if i < self.pass_num - 1:
                h = self.norms[i](h)
                h = F.relu(h)
                # h = self.dropout(h)
        return h

        # for _ in range(self.pass_num):
        #     # message passing
        #     # if self.pass_type == "mean":
        #     #     # _message: batch_size, graphlet_num, 1, embed_size
        #     #     _message = th.mean(graphlet_embed, dim=2, keepdim=True)
        #     #     # _message: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        #     #     _message = th.repeat_interleave(_message, self.graphlet_type, dim=2)
        #     #     # _message = _message.repeat(1, 1, self.graphlet_type, 1)
        #     # elif self.pass_type == "concat":
        #     #     # _message: batch_size, graphlet_num, 3 for triangle (4 for square) * embed_size
        #     #     _message = graphlet_embed.reshape(
        #     #         graphlet_embed.shape[0], graphlet_embed.shape[2], graphlet_embed.shape[3] * graphlet_embed.shape[4]
        #     #     )
        #     #     # _message: batch_size, graphlet_num, embed_size
        #     #     _message = self._Fpass1(_message)
        #     #     # _message: batch_size, graphlet_num, 1, embed_size
        #     #     _message = _message.unsqueeze(2)
        #     #     # _message: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        #     #     _message = th.repeat_interleave(_message, self.graphlet_type, dim=1)
        #     #     # _message = _message.repeat(1, 1, self.graphlet_type, 1)
        #     # elif self.pass_type == "project":
        #     #     # _message: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        #     #     _message = th.matmul(self._Wpass1, graphlet_embed)
        #     # else:
        #     #     raise NotImplementedError

        #     _message = th.matmul(self._Wpass1, graphlet_embed)

        #     # node updating
        #     _graphlet_embed = []
        #     for _node in range(self.graphlet_type):
        #         if self.update_type == "concat":
        #             # _embed: batch_size, graphlet_num, 2embed_size
        #             _embed = th.cat((_init_embed[:, :, _node, :], _message[:, :, _node, :]), dim=2)
        #             # batch_size, graphlet_num, embed_size
        #             _embed = self._Fupdate1(_embed)
        #         elif self.update_type == "gru":
        #             # _embed: batch_size, graphlet_num, 2embed_size
        #             _embed = th.cat((_init_embed[:, :, _node, :], _message[:, :, _node, :]), dim=2)
        #             # batch_size, graphlet_num, embed_size
        #             _embed = self._Fupdate1(_embed)
        #             _embed = _embed.reshape(_embed.shape[1], _embed.shape[0], _embed.shape[2])
        #             # input of GRU: seq_len, batch_size, input_size
        #             # _embed: graphlet_num, batch_size, embed_size -> graphlet_num, batch_size, 2embed_size
        #             _embed, _ = self._Fupdate2(_embed, _embed)
        #             _embed = _embed.reshape(_embed.shape[1], _embed.shape[0], _embed.shape[2])
        #             # _embed: batch_size, graphlet_num, embed_size
        #             _embed = self._Fupdate3(_embed)
        #         else:
        #             raise NotImplementedError
        #         _graphlet_embed.append(_embed)
        #     # graphlet_embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        #     graphlet_embed = th.stack(_graphlet_embed, dim=2)

        #     # graph readout
        #     if not self.is_neighbor and self.readout_type == "softmax":
        #         _readout_embed = []
        #         for _node in range(self.graphlet_type):
        #             # _embed: batch_size, graphlet_num, embed_size
        #             # _embed = th.matmul(self._Wread1, graphlet_embed[:, :, _node, :])
        #             _embed = graphlet_embed[:, :, _node, :]
        #             _embed = self._Fread1(_embed)
        #             _readout_embed.append(_embed)
        #         # _embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        #         _embed = th.stack(_readout_embed, dim=2)
        #         # _embed: batch_size, graphlet_num, embed_size
        #         _embed = th.sum(_embed, dim=2, keepdim=False)
        #         # _embed: batch_size, graphlet_num, embed_size
        #         _embed = self._Fread2(_embed)
        #     elif self.readout_type == "concat":
        #         # _embed: batch_size, graphlet_num, 3 for triangle (4 for square) * embed_size
        #         _embed = graphlet_embed.reshape(
        #             graphlet_embed.shape[0], graphlet_embed.shape[1], graphlet_embed.shape[2] * graphlet_embed.shape[3]
        #         )
        #         # _embed: batch_size, graphlet_num, embed_size
        #         _embed = self._Fread3(_embed)
        #     elif self.readout_type == "product":
        #         _readout_embed = []
        #         for _node in range(self.graphlet_type):
        #             # _embed1: batch_size, graphlet_num, 2embed_size
        #             _embed1 = th.cat((graphlet_embed[:, :, _node, :], _init_embed[:, :, _node, :]), dim=2)
        #             # _embed1: batch_size, graphlet_num, embed_size
        #             _embed1 = self._Fread4(_embed1)
        #             # _embed2: batch_size, graphlet_num, embed_size
        #             _embed2 = self._Fread2(graphlet_embed[:, :, _node, :])
        #             # _embed: batch_size, graphlet_num, embed_size
        #             _embed = _embed1 * _embed2
        #             # _embed: batch_size, graphlet_num, embed_size
        #             _embed = self._Fread5(_embed)
        #             _readout_embed.append(_embed)
        #         # _embed: batch_size, graphlet_num, 3 for triangle (4 for square), embed_size
        #         _embed = th.stack(_readout_embed, dim=2)
        #         # _embed: batch_size, graphlet_num, embed_size
        #         _embed = th.sum(_embed, dim=2, keepdim=False)

        # if self.is_neighbor:
        #     # _embed: batch_size, graphlet_num, 3, embed_size
        #     _embed = th.stack(
        #         (
        #             graphlet_embed[:, :, 0, :],
        #             graphlet_embed[:, :, self.graphlet_type - 1, :],
        #             graphlet_embed[:, :, 0, :] * graphlet_embed[:, :, self.graphlet_type - 1, :],
        #         ),
        #         dim=2,
        #     )
        #     # _embed: batch_size, graphlet_num, 3 * embed_size
        #     _embed = _embed.reshape(_embed.shape[0], _embed.shape[1], _embed.shape[2] * _embed.shape[3])
        #     return self._Fout(_embed)
        # else:
        #     return self._Fout(_embed)


class GraphletAggregate(nn.Module):
    def __init__(
        self,
        aggregate_type,
        graphlet_num,
        aggregate_dim,
        in_size,
        out_size,
        # batch_size,
        data_type,
        is_neighbor,
        padding_len=1,
    ):
        super(GraphletAggregate, self).__init__()
        # aggregate_type: mean, concat, gru, project
        self.aggregate_type = aggregate_type
        self.is_neighbor = is_neighbor
        self.data_type = data_type
        self.embed_size = aggregate_dim
        self._Faggregate1 = nn.Sequential(
            nn.Linear(in_size * graphlet_num, aggregate_dim), nn.ReLU(), nn.Linear(aggregate_dim, aggregate_dim)
        )
        self._Faggregate2 = nn.Sequential(
            nn.Linear(in_size, aggregate_dim), nn.ReLU(), nn.Linear(aggregate_dim, aggregate_dim)
        )
        self._Waggregate1 = nn.Parameter(glorot((in_size * graphlet_num, aggregate_dim)))
        assert graphlet_num % 2 == 0
        self._Faggregate3 = nn.GRU(
            input_size=in_size, hidden_size=aggregate_dim, num_layers=int(graphlet_num / 2), bidirectional=True
        )
        self._Faggregate4 = nn.Sequential(
            nn.Linear(in_size * 2 * graphlet_num, aggregate_dim), nn.ReLU(), nn.Linear(aggregate_dim, aggregate_dim)
        )
        self._Fout = nn.Sequential(
            nn.Linear(aggregate_dim, aggregate_dim), nn.ReLU(), nn.Linear(aggregate_dim, out_size)
        )
        # data_type: 1 for positive sample and -1 for negative sample
        # _padding = th.unsqueeze(th.unsqueeze(th.tensor([data_type]), dim=0), dim=0)
        # batch_size, graphlet_num, padding_len
        # self._Mpadding = th.repeat_interleave(
        #     th.repeat_interleave(th.repeat_interleave(_padding, batch_size, dim=0), graphlet_num, dim=1),
        #     padding_len,
        #     dim=2,
        # )
        self._Fpadding = nn.Sequential(
            nn.Linear(in_size + padding_len, aggregate_dim)  # , nn.ReLU(), nn.Linear(aggregate_dim, aggregate_dim)
        )
        self.padding_len = padding_len

    def forward(self, graphlet_embed, graphlet_mask=None):
        if not self.is_neighbor:
            # graphlet_embed: batch_size, graphlet_num, embed_size+padding_len
            graphlet_embed = th.cat(
                (
                    graphlet_embed,
                    th.ones(graphlet_embed.shape[0], self.embed_size, self.padding_len, device=graphlet_embed.device)
                    * self.data_type,
                ),
                dim=2,
            )
            # graphlet_embed: batch_size, graphlet_num, embed_size
            graphlet_embed = self._Fpadding(graphlet_embed)

        if graphlet_mask is not None:
            # graphlet_mask: batch_size, graphlet_num, 0/1
            graphlet_mask = th.repeat_interleave(graphlet_mask, self.embed_size, dim=1)
            graphlet_mask = graphlet_mask.reshape(graphlet_mask.shape[0], graphlet_embed.shape[1], self.embed_size)
            graphlet_embed = graphlet_embed * graphlet_mask

        # if self.aggregate_type == "concat":
        #     # _embed: batch_size, graphlet_num * embed_size
        #     _embed = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[1] * graphlet_embed.shape[2])
        #     # _embed: batch_size, embed_size
        #     _embed = self._Faggregate1(_embed)
        if self.aggregate_type == "mean":
            # _embed: batch_size, embed_size
            _embed = th.mean(graphlet_embed, dim=1)
            # _embed: batch_size, embed_size
            _embed = self._Faggregate2(_embed)
        # elif self.aggregate_type == "project":
        #     # _embed: batch_size, graphlet_num * embed_size
        #     _embed = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[1] * graphlet_embed[2])
        #     # _embed: batch_size, embed_size
        #     _embed = th.matmul(_embed, self._Waggregate1)
        elif self.aggregate_type == "gru":
            # _embed: graphlet_num, batch_size, embed_size
            _embed = graphlet_embed.reshape(graphlet_embed.shape[1], graphlet_embed.shape[0], graphlet_embed.shape[2])
            # input of GRU: seq_len, batch_size, input_size
            # _embed: graphlet_num, batch_size, embed_size -> graphlet_num, batch_size, 2embed_size
            _embed, _ = self._Faggregate3(_embed, _embed)
            # _embed: batch_size, graphlet_num * 2embed_size
            _embed = _embed.reshape(_embed.shape[1], _embed.shape[0] * _embed.shape[2])
            # _embed: batch_size, embed_size
            _embed = self._Faggregate4(_embed)
        elif self.aggregate_type == "max":
            # _embed: batch_size, embed_size
            _embed = th.max(graphlet_embed, dim=1)
            # _embed: batch_size, embed_size
            _embed = self._Faggregate2(_embed)
        else:
            raise NotImplementedError

        return self._Fout(_embed)


class LogicCombineHG(nn.Module):
    def __init__(
        self,
        trianglelogic_num,
        squarelogic_num,
        combine_type,
        combine_embed,
        triangle_signal,
        square_signal,
        in_size,
        out_size,
    ):
        super(LogicCombineHG, self).__init__()
        # combine_type: concat, product
        # for the final prediction, use sigmoid
        self.combine_type = combine_type
        self._istriangle = triangle_signal
        self._issquare = square_signal
        self._Fcombine1 = nn.Sequential(
            nn.Linear(in_size * trianglelogic_num * 2, combine_embed),
            nn.ReLU(),
            nn.Linear(combine_embed, combine_embed),
        )
        self._Fcombine2 = nn.Sequential(
            nn.Linear(in_size * trianglelogic_num, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine3 = nn.Sequential(
            nn.Linear(in_size * squarelogic_num * 2, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine4 = nn.Sequential(
            nn.Linear(in_size * squarelogic_num, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine5 = nn.Sequential(
            nn.Linear(in_size * 5, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine6 = nn.Sequential(
            nn.Linear(in_size * 5, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine7 = nn.Sequential(
            nn.Linear(in_size * 3, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine8 = nn.Sequential(
            nn.Linear(in_size * 3, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        # if self.combine_type == "concat":
        # source_embed, target_embed: batch_size, embed_size
        # _embed: batch_size, 3, embed_size

        out_dim = 0

        if self.combine_type == "concat":
            if self._istriangle and self._issquare:
                out_dim = 7
                # return th.sigmoid(self._Fout1(_embed))
            elif self._istriangle and not self._issquare:
                out_dim = 5
                # return th.sigmoid(self._Fout2(_embed))
            elif self._issquare and not self._istriangle:
                out_dim = 5
                # return th.sigmoid(self._Fout2(_embed))
            elif not self._istriangle and not self._issquare:
                out_dim = 3
                # return th.sigmoid(self._Fout3(_embed))
            else:
                raise NotImplementedError
        elif self.combine_type == "product":
            if self._istriangle and self._issquare:
                out_dim = 6
                # return th.sigmoid(self._Fout4(_embed))
            elif self._istriangle and not self._issquare:
                out_dim = 6
                # return th.sigmoid(self._Fout4(_embed))
            elif self._issquare and not self._istriangle:
                out_dim = 6
                # return th.sigmoid(self._Fout4(_embed))
            else:
                out_dim = 3
                # return th.sigmoid(self._Fout3(_embed))
        else:
            raise NotImplementedError

        self._Fout = nn.Sequential(
            nn.Linear(combine_embed * out_dim, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )

    def forward(
        self,
        source_embed,
        target_embed,
        triangle_embed=None,
        triangleneighbor_embed=None,
        square_embed=None,
        squareneighbor_embed=None,
    ):
        # triangle_embed: batch_size, trianglelogic_num*2, embed_size
        # triangleneighbor_embed: batch_size, trianglelogic_num, embed_size
        if self._istriangle:
            # _triangleembed: batch_size, trianglelogic_num*2*embed_size
            _triangleembed = triangle_embed.reshape(
                triangle_embed.shape[0], triangle_embed.shape[1] * triangle_embed.shape[2]
            )
            # _triangleembed: batch_size, embed_size
            _triangleembed = self._Fcombine1(_triangleembed)
            # _triangleneighborembed: batch_size, trianglelogic_num*embed_size
            _triangleneighborembed = triangleneighbor_embed.reshape(
                triangleneighbor_embed.shape[0], triangleneighbor_embed.shape[1] * triangleneighbor_embed.shape[2]
            )
            # _triangleneighborembed: batch_size, embed_size
            _triangleneighborembed = self._Fcombine2(_triangleneighborembed)
        if self._issquare:
            # _squareembed: batch_size, squarelogic_num*2*embed_size
            _squareembed = square_embed.reshape(square_embed.shape[0], square_embed.shape[1] * square_embed.shape[2])
            # _squareembed: batch_size, embed_size
            _squareembed = self._Fcombine3(_squareembed)
            # _squareneighborembed: batch_size, squarelogic_num*embed_size
            _squareneighborembed = squareneighbor_embed.reshape(
                squareneighbor_embed.shape[0], squareneighbor_embed.shape[1] * squareneighbor_embed.shape[2]
            )
            # _squareneighborembed: batch_size, embed_size
            _squareneighborembed = self._Fcombine4(_squareneighborembed)
        if self.combine_type == "concat":
            # source_embed, target_embed: batch_size, embed_size
            # _embed: batch_size, 3, embed_size
            if self._istriangle and self._issquare:
                # _embed: batch_size, 7embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        source_embed * target_embed,
                        _triangleembed,
                        _triangleneighborembed,
                        _squareembed,
                        _squareneighborembed,
                    ),
                    dim=1,
                )
                # _embed: batch_size, out_size
                # return th.sigmoid(self._Fout1(_embed))
            elif self._istriangle and not self._issquare:
                # _embed: batch_size, 5embed_size
                _embed = th.cat(
                    (source_embed, target_embed, source_embed * target_embed, _triangleembed, _triangleneighborembed),
                    dim=1,
                )
                # _embed: batch_size, out_size
                # return th.sigmoid(self._Fout2(_embed))
            elif self._issquare and not self._istriangle:
                # _embed:batch_size, 5embed_size
                _embed = th.cat(
                    (source_embed, target_embed, source_embed * target_embed, _squareembed, _squareneighborembed), dim=1
                )
                # _embed: batch_size, out_size
                # return th.sigmoid(self._Fout2(_embed))
            elif not self._istriangle and not self._issquare:
                # _embed:batch_size, 3embed_size
                _embed = th.cat((source_embed, target_embed, source_embed * target_embed), dim=1)
                # _embed: batch_size, out_size
                # return th.sigmoid(self._Fout3(_embed))
            else:
                raise NotImplementedError
        elif self.combine_type == "product":
            if self._istriangle and self._issquare:
                # _sourceembed: batch_size, 5embed_size
                _sourceembed = th.cat(
                    (source_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), dim=1
                )
                _sourceembed = self._Fcombine5(_sourceembed)
                _targetembed = th.cat(
                    (target_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), dim=1
                )
                _targetembed = self._Fcombine6(_targetembed)
                # _embed: batch_size, 6embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        _sourceembed,
                        _targetembed,
                        source_embed * target_embed,
                        _sourceembed * _targetembed,
                    ),
                    dim=1,
                )
                # return th.sigmoid(self._Fout4(_embed))
            elif self._istriangle and not self._issquare:
                # _sourceembed: batch_size, 3embed_size
                _sourceembed = th.cat((source_embed, triangle_embed, triangleneighbor_embed), dim=1)
                _sourceembed = self._Fcombine7(_sourceembed)
                _targetembed = th.cat((target_embed, triangle_embed, triangleneighbor_embed), dim=1)
                _targetembed = self._Fcombine8(_targetembed)
                # _embed: batch_size, 6embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        _sourceembed,
                        _targetembed,
                        source_embed * target_embed,
                        _sourceembed * _targetembed,
                    ),
                    dim=1,
                )
                # return th.sigmoid(self._Fout4(_embed))
            elif self._issquare and not self._istriangle:
                # _sourceembed: batch_size, 3embed_size
                _sourceembed = th.cat((source_embed, square_embed, squareneighbor_embed), dim=1)
                _sourceembed = self._Fcombine7(_sourceembed)
                _targetembed = th.cat((target_embed, square_embed, squareneighbor_embed), dim=1)
                _targetembed = self._Fcombine8(_targetembed)
                # _embed: batch_size, 6embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        _sourceembed,
                        _targetembed,
                        source_embed * target_embed,
                        _sourceembed * _targetembed,
                    ),
                    dim=1,
                )
                # return th.sigmoid(self._Fout4(_embed))
            else:
                _embed = th.cat((source_embed, target_embed, source_embed * target_embed))
                # return th.sigmoid(self._Fout3(_embed))
        else:
            raise NotImplementedError

        return th.sigmoid(self._Fout(_embed))


class LogicCombineKG(nn.Module):
    def __init__(
        self,
        trianglelogic_num,
        squarelogic_num,
        combine_type,
        combine_embed,
        triangle_signal,
        square_signal,
        in_size,
        out_size,
    ):
        super(LogicCombineKG, self).__init__()
        # combine_type: concat, product
        self.combine_type = combine_type
        self._istriangle = triangle_signal
        self._issquare = square_signal
        self._Fcombine1 = nn.Sequential(
            nn.Linear(in_size * trianglelogic_num * 2, combine_embed),
            nn.ReLU(),
            nn.Linear(combine_embed, combine_embed),
        )
        self._Fcombine2 = nn.Sequential(
            nn.Linear(in_size * trianglelogic_num, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine3 = nn.Sequential(
            nn.Linear(in_size * squarelogic_num * 2, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine4 = nn.Sequential(
            nn.Linear(in_size * squarelogic_num, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine5 = nn.Sequential(
            nn.Linear(combine_embed * 6, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine6 = nn.Sequential(
            nn.Linear(combine_embed * 6, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine7 = nn.Sequential(
            nn.Linear(combine_embed * 4, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine8 = nn.Sequential(
            nn.Linear(combine_embed * 4, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fout1 = nn.Sequential(
            nn.Linear(combine_embed * 8, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )
        self._Fout2 = nn.Sequential(
            nn.Linear(combine_embed * 6, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )
        self._Fout3 = nn.Sequential(
            nn.Linear(combine_embed * 4, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )
        self._Fout4 = nn.Sequential(
            nn.Linear(combine_embed * 7, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )

    def forward(
        self,
        source_embed,
        relation_embed,
        target_embed,
        triangle_embed=None,
        triangleneighbor_embed=None,
        square_embed=None,
        squareneighbor_embed=None,
    ):
        # triangle_embed: batch_size, trianglelogic_num*2, embed_size
        # triangle_embed: batch_size, trianglelogic_num, embed_size
        if self._istriangle:
            # triangle_embed: batch_size, trianglelogic_num*2*embed_size
            _triangleembed = triangle_embed.reshape(
                triangle_embed.shape[0], triangle_embed.shape[1] * triangle_embed.shape[2]
            )
            # _triangleembed: batch_size, embed_size
            _triangleembed = self._Fcombine1(_triangleembed)
            # _triangleneighborembed: batch_size, trianglelogic_num*embed_size
            _triangleneighborembed = triangleneighbor_embed.reshape(
                triangleneighbor_embed.shape[0], triangleneighbor_embed.shape[1] * triangleneighbor_embed.shape[2]
            )
            # _triangleneighborembed: batch_size, embed_size
            _triangleneighborembed = self._Fcombine2(_triangleneighborembed)
        if self._issquare:
            # _squareembed: batch_size, squarelogic_num*2*embed_size
            _squareembed = square_embed.reshape(square_embed.shape[0], square_embed.shape[1] * square_embed.shape[2])
            # _squareembed: batch_size, embed_size
            _squareembed = self._Fcombine3(_squareembed)
            # _squareneighborembed: batch_size, squarelogic_num*embed_size
            _squareneighborembed = squareneighbor_embed.reshape(
                squareneighbor_embed.shape[0], squareneighbor_embed.shape[1] * squareneighbor_embed.shape[2]
            )
            # _squareneighborembed: batch_size, embed_size
            _squareneighborembed = self._Fcombine4(_squareneighborembed)
        if self.combine_type == "concat":
            if self._istriangle and self._issquare:
                # _embed: batch_size, 8embed_size
                # source_embed, target_embed, relation_embed: batch_size, embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        relation_embed,
                        source_embed * target_embed,
                        _triangleembed,
                        _triangleneighborembed,
                        _squareembed,
                        _squareneighborembed,
                    ),
                    dim=1,
                )
                # _embed: batch_size, out_size
                return th.sigmoid(self._Fout1(_embed))
            elif self._istriangle and not self._issquare:
                # _embed: batch_size, 6embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        relation_embed,
                        source_embed * target_embed,
                        _triangleembed,
                        _triangleneighborembed,
                    ),
                    dim=1,
                )
                return th.sigmoid(self._Fout2(_embed))
            elif self._issquare and not self._istriangle:
                # _embed: batch_size, 6embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        target_embed,
                        relation_embed,
                        source_embed * target_embed,
                        _squareembed,
                        _squareneighborembed,
                    ),
                    dim=1,
                )
                return th.sigmoid(self._Fout2(_embed))
            else:
                # _embed: batch_size, 4embed_size
                _embed = th.cat((source_embed, target_embed, relation_embed, source_embed * target_embed), dim=1)
                return th.sigmoid(self._Fout3(_embed))
        elif self.combine_type == "product":
            if self._istriangle and self._issquare:
                # _sourceembed: batch_size, 6embed_size
                _sourceembed = th.cat(
                    (source_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), dim=1
                )
                _sourceembed = self._Fcombine5(_sourceembed)
                _targetembed = th.cat(
                    (target_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), dim=1
                )
                _targetembed = self._Fcombine6(_targetembed)
                # _embed: batch_size, 7embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        relation_embed,
                        target_embed,
                        _sourceembed,
                        _targetembed,
                        source_embed * target_embed,
                        _sourceembed * _targetembed,
                    ),
                    dim=1,
                )
                return th.sigmoid(self._Fout4(_embed))
            elif self._istriangle and not self._issquare:
                # _sourceembed: batch_size, 4embed_size
                _sourceembed = th.cat((source_embed, triangle_embed, triangleneighbor_embed), dim=1)
                _sourceembed = self._Fcombine7(_sourceembed)
                _targetembed = th.cat((target_embed, triangle_embed, triangleneighbor_embed), dim=1)
                _targetembed = self._Fcombine8(_targetembed)
                # _embed: batch_size, 7embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        relation_embed,
                        target_embed,
                        _sourceembed,
                        _targetembed,
                        source_embed * target_embed,
                        _sourceembed * _targetembed,
                    ),
                    dim=1,
                )
                return th.sigmoid(self._Fout4(_embed))
            elif self._issquare and not self._istriangle:
                # _sourceembed: batch_size, 4embed_size
                _sourceembed = th.cat((source_embed, square_embed, squareneighbor_embed), dim=1)
                _sourceembed = self._Fcombine7(_sourceembed)
                _targetembed = th.cat((target_embed, square_embed, squareneighbor_embed), dim=1)
                _targetembed = self._Fcombine8(_targetembed)
                # _embed: batch_size, 7embed_size
                _embed = th.cat(
                    (
                        source_embed,
                        relation_embed,
                        target_embed,
                        _sourceembed,
                        _targetembed,
                        source_embed * target_embed,
                        _sourceembed * _targetembed,
                    ),
                    dim=1,
                )
                return th.sigmoid(self._Fout4(_embed))
            else:
                # _embed: batch_size, 4embed_size
                _embed = th.cat((source_embed, relation_embed, target_embed, source_embed * target_embed), dim=1)
                return th.sigmoid(self._Fout3(_embed))
        else:
            raise NotImplementedError


class LogicCombineNode(nn.Module):
    def __init__(
        self,
        trianglelogic_num,
        squarelogic_num,
        combine_type,
        combine_embed,
        triangle_signal,
        square_signal,
        in_size,
        out_size,
    ):
        super(LogicCombineNode, self).__init__()
        # combine_type: concat
        self.combine_type = combine_type
        self._istriangle = triangle_signal
        self._issquare = square_signal
        self._Fcombine1 = nn.Sequential(
            nn.Linear(in_size * trianglelogic_num * 2, combine_embed),
            nn.ReLU(),
            nn.Linear(combine_embed, combine_embed),
        )
        self._Fcombine2 = nn.Sequential(
            nn.Linear(in_size * trianglelogic_num, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine3 = nn.Sequential(
            nn.Linear(in_size * squarelogic_num * 2, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fcombine4 = nn.Sequential(
            nn.Linear(in_size * squarelogic_num, combine_embed), nn.ReLU(), nn.Linear(combine_embed, combine_embed)
        )
        self._Fout1 = nn.Sequential(
            nn.Linear(combine_embed * 5, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )
        self._Fout2 = nn.Sequential(
            nn.Linear(combine_embed * 3, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )
        self._Fout3 = nn.Sequential(
            nn.Linear(combine_embed, combine_embed), nn.ReLU(), nn.Linear(combine_embed, out_size)
        )

    def forward(
        self, node_embed, triangle_embed=None, triangleneighbor_embed=None, square_embed=None, squareneighbor_embed=None
    ):
        # triangle_embed: batch_size, trianglelogic_num*2, embed_size
        # triangle_embed: batch_size, trianglelogic_num, embed_size
        if self._istriangle:
            # triangle_embed: batch_size, trianglelogic_num*2*embed_size
            _triangleembed = triangle_embed.reshape(
                triangle_embed.shape[0], triangle_embed.shape[1] * triangle_embed.shape[2]
            )
            # _triangleembed: batch_size, embed_size
            _triangleembed = self._Fcombine1(_triangleembed)
            # _triangleneighborembed: batch_size, trianglelogic_num*embed_size
            _triangleneighborembed = triangleneighbor_embed.reshape(
                triangleneighbor_embed.shape[0], triangleneighbor_embed.shape[1] * triangleneighbor_embed.shape[2]
            )
            # _triangleneighborembed: batch_size, embed_size
            _triangleneighborembed = self._Fcombine2(_triangleneighborembed)
        if self._issquare:
            # _squareembed: batch_size, squarelogic_num*2*embed_size
            _squareembed = square_embed.reshape(square_embed.shape[0], square_embed.shape[1] * square_embed.shape[2])
            # _squareembed: batch_size, embed_size
            _squareembed = self._Fcombine3(_squareembed)
            # _squareneighborembed: batch_size, squarelogic_num*embed_size
            _squareneighborembed = squareneighbor_embed.reshape(
                squareneighbor_embed.shape[0], squareneighbor_embed.shape[1] * squareneighbor_embed.shape[2]
            )
            # _squareneighborembed: batch_size, embed_size
            _squareneighborembed = self._Fcombine4(_squareneighborembed)
        if self.combine_type == "concat":
            if self._istriangle and self._issquare:
                # _embed: batch_size, 5embed_size
                _embed = th.cat(
                    (node_embed, _triangleembed, _triangleneighborembed, _squareembed, _squareneighborembed), dim=1
                )
                # _embed: batch_size, out_size
                return th.sigmoid(self._Fout1(_embed))
            elif self._istriangle and not self._issquare:
                # _embed: batch_size, 3embed_size
                _embed = th.cat((node_embed, _triangleembed, _triangleneighborembed), dim=1)
                # _embed: batch_size, out_size
                return th.sigmoid(self._Fout2(_embed))
            elif self._issquare and not self._istriangle:
                # _embed: batch_size, 3embed_size
                _embed = th.cat((node_embed, _squareembed, _squareneighborembed), dim=1)
                # _embed: batch_size, out_size
                return th.sigmoid(self._Fout2(_embed))
            else:
                return th.sigmoid(self._Fout3(node_embed))
        else:
            raise NotImplementedError
