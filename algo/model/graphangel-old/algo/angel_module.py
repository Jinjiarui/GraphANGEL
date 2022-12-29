import math
import os
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from utils.base import glorot


class GraphletMessagePass(nn.Module):
    def __init__(
        self, graphlet_type, graphlet_num, pass_dim, in_dim, out_dim, layer_num, is_neighbor, in_dropprob, out_dropprob
    ):
        super(GraphletMessagePass, self).__init__()
        # graphlet_type: 3 for triangle, 4 for square
        # is_neighbor: true or false
        self.layer_num = layer_num
        self.is_neighbor = is_neighbor
        self.Flayer = nn.ModuleList()
        self.Fnorm = nn.ModuleList()
        self.Findropout = nn.Dropout(in_dropprob)
        self.Foutdropout = nn.Dropout(out_dropprob)
        self.graphlet_num = graphlet_num
        self.graphlet_type = graphlet_type

        middle_dim = 3 * pass_dim if self.is_neighbor else graphlet_type * pass_dim
        for l in range(self.layer_num):
            if self.is_neighbor:
                in_dims = middle_dim if l > 0 else 3 * in_dim
            else:
                in_dims = middle_dim if l > 0 else graphlet_type * in_dim
            out_dims = middle_dim if l < self.layer_num - 1 else out_dim
            self.Flayer.append(nn.Linear(in_dims, out_dims))
            if l < self.layer_num - 1:
                self.Fnorm.append(nn.BatchNorm1d(graphlet_num))

    def forward(self, graphlet_embed):
        graphlet_embed = self.Findropout(graphlet_embed)
        if self.is_neighbor:
            # _embed: batch_size, 3 * embed_size
            _embed = th.cat(
                (graphlet_embed[:, :, 0], graphlet_embed[:, :, -1], graphlet_embed[:, :, 0] * graphlet_embed[:, :, -1]),
                axis=-1,
            )
        else:
            # batch_size, graphlet_num, triangle (4 for square) * embed_size
            _embed = graphlet_embed.reshape(
                graphlet_embed.shape[0], graphlet_embed.shape[1], graphlet_embed.shape[2] * graphlet_embed.shape[3]
            )
        for l in range(self.layer_num):
            _embed = self.Flayer[l](_embed)
            if l < self.layer_num - 1:
                _embed = self.Fnorm[l](_embed)
                _embed = F.relu(_embed)
        _embed = self.Foutdropout(_embed)
        return _embed


class GraphletAggregate(nn.Module):
    def __init__(
        self,
        aggregate_type,  # aggregate_type: mlp, mean, max, gru
        graphlet_num,
        aggregate_dim,
        in_dim,
        out_dim,
        data_type,  # 1 for positive, -1 for negative
        is_neighbor,
        layer_num,
        in_dropprob,
        out_dropprob,
        padding_len=1,
    ):
        super(GraphletAggregate, self).__init__()
        self.layer_num = layer_num
        self.is_neighbor = is_neighbor
        self.in_dim = in_dim
        self.aggregate_type = aggregate_type
        self.data_type = data_type
        if self.aggregate_type == "mlp":
            self.Flayer = nn.ModuleList()
            self.Fnorm = nn.ModuleList()
            for l in range(self.layer_num):
                in_dims = graphlet_num * aggregate_dim if l > 0 else graphlet_num * in_dim
                out_dims = graphlet_num * aggregate_dim if l < self.layer_num - 1 else out_dim
                self.Flayer.append(nn.Linear(in_dims, out_dims))
                if l < self.layer_num - 1:
                    self.Fnorm.append(nn.BatchNorm1d(out_dims))
        elif self.aggregate_type == "gru":
            self.Flayer = nn.GRU(
                input_size=in_dim, hidden_size=aggregate_dim, num_layers=int(graphlet_num // 2), bidirectional=True
            )
            self.Fgru = nn.Sequential(
                nn.Linear(2 * aggregate_dim, aggregate_dim), nn.ReLU(), nn.Linear(aggregate_dim, out_dim)
            )
        self.Fpadding = nn.Sequential(
            nn.Linear(in_dim + padding_len, aggregate_dim), nn.ReLU(), nn.Linear(aggregate_dim, aggregate_dim)
        )

        self.padding_len = padding_len
        self.Findropout = nn.Dropout(in_dropprob)
        self.Foutdropout = nn.Dropout(out_dropprob)

    def forward(self, graphlet_embed, graphlet_mask=None):
        graphlet_embed = self.Findropout(graphlet_embed)
        if not self.is_neighbor:
            # graphlet_embed: batch_size, graphlet_num, embed_size+padding_len
            graphlet_embed = th.cat(
                (
                    graphlet_embed,
                    th.ones(
                        graphlet_embed.shape[0], graphlet_embed.shape[1], self.padding_len, device=graphlet_embed.device
                    )
                    * self.data_type,
                ),
                dim=-1,
            )
            # graphlet_embed: batch_size, graphlet_num, embed_size
            graphlet_embed = self.Fpadding(graphlet_embed)
        if graphlet_mask is not None:
            # graphlet_mask: batch_size, graphlet_num, 0/1
            graphlet_mask = th.repeat_interleave(graphlet_mask, self.in_dim, dim=1)
            graphlet_mask = graphlet_mask.reshape(graphlet_mask.shape[0], graphlet_embed.shape[1], self.in_dim)
            graphlet_embed = graphlet_embed * graphlet_mask
        if self.aggregate_type == "mean":
            _embed = th.mean(graphlet_embed, dim=1)  # batch_size, embed_size
        elif self.aggregate_type == "gru":
            _embed = graphlet_embed.reshape(
                graphlet_embed.shape[1], graphlet_embed.shape[0], graphlet_embed.shape[2]
            )  # graphlet_num, batch_size, embed_size
            # _embed: graphlet_num, batch_size, embed_size -> graphlet_num, batch_size, 2embed_size
            _embed, _ = self.Flayer(_embed, _embed)
            _embed = _embed.reshape(
                _embed.shape[1], _embed.shape[0] * _embed.shape[2]
            )  # batch_size, graphlet_num * 2embed_size
            _embed = self.Fgru(_embed)
        elif self.aggregate_type == "max":
            _embed = th.max(graphlet_embed, dim=1)  # batch_size, embed_size
        elif self.aggregate_type == "mlp":
            _embed = graphlet_embed.reshape(graphlet_embed.shape[0], graphlet_embed.shape[1] * graphlet_embed.shape[2])
            for l in range(self.layer_num):
                _embed = self.Flayer[l](_embed)
                if l < self.layer_num - 1:
                    _embed = self.Fnorm[l](_embed)
                    _embed = F.relu(_embed)
        else:
            raise NotImplementedError
        _embed = self.Foutdropout(_embed)
        return _embed


class LogicCombineHG(nn.Module):
    def __init__(
        self,
        trianglelogic_num,
        squarelogic_num,
        combine_type,
        combine_embed,
        triangle_signal,
        square_signal,
        in_dim,
        out_dim,
        layer_num,
        in_dropprob,
        out_dropprob,
    ):
        super(LogicCombineHG, self).__init__()
        self.combine_type = combine_type
        self.is_triangle = triangle_signal
        self.is_square = square_signal
        self.layer_num = layer_num
        if combine_type == "mlp":
            self.Fcombinetriangle = nn.Sequential(
                nn.Linear(in_dim * 2 * trianglelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombineneighbortriangle = nn.Sequential(
                nn.Linear(in_dim * trianglelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombinesquare = nn.Sequential(
                nn.Linear(in_dim * 2 * squarelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombineneighborsquare = nn.Sequential(
                nn.Linear(in_dim * squarelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
        self.Flayer = nn.ModuleList()
        self.Fnorm = nn.ModuleList()

        for l in range(self.layer_num):
            if self.is_triangle and self.is_square:
                in_dims = 7 * combine_embed if l > 0 else 3 * in_dim + 4 * combine_embed
                out_dims = 7 * combine_embed if l < self.layer_num - 1 else out_dim
            elif self.is_triangle or self.is_square:
                in_dims = 5 * combine_embed if l > 0 else 3 * in_dim + 2 * combine_embed
                out_dims = 5 * combine_embed if l < self.layer_num - 1 else out_dim
            else:
                in_dims = 3 * combine_embed if l > 0 else 3 * in_dim
                out_dims = 3 * combine_embed if l < self.layer_num - 1 else out_dim
            self.Flayer.append(nn.Linear(in_dims, out_dims))
            if l < self.layer_num - 1:
                self.Fnorm.append(nn.BatchNorm1d(out_dims))
        self.Findropout = nn.Dropout(in_dropprob)
        self.Foutdropout = nn.Dropout(out_dropprob)

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
        # squareembed: batch_size, squarelogic_num*2*embed_size
        # squareneighborembed: batch_size, squarelogic_num*embed_size
        source_embed = self.Findropout(source_embed)
        target_embed = self.Findropout(target_embed)
        if self.combine_type == "mean":
            if self.is_triangle:
                triangle_embed = th.mean(triangle_embed, axis=1)  # batch_size, embed_size
                triangleneighbor_embed = th.mean(triangleneighbor_embed, axis=1)  # batch_size, embed_size
            if self.is_square:
                square_embed = th.mean(square_embed, axis=1)
                squareneighbor_embed = th.mean(squareneighbor_embed, axis=1)
        elif self.combine_type == "max":
            if self.is_triangle:
                triangle_embed = th.max(triangle_embed, axis=1)  # batch_size, embed_size
                triangleneighbor_embed = th.max(triangleneighbor_embed, axis=1)  # batch_size, embed_size
            if self.is_square:
                square_embed = th.max(square_embed, axis=1)
                squareneighbor_embed = th.max(squareneighbor_embed, axis=1)
        elif self.combine_type == "mlp":
            if self.is_triangle:
                triangle_embed = self.Fcombinetriangle(
                    triangle_embed.reshape(triangle_embed.shape[0], triangle_embed.shape[1] * triangle_embed.shape[2])
                )
                triangleneighbor_embed = self.Fcombineneighbortriangle(
                    triangleneighbor_embed.reshape(
                        triangleneighbor_embed.shape[0],
                        triangleneighbor_embed.shape[1] * triangleneighbor_embed.shape[2],
                    )
                )
            if self.is_square:
                square_embed = self.Fcombinesquare(
                    square_embed.reshape(square_embed.shape[0], square_embed.shape[1] * square_embed.shape[2])
                )
                squareneighbor_embed = self.Fcombineneighborsquare(
                    squareneighbor_embed.reshape(
                        squareneighbor_embed.shape[0], squareneighbor_embed.shape[1] * squareneighbor_embed.shape[2]
                    )
                )
        else:
            raise NotImplementedError
        if self.is_triangle and self.is_square:
            _embed = th.cat(
                (
                    source_embed,
                    target_embed,
                    source_embed * target_embed,
                    triangle_embed,
                    triangleneighbor_embed,
                    square_embed,
                    squareneighbor_embed,
                ),
                axis=-1,
            )
        elif self.is_triangle and not self.is_square:
            _embed = th.cat(
                (source_embed, target_embed, source_embed * target_embed, triangle_embed, triangleneighbor_embed),
                axis=-1,
            )
        elif self.is_square and not self.is_triangle:
            _embed = th.cat(
                (source_embed, target_embed, source_embed * target_embed, square_embed, squareneighbor_embed), axis=-1
            )
        else:
            _embed = th.cat((source_embed, target_embed, source_embed * target_embed), axis=-1)
        for l in range(self.layer_num):
            _embed = self.Flayer[l](_embed)
            if l < self.layer_num - 1:
                _embed = self.Fnorm[l](_embed)
                _embed = F.relu(_embed)
        _embed = self.Foutdropout(_embed)
        return th.sigmoid(_embed)


class LogicCombineKG(nn.Module):
    def __init__(
        self,
        trianglelogic_num,
        squarelogic_num,
        combine_type,
        combine_embed,
        triangle_signal,
        square_signal,
        in_dim,
        out_dim,
        layer_num,
        in_dropprob,
        out_dropprob,
    ):
        super(LogicCombineKG, self).__init__()
        self.combine_type = combine_type
        self.is_triangle = triangle_signal
        self.is_square = square_signal
        self.layer_num = layer_num
        if combine_type == "mlp":
            self.Fcombinetriangle = nn.Sequential(
                nn.Linear(in_dim * 2 * trianglelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombineneighbortriangle = nn.Sequential(
                nn.Linear(in_dim * trianglelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombinesquare = nn.Sequential(
                nn.Linear(in_dim * 2 * squarelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombineneighborsquare = nn.Sequential(
                nn.Linear(in_dim * squarelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
        self.Flayer = nn.ModuleList()
        self.Fnorm = nn.ModuleList()
        for l in range(self.layer_num):
            if self.is_triangle and self.is_square:
                in_dims = 8 * combine_embed if l > 0 else 3 * in_dim + 5 * combine_embed
                out_dims = 8 * combine_embed if l < self.layer_num - 1 else out_dim
            elif self.is_triangle or self.is_square:
                in_dims = 6 * combine_embed if l > 0 else 3 * in_dim + 3 * combine_embed
                out_dims = 6 * combine_embed if l < self.layer_num - 1 else out_dim
            else:
                in_dims = 4 * combine_embed if l > 0 else 3 * in_dim + 1 * combine_embed
                out_dims = 4 * combine_embed if l < self.layer_num - 1 else out_dim
            self.Flayer.append(nn.Linear(in_dims, out_dims))
            if l < self.layer_num - 1:
                self.Fnorm.append(nn.BatchNorm1d(out_dims))
        self.Findropout = nn.Dropout(in_dropprob)
        self.Foutdropout = nn.Dropout(out_dropprob)

        self.combine_embed = combine_embed

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
        # triangleneighbor_embed: batch_size, trianglelogic_num, embed_size
        # squareembed: batch_size, squarelogic_num*2*embed_size
        # squareneighborembed: batch_size, squarelogic_num*embed_size
        source_embed = self.Findropout(source_embed)
        relation_embed = self.Findropout(relation_embed)
        target_embed = self.Findropout(target_embed)
        if self.combine_type == "mean":
            if self.is_triangle:
                triangle_embed = th.mean(triangle_embed, axis=1)  # batch_size, embed_size
                triangleneighbor_embed = th.mean(triangleneighbor_embed, axis=1)  # batch_size, embed_size
            if self.is_square:
                square_embed = th.mean(square_embed, axis=1)
                squareneighbor_embed = th.mean(squareneighbor_embed, axis=1)
        elif self.combine_type == "max":
            if self.is_triangle:
                triangle_embed = th.max(triangle_embed, axis=1)  # batch_size, embed_size
                triangleneighbor_embed = th.max(triangleneighbor_embed, axis=1)  # batch_size, embed_size
            if self.is_square:
                square_embed = th.max(square_embed, axis=1)
                squareneighbor_embed = th.max(squareneighbor_embed, axis=1)
        elif self.combine_type == "mlp":
            if self.is_triangle:
                triangle_embed = self.Fcombinetriangle(
                    triangle_embed.reshape(triangle_embed.shape[0], triangle_embed.shape[1] * triangle_embed.shape[2])
                )
                triangleneighbor_embed = self.Fcombineneighbortriangle(
                    triangleneighbor_embed.reshape(
                        triangleneighbor_embed.shape[0],
                        triangleneighbor_embed.shape[1] * triangleneighbor_embed.shape[2],
                    )
                )
            if self.is_square:
                square_embed = self.Fcombinesquare(
                    square_embed.reshape(square_embed.shape[0], square_embed.shape[1] * square_embed.shape[2])
                )
                squareneighbor_embed = self.Fcombineneighborsquare(
                    squareneighbor_embed.reshape(
                        squareneighbor_embed.shape[0], squareneighbor_embed.shape[1] * squareneighbor_embed.shape[2]
                    )
                )
        else:
            raise NotImplementedError
        # print("=======BUG========") # DEBUG
        # print(self.combine_embed)
        # print(source_embed.shape)
        # print(relation_embed.shape)
        # print(target_embed.shape)
        # print(triangleneighbor_embed.shape)
        # print(square_embed.shape)
        # print(squareneighbor_embed.shape)
        if self.is_triangle and self.is_square:
            _embed = th.cat(
                (
                    source_embed,
                    relation_embed,
                    target_embed,
                    source_embed * target_embed,
                    triangle_embed,
                    triangleneighbor_embed,
                    square_embed,
                    squareneighbor_embed,
                ),
                axis=-1,
            )
        elif self.is_triangle and not self.is_square:
            _embed = th.cat(
                (
                    source_embed,
                    relation_embed,
                    target_embed,
                    source_embed * target_embed,
                    triangle_embed,
                    triangleneighbor_embed,
                ),
                axis=-1,
            )
        elif self.is_square and not self.is_triangle:
            _embed = th.cat(
                (
                    source_embed,
                    relation_embed,
                    target_embed,
                    source_embed * target_embed,
                    square_embed,
                    squareneighbor_embed,
                ),
                axis=-1,
            )
        else:
            _embed = th.cat((source_embed, relation_embed, target_embed, source_embed * target_embed), axis=-1)
        for l in range(self.layer_num):
            _embed = self.Flayer[l](_embed)
            if l < self.layer_num - 1:
                _embed = self.Fnorm[l](_embed)
                _embed = F.relu(_embed)
                _embed = self.Foutdropout(_embed)

        return th.sigmoid(_embed)


class LogicCombineNode(nn.Module):
    def __init__(
        self,
        trianglelogic_num,
        squarelogic_num,
        combine_type,
        combine_embed,
        triangle_signal,
        square_signal,
        in_dim,
        out_dim,
        layer_num,
        in_dropprob,
        out_dropprob,
    ):
        super(LogicCombineNode, self).__init__()
        self.combine_type = combine_type
        self.is_triangle = triangle_signal
        self.is_square = square_signal
        self.layer_num = layer_num
        if combine_type == "mlp":
            self.Fcombinetriangle = nn.Sequential(
                nn.Linear(in_dim * 2 * trianglelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombineneighbortriangle = nn.Sequential(
                nn.Linear(in_dim * trianglelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombinesquare = nn.Sequential(
                nn.Linear(in_dim * 2 * squarelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
            self.Fcombineneighborsquare = nn.Sequential(
                nn.Linear(in_dim * squarelogic_num, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )
        self.Flayer = nn.ModuleList()
        self.Fnorm = nn.ModuleList()
        for l in range(self.layer_num):
            if self.is_triangle and self.is_square:
                in_dims = 5 * combine_embed if l > 0 else 1 * in_dim + 4 * combine_embed
                out_dims = 5 * combine_embed if l < self.layer_num - 1 else out_dim
            elif self.is_triangle or self.is_square:
                in_dims = 3 * combine_embed if l > 0 else 1 * in_dim + 2 * combine_embed
                out_dims = 3 * combine_embed if l < self.layer_num - 1 else out_dim
            else:
                in_dims = 1 * combine_embed if l > 0 else 1 * in_dim
                out_dims = 1 * combine_embed if l < self.layer_num - 1 else out_dim
            self.Flayer.append(nn.Linear(in_dims, out_dims))
            if l < self.layer_num - 1:
                self.Fnorm.append(nn.BatchNorm1d(out_dims))
        self.Findropout = nn.Dropout(in_dropprob)
        self.Foutdropout = nn.Dropout(out_dropprob)

    def forward(
        self, node_embed, triangle_embed=None, triangleneighbor_embed=None, square_embed=None, squareneighbor_embed=None
    ):
        # triangle_embed: batch_size, trianglelogic_num*2, embed_size
        # triangleneighbor_embed: batch_size, trianglelogic_num, embed_size
        # squareembed: batch_size, squarelogic_num*2*embed_size
        # squareneighborembed: batch_size, squarelogic_num*embed_size
        node_embed = self.Findropout(node_embed)
        if self.combine_type == "mean":
            if self.is_triangle:
                triangle_embed = th.mean(triangle_embed, axis=1)  # batch_size, embed_size
                triangleneighbor_embed = th.mean(triangleneighbor_embed, axis=1)  # batch_size, embed_size
            if self.is_square:
                square_embed = th.mean(square_embed, axis=1)
                squareneighbor_embed = th.mean(squareneighbor_embed, axis=1)
        elif self.combine_type == "max":
            if self.is_triangle:
                triangle_embed = th.max(triangle_embed, axis=1)  # batch_size, embed_size
                triangleneighbor_embed = th.max(triangleneighbor_embed, axis=1)  # batch_size, embed_size
            if self.is_square:
                square_embed = th.max(square_embed, axis=1)
                squareneighbor_embed = th.max(squareneighbor_embed, axis=1)
        elif self.combine_type == "mlp":
            if self.is_triangle:
                triangle_embed = self.Fcombinetriangle(
                    triangle_embed.reshape(triangle_embed.shape[0], triangle_embed.shape[1] * triangle_embed.shape[2])
                )
                triangleneighbor_embed = self.Fcombineneighbortriangle(
                    triangleneighbor_embed.reshape(
                        triangleneighbor_embed.shape[0],
                        triangleneighbor_embed.shape[1] * triangleneighbor_embed.shape[2],
                    )
                )
            if self.is_square:
                square_embed = self.Fcombinesquare(
                    square_embed.reshape(square_embed.shape[0], square_embed.shape[1] * square_embed.shape[2])
                )
                squareneighbor_embed = self.Fcombineneighborsquare(
                    squareneighbor_embed.reshape(
                        squareneighbor_embed.shape[0], squareneighbor_embed.shape[1] * squareneighbor_embed.shape[2]
                    )
                )
        else:
            raise NotImplementedError

        if self.is_triangle and self.is_square:
            _embed = th.cat(
                (node_embed, triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed), axis=-1
            )
        elif self.is_triangle and not self.is_square:
            _embed = th.cat((node_embed, triangle_embed, triangleneighbor_embed), axis=-1)
        elif self.is_square and not self.is_triangle:
            _embed = th.cat((node_embed, square_embed, squareneighbor_embed), axis=-1)
        else:
            _embed = node_embed

        for l in range(self.layer_num):
            _embed = self.Flayer[l](_embed)
            if l < self.layer_num - 1:
                _embed = self.Fnorm[l](_embed)
                _embed = F.relu(_embed)
        _embed = self.Foutdropout(_embed)

        return th.sigmoid(_embed)
