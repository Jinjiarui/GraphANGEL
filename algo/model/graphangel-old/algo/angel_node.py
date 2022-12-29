import math
import os
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from utils.base import glorot

from algo.angel_module import GraphletAggregate, GraphletMessagePass, LogicCombineNode


class Angel(nn.Module):
    def __init__(
        self,
        nodetype_num,
        embed_size,
        triangle_signal,
        square_signal,
        aggregate_type,
        combine_type,
        pass_embed,
        aggregate_embed,
        combine_embed,
        pass_num,
        aggregate_num,
        combine_num,
        trianglelogic_num,
        squarelogic_num,
        triangleneighbor_num,
        squareneighbor_num,
        triangle_num,
        square_num,
        embed_dim,
        in_dropprob,
        out_dropprob,
        node_embedding=None,
        label_num=1,
    ):
        super(Angel, self).__init__()
        # triangle signal, square signal: bool
        self._istriangle = triangle_signal
        self._issquare = square_signal
        # number of logic in each shape
        self._trianglelogicnum = trianglelogic_num
        self._squarelogicnum = squarelogic_num
        self._triangleneighbornum = triangleneighbor_num
        self._squareneighbornum = squareneighbor_num
        self._trianglenum = triangle_num
        self._squarenum = square_num
        self._embedding = nn.Embedding.from_pretrained(node_embedding)  # .requires_grad_()
        # node embedding + node type embedding for heterogeneous graph and node classification
        # node embedding + edge type embedding for knowledge graph

        if self._istriangle:
            # graphlet_type = ["triangle", "triangle", "notriangle", "notriangle"]
            self.TriangleMessagePass = nn.ModuleList()
            self.TriangleNeighborPass = nn.ModuleList()
            # aggregation for positive
            self.TrianglePosAggregate = nn.ModuleList()
            # aggregation for negative
            self.TriangleNegAggregate = nn.ModuleList()
            self.TriangleNeighborAggregate = nn.ModuleList()
            for _ in range(self._trianglelogicnum):
                self.TriangleMessagePass.append(
                    GraphletMessagePass(
                        3, triangle_num, pass_embed, embed_size, embed_size, pass_num, False, in_dropprob, out_dropprob
                    )
                )
                self.TriangleNeighborPass.append(
                    GraphletMessagePass(
                        3,
                        triangleneighbor_num,
                        pass_embed,
                        embed_size,
                        embed_size,
                        pass_num,
                        True,
                        in_dropprob,
                        out_dropprob,
                    )
                )
                self.TrianglePosAggregate.append(
                    GraphletAggregate(
                        aggregate_type,
                        triangle_num,
                        aggregate_embed,
                        embed_size,
                        embed_size,
                        1,
                        False,
                        aggregate_num,
                        in_dropprob,
                        out_dropprob,
                    )
                )
                self.TriangleNegAggregate.append(
                    GraphletAggregate(
                        aggregate_type,
                        triangle_num,
                        aggregate_embed,
                        embed_size,
                        embed_size,
                        -1,
                        False,
                        aggregate_num,
                        in_dropprob,
                        out_dropprob,
                    )
                )
                self.TriangleNeighborAggregate.append(
                    GraphletAggregate(
                        aggregate_type,
                        triangleneighbor_num,
                        aggregate_embed,
                        embed_size,
                        embed_size,
                        1,
                        True,
                        aggregate_num,
                        in_dropprob,
                        out_dropprob,
                    )
                )
        if self._issquare:
            # graphlet_type = ["square", "square", "square", "square", "nosquare", "nosquare", "nosquare", "nosquare"]
            self.SquareMessagePass = nn.ModuleList()
            self.SquareNeighborPass = nn.ModuleList()
            # aggregation for positive
            self.SquarePosAggregate = nn.ModuleList()
            # aggregation for negative
            self.SquareNegAggregate = nn.ModuleList()
            self.SquareNeighborAggregate = nn.ModuleList()
            for _ in range(self._squarelogicnum):
                # square1, square2, square3, square4, nosquare1, nosquare2, nosquare3, nosquare4, square_neighbor1, square_neighbor2, square_neighbor3, square_neighbor4
                self.SquareMessagePass.append(
                    GraphletMessagePass(
                        4, square_num, pass_embed, embed_size, embed_size, pass_num, False, in_dropprob, out_dropprob
                    )
                )
                self.SquareNeighborPass.append(
                    GraphletMessagePass(
                        4,
                        squareneighbor_num,
                        pass_embed,
                        embed_size,
                        embed_size,
                        pass_num,
                        True,
                        in_dropprob,
                        out_dropprob,
                    )
                )
                self.SquarePosAggregate.append(
                    GraphletAggregate(
                        aggregate_type,
                        square_num,
                        aggregate_embed,
                        embed_size,
                        embed_size,
                        1,
                        False,
                        aggregate_num,
                        in_dropprob,
                        out_dropprob,
                    )
                )
                self.SquareNegAggregate.append(
                    GraphletAggregate(
                        aggregate_type,
                        square_num,
                        aggregate_embed,
                        embed_size,
                        embed_size,
                        -1,
                        False,
                        aggregate_num,
                        in_dropprob,
                        out_dropprob,
                    )
                )
                self.SquareNeighborAggregate.append(
                    GraphletAggregate(
                        aggregate_type,
                        squareneighbor_num,
                        aggregate_embed,
                        embed_size,
                        embed_size,
                        1,
                        True,
                        aggregate_num,
                        in_dropprob,
                        out_dropprob,
                    )
                )
        self.LogicCombine = LogicCombineNode(
            trianglelogic_num,
            squarelogic_num,
            combine_type,
            combine_embed,
            triangle_signal,
            square_signal,
            embed_size,
            label_num,
            combine_num,
            in_dropprob,
            out_dropprob,
        )

    def forward(
        self,
        node_id,
        trianglelogic,
        squarelogic,
        triangle,
        notriangle,
        square,
        nosquare,
        triangle_neighbor,
        triangle_mask,
        square_neighbor,
        square_mask,
    ):
        triangle_embed, square_embed = [], []
        triangleneighbor_embed, squareneighbor_embed = [], []
        if self._istriangle:
            # graphlet_logic: batch_size, graphletlogic_num, 3 for triangle (4 for square)
            # graphlet_neighbor: batch_size, graphletlogic_num, graphletneighbor_num, 3 for triangle (4 for square)
            # triangleneighborlogic = th.repeat_interleave(th.unsqueeze(trianglelogic, dim=2), self._triangleneighbornum, dim=2)
            # triangleslogic = th.repeat_interleave(th.unsqueeze(trianglelogic, dim=2), self._trianglenum, dim=2)
            # graphlet: batch_size, graphletlogic_num, graphlet_num, 3 for triangle (4 for square)
            for _logic in range(len(trianglelogic)):
                _triangleneighbors, _triangles, _notriangles = [], [], []
                for _node in range(3):
                    # _embed: batch_size, graphletneighbor_num, embed_size
                    _embed = self._embedding(triangle_neighbor[:, _logic, :, _node])
                    # _embed: batch_size, graphletneighbor_num, embed_size -> batch_size, graphletneighbor_num, 1, embed_size
                    _triangleneighbors.append(_embed)
                    _embed = self._embedding(triangle[:, _logic, :, _node])
                    _triangles.append(_embed)
                    _embed = self._embedding(notriangle[:, _logic, :, _node])
                    _notriangles.append(_embed)
                # _triangleneighborembed: batch_size, graphletneighbor_num, 3 for triangle (4 for square), embed_size
                _triangleneighborembed = th.stack(_triangleneighbors, dim=2)
                _triangleembed = th.stack(_triangles, dim=2)
                _notriangleembed = th.stack(_notriangles, dim=2)
                # _triangleneighborembed: batch_size, graphletneighbor_num, embed_size
                _triangleneighborembed = self.TriangleNeighborPass[_logic](_triangleneighborembed)
                # triangle_mask: batch_size, graphletlogic_num, graphletneighbor_num, 0/1
                # _triangleneighborembed: batch_size, graphletneighbor_num, embed_size -> batch_size, embed_size
                _triangleneighborembed = self.TriangleNeighborAggregate[_logic](
                    _triangleneighborembed, triangle_mask[:, _logic, :, :]
                )
                # triangleneighbor_embed: batch_size, 1, embed_size
                triangleneighbor_embed.append(_triangleneighborembed)
                _triangleembed = self.TriangleMessagePass[_logic](_triangleembed)
                _triangleembed = self.TrianglePosAggregate[_logic](_triangleembed)
                triangle_embed.append(_triangleembed)
                _notriangleembed = self.TriangleMessagePass[_logic](_notriangleembed)
                _notriangleembed = self.TriangleNegAggregate[_logic](_notriangleembed)
                triangle_embed.append(_notriangleembed)
        if self._issquare:
            # squareneighborlogic = th.repeat_interleave(th.unsqueeze(squarelogic, dim=2), self._squareneighbornum, dim=2)
            # squareslogic = th.repeat_interleave(th.unsqueeze(squarelogic, dim=2), self._squarenum, dim=2)
            for _logic in range(len(squarelogic)):
                _squareneighbors, _squares, _nosquares = [], [], []
                for _node in range(4):
                    # _embed: batch_size, graphletneighbor_num, embed_size
                    _embed = self._embedding(square_neighbor[:, _logic, :, _node])
                    # _embed: batch_size, graphletneighbor_num, embed_size -> batch_size, graphletneighbor_num, 1, embed_size
                    _squareneighbors.append(_embed)
                    _embed = self._embedding(square[:, _logic, :, _node])
                    _squares.append(_embed)
                    _embed = self._embedding(nosquare[:, _logic, :, _node])
                    _nosquares.append(_embed)
                _squareneighborembed = th.stack(_squareneighbors, dim=2)
                _squareembed = th.stack(_squares, dim=2)
                _nosquareembed = th.stack(_nosquares, dim=2)
                _squareneighborembed = self.SquareNeighborPass[_logic](_squareneighborembed)
                _squareneighborembed = self.SquareNeighborAggregate[_logic](
                    _squareneighborembed, square_mask[:, _logic, :, :]
                )
                squareneighbor_embed.append(_squareneighborembed)
                _squareembed = self.SquareMessagePass[_logic](_squareembed)
                _squareembed = self.SquarePosAggregate[_logic](_squareembed)
                square_embed.append(_squareembed)
                _nosquareembed = self.SquareMessagePass[_logic](_nosquareembed)
                _nosquareembed = self.SquareNegAggregate[_logic](_nosquareembed)
                square_embed.append(_nosquareembed)
        triangle_embed = th.stack(triangle_embed, dim=1) if self._istriangle else None
        triangleneighbor_embed = th.stack(triangleneighbor_embed, dim=1) if self._istriangle else None
        square_embed = th.stack(square_embed, dim=1) if self._issquare else None
        squareneighbor_embed = th.stack(squareneighbor_embed, dim=1) if self._issquare else None
        return self.LogicCombine(
            self._embedding(node_id), triangle_embed, triangleneighbor_embed, square_embed, squareneighbor_embed
        )

