from __future__ import absolute_import, division, print_function

import argparse
import gc
import math
import os
import pickle as pkl
import sys

import dgl
import numpy as np
import torch as th
import torch.nn as nn
from dgl.data import FB15k237Dataset, WN18Dataset

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basedir, ".."))
from algo.angel_hg import Angel as AngelHG
from algo.angel_kg import Angel as AngelKG
from algo.angel_node import Angel as AngelNode
from tensorboardX import SummaryWriter
from utils.base import cal_acc, cal_auc
from utils.data_loader import DataLoader

logdir = os.path.join(basedir, "../log/")
datadir = os.path.join(basedir, "../data/")
savedir = os.path.join(logdir, "model/")


def training_angel(
    task,
    traindata,
    evaldata,
    logname,
    model,
    device,
    batchsize,
    optimizer,
    epoch,
    labelnum,
    savedir,
    modelname,
    patience,
    istriangle=False,
    issquare=False,
):
    traindir = logname + "_train"
    evaldir = logname + "_eval"
    if not os.path.exists(traindir):
        os.mkdir(traindir)
    if not os.path.exists(evaldir):
        os.mkdir(evaldir)
    trainwriter = SummaryWriter(log_dir=traindir)
    evalwriter = SummaryWriter(log_dir=evaldir)

    early_stop = False
    _count = 0
    for _epoch in range(epoch):
        if early_stop:
            break

        train_loss, train_acc, train_auc = [], [], []
        eval_loss, eval_acc, eval_auc = [], [], []
        print("===== EPOCH {} =====".format(_epoch))
        logfile.write("Epoch {} >>".format(_epoch))
        for _iter in range(int(len(traindata) / batchsize)):
            # print("iter:", _iter)
            model.train()
            batch_index = list(np.random.choice(len(traindata) - 1, batchsize))
            optimizer.zero_grad()
            (
                sample_datainput,
                sample_label,
                sample_trianglelogic,
                sample_triangleneighbor,
                sample_trianglemask,
                sample_squarelogic,
                sample_squareneighbor,
                sample_squaremask,
                sample_triangles,
                sample_notriangles,
                sample_squares,
                sample_nosquares,
            ) = traindata.sample(batch_index)
            if task == "hg":
                # data_input: batch_size, (source, target)
                print("====== TRAINING =======")
                sample_trianglelogic = (
                    th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
                )
                sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
                sample_notriangles = th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
                sample_triangleneighbor = (
                    th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
                )
                sample_trianglemask = th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
                sample_squarelogic = th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
                sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
                sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
                sample_squareneighbor = (
                    th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
                )
                sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
                sample_label = th.tensor(sample_label, device=device).to(th.float)
                predictions = model(
                    th.tensor(sample_datainput, device=device)[:, 0].to(th.long),
                    th.tensor(sample_datainput, device=device)[:, 1].to(th.long),
                    sample_trianglelogic,
                    sample_squarelogic,
                    sample_triangles,
                    sample_notriangles,
                    sample_squares,
                    sample_nosquares,
                    sample_triangleneighbor,
                    sample_trianglemask,
                    sample_squareneighbor,
                    sample_squaremask,
                )
            elif task == "kg":
                # data_input: batch_size, (source, relation, target)
                # print("====== TRAINING =======")
                sample_trianglelogic = (
                    th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
                )
                sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
                sample_notriangles = th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
                sample_triangleneighbor = (
                    th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
                )
                sample_trianglemask = th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
                sample_squarelogic = th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
                sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
                sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
                sample_squareneighbor = (
                    th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
                )
                sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
                sample_label = th.tensor(sample_label, device=device).to(th.float)
                predictions = model(
                    th.tensor(sample_datainput, device=device)[:, 0].to(th.long),
                    th.tensor(sample_datainput, device=device)[:, 1].to(th.long),
                    th.tensor(sample_datainput, device=device)[:, 2].to(th.long),
                    sample_trianglelogic,
                    sample_squarelogic,
                    sample_triangles,
                    sample_notriangles,
                    sample_squares,
                    sample_nosquares,
                    sample_triangleneighbor,
                    sample_trianglemask,
                    sample_squareneighbor,
                    sample_squaremask,
                )
            elif task == "node":
                # data_input: batch_size, (source)
                print("====== TRAINING =======")
                sample_trianglelogic = (
                    th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
                )
                sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
                sample_notriangles = th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
                sample_triangleneighbor = (
                    th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
                )
                sample_trianglemask = th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
                sample_squarelogic = th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
                sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
                sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
                sample_squareneighbor = (
                    th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
                )
                sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
                sample_label = th.tensor(sample_label, device=device).to(th.float)
                predictions = model(
                    th.tensor(sample_datainput)[:, 0].to(th.long).to(device),
                    sample_trianglelogic,
                    sample_squarelogic,
                    sample_triangles,
                    sample_notriangles,
                    sample_squares,
                    sample_nosquares,
                    sample_triangleneighbor,
                    sample_trianglemask,
                    sample_squareneighbor,
                    sample_squaremask,
                )
            else:
                raise NotImplementedError
            # label: batch_size, label_num
            losses = criterion(predictions, sample_label)
            losses.backward()
            optimizer.step()
            _auc = cal_auc(predictions.detach().cpu().numpy(), sample_label.detach().cpu().numpy(), labelnum)
            _acc = cal_acc(predictions.detach().cpu().numpy(), sample_label.detach().cpu().numpy(), labelnum)
            train_loss.append(losses.item())
            train_auc.append(_auc)
            train_acc.append(_acc)
            if _iter % 10 == 0:
                trainwriter.add_scalar("train_loss", losses.item(), _iter)
                trainwriter.add_scalar("train_acc", _acc, _iter)
                trainwriter.add_scalar("train_auc", _auc, _iter)
                if _iter % 100 == 0:
                    print(
                        "===== EPOCH {:d} | LOSS {:.4f} | AUC {:.4f} | ACC {:4f}".format(
                            _epoch, losses.item(), _auc, _acc
                        )
                    )
                    logfile.write(
                        "Epoch {:d} | Loss {:.4f} | AUC {:.4f} | ACC {:4f}".format(_epoch, losses.item(), _auc, _acc)
                    )
        # with th.cuda.device(device):
        #     th.cuda.empty_cache()
        print(
            "===== TRAIN LOSS {:.4f} | TRAIN AUC {:.4f} | TRAIN ACC {:4f}".format(
                np.mean(train_loss), np.mean(train_auc), np.mean(train_acc)
            )
        )
        logfile.write(
            "TRAIN Loss {:.4f} | TRAIN AUC {:.4f} | TRAIN ACC {:4f}".format(
                np.mean(train_loss), np.mean(train_auc), np.mean(train_acc)
            )
        )
        # eval

        print("eval")
        model.eval()
        with th.no_grad():
            _lastloss = float("inf")
            for _iter in range(int(len(evaldata) // batchsize)):
                # print("eval iter:", _iter)
                batch_index = list(np.random.choice(len(evaldata) - 1, batchsize))
                (
                    sample_datainput,
                    sample_label,
                    sample_trianglelogic,
                    sample_triangleneighbor,
                    sample_trianglemask,
                    sample_squarelogic,
                    sample_squareneighbor,
                    sample_squaremask,
                    sample_triangles,
                    sample_notriangles,
                    sample_squares,
                    sample_nosquares,
                ) = evaldata.sample(batch_index)
                if task == "hg":
                    # data_input: batch_size, (source, target)
                    # print("====== EVALING =======")
                    sample_trianglelogic = (
                        th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
                    )
                    sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
                    sample_notriangles = (
                        th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
                    )
                    sample_triangleneighbor = (
                        th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
                    )
                    sample_trianglemask = (
                        th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
                    )
                    sample_squarelogic = (
                        th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
                    )
                    sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
                    sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
                    sample_squareneighbor = (
                        th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
                    )
                    sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
                    sample_label = th.tensor(sample_label, device=device).to(th.float)
                    predictions = model(
                        th.tensor(sample_datainput)[:, 0].to(th.long).to(device),
                        th.tensor(sample_datainput)[:, 1].to(th.long).to(device),
                        th.tensor(sample_trianglelogic).to(th.long).to(device),
                        th.tensor(sample_squarelogic).to(th.long).to(device),
                        th.tensor(sample_triangles).to(th.long).to(device),
                        th.tensor(sample_notriangles).to(th.long).to(device),
                        th.tensor(sample_squares).to(th.long).to(device),
                        th.tensor(sample_nosquares).to(th.long).to(device),
                        th.tensor(sample_triangleneighbor).to(th.long).to(device),
                        th.tensor(sample_trianglemask).to(th.long).to(device),
                        th.tensor(sample_squareneighbor).to(th.long).to(device),
                        th.tensor(sample_squaremask).to(th.long).to(device),
                    )
                elif task == "kg":
                    # data_input: batch_size, (source, relation, target)
                    # print("====== EVALING =======")
                    sample_trianglelogic = (
                        th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
                    )
                    sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
                    sample_notriangles = (
                        th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
                    )
                    sample_triangleneighbor = (
                        th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
                    )
                    sample_trianglemask = (
                        th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
                    )
                    sample_squarelogic = (
                        th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
                    )
                    sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
                    sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
                    sample_squareneighbor = (
                        th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
                    )
                    sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
                    sample_label = th.tensor(sample_label, device=device).to(th.float)
                    predictions = model(
                        th.tensor(sample_datainput, device=device)[:, 0].to(th.long),
                        th.tensor(sample_datainput, device=device)[:, 1].to(th.long),
                        th.tensor(sample_datainput, device=device)[:, 2].to(th.long),
                        sample_trianglelogic,
                        sample_squarelogic,
                        sample_triangles,
                        sample_notriangles,
                        sample_squares,
                        sample_nosquares,
                        sample_triangleneighbor,
                        sample_trianglemask,
                        sample_squareneighbor,
                        sample_squaremask,
                    )
                elif task == "node":
                    # data_input: batch_size, (source)
                    # print("====== EVALING =======")
                    sample_trianglelogic = (
                        th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
                    )
                    sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
                    sample_notriangles = (
                        th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
                    )
                    sample_triangleneighbor = (
                        th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
                    )
                    sample_trianglemask = (
                        th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
                    )
                    sample_squarelogic = (
                        th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
                    )
                    sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
                    sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
                    sample_squareneighbor = (
                        th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
                    )
                    sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
                    sample_label = th.tensor(sample_label, device=device).to(th.float)
                    predictions = model(
                        th.tensor(sample_datainput)[:, 0].to(th.long).to(device),
                        sample_trianglelogic,
                        sample_squarelogic,
                        sample_triangles,
                        sample_notriangles,
                        sample_squares,
                        sample_nosquares,
                        sample_triangleneighbor,
                        sample_trianglemask,
                        sample_squareneighbor,
                        sample_squaremask,
                    )
                else:
                    raise NotImplementedError
                losses = criterion(predictions, sample_label)
                _auc = cal_auc(predictions.detach().cpu().numpy(), sample_label.detach().cpu().numpy(), labelnum)
                _acc = cal_acc(predictions.detach().cpu().numpy(), sample_label.detach().cpu().numpy(), labelnum)
                eval_loss.append(losses.item())
                eval_auc.append(_auc)
                eval_acc.append(_acc)
                # gc.collect()
                with th.cuda.device(device):
                    th.cuda.empty_cache()
                if _iter % 10 == 0:
                    # evalwriter.add_scalar("eval_loss", losses.item(), _iter)
                    evalwriter.add_scalar("eval_acc", _acc, _iter)
                    evalwriter.add_scalar("eval_auc", _auc, _iter)
                # if np.mean(eval_acc) > np.max(train_acc):
                #     logfile.write("===== SAVING BEST ======")
                #     if not os.path.exists(savedir):
                #         os.makedirs(savedir, exist_ok=True)
                #     th.save(model.state_dict(), os.path.join(savedir, modelname + ".pth"))

            mean_eval_loss = np.mean(eval_loss)
            print(mean_eval_loss, _lastloss)

            if mean_eval_loss >= _lastloss:
                _count += 1
            else:
                _count = 0
                _lastloss = mean_eval_loss
            # _lastloss = min(eval_loss, mean_eval_loss)

            if _count >= patience:
                print("===== EARLY STOP =====")
                early_stop = True

            print(
                "===== EVAL LOSS {:.4f} | EVAL AUC {:.4f} | EVAL ACC {:4f}".format(
                    mean_eval_loss, np.mean(eval_auc), np.mean(eval_acc)
                )
            )
            # logfile.write(
            #     "EVAL Loss {:.4f} | TRAIN AUC {:.4f} | TRAIN ACC {:4f}".format(
            #         np.mean(train_loss), np.mean(train_auc), np.mean(train_acc)
            #     )
            # )


def testing_angel(task, testdata, logname, model, device, batchsize, labelnum, istriangle=False, issquare=False):
    testdir = logname + "_test"
    if not os.path.exists(testdir):
        os.mkdir(testdir)
    testwriter = SummaryWriter(log_dir=testdir)
    test_loss, test_auc, test_acc = [], [], []
    with th.no_grad():
        batch_index = list(np.range(len(testdata)))
        (
            sample_datainput,
            sample_label,
            sample_trianglelogic,
            sample_triangleneighbor,
            sample_trianglemask,
            sample_squarelogic,
            sample_squareneighbor,
            sample_squaremask,
            sample_triangles,
            sample_notriangles,
            sample_squares,
            sample_nosquares,
        ) = testdata.sample(batch_index)
        if task == "hg":
            # data_input: batch_size, (source, target)
            print("====== TESTING =======")
            sample_trianglelogic = th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
            sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
            sample_notriangles = th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
            sample_triangleneighbor = (
                th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
            )
            sample_trianglemask = th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
            sample_squarelogic = th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
            sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
            sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
            sample_squareneighbor = th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
            sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
            sample_label = th.tensor(sample_label, device=device).to(th.float)
            predictions = model(
                th.tensor(sample_datainput)[:, 0].to(th.long).to(device),
                th.tensor(sample_datainput)[:, 1].to(th.long).to(device),
                th.tensor(sample_trianglelogic).to(th.long).to(device),
                th.tensor(sample_squarelogic).to(th.long).to(device),
                th.tensor(sample_triangles).to(th.long).to(device),
                th.tensor(sample_notriangles).to(th.long).to(device),
                th.tensor(sample_squares).to(th.long).to(device),
                th.tensor(sample_nosquares).to(th.long).to(device),
                th.tensor(sample_triangleneighbor).to(th.long).to(device),
                th.tensor(sample_trianglemask).to(th.long).to(device),
                th.tensor(sample_squareneighbor).to(th.long).to(device),
                th.tensor(sample_squaremask).to(th.long).to(device),
            )
        elif task == "kg":
            # data_input: batch_size, (source, relation, target)
            print("====== TESTING =======")
            sample_trianglelogic = th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
            sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
            sample_notriangles = th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
            sample_triangleneighbor = (
                th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
            )
            sample_trianglemask = th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
            sample_squarelogic = th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
            sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
            sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
            sample_squareneighbor = th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
            sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
            sample_label = th.tensor(sample_label, device=device).to(th.float)
            predictions = model(
                th.tensor(sample_datainput, device=device)[:, 0].to(th.long),
                th.tensor(sample_datainput, device=device)[:, 1].to(th.long),
                th.tensor(sample_datainput, device=device)[:, 2].to(th.long),
                sample_trianglelogic,
                sample_squarelogic,
                sample_triangles,
                sample_notriangles,
                sample_squares,
                sample_nosquares,
                sample_triangleneighbor,
                sample_trianglemask,
                sample_squareneighbor,
                sample_squaremask,
            )
        elif task == "node":
            # data_input: batch_size, (source)
            print("====== TESTING =======")
            sample_trianglelogic = th.tensor(sample_trianglelogic, device=device).to(th.long) if istriangle else None
            sample_triangles = th.tensor(sample_triangles, device=device).to(th.long) if istriangle else None
            sample_notriangles = th.tensor(sample_notriangles, device=device).to(th.long) if istriangle else None
            sample_triangleneighbor = (
                th.tensor(sample_triangleneighbor, device=device).to(th.long) if istriangle else None
            )
            sample_trianglemask = th.tensor(sample_trianglemask, device=device).to(th.long) if istriangle else None
            sample_squarelogic = th.tensor(sample_squarelogic, device=device).to(th.long) if istriangle else None
            sample_squares = th.tensor(sample_squares, device=device).to(th.long) if istriangle else None
            sample_nosquares = th.tensor(sample_nosquares, device=device).to(th.long) if issquare else None
            sample_squareneighbor = th.tensor(sample_squareneighbor, device=device).to(th.long) if issquare else None
            sample_squaremask = th.tensor(sample_squaremask, device=device).to(th.long) if issquare else None
            sample_label = th.tensor(sample_label, device=device).to(th.float)
            predictions = model(
                th.tensor(sample_datainput)[:, 0].to(th.long).to(device),
                th.tensor(sample_trianglelogic).to(th.long).to(device),
                th.tensor(sample_squarelogic).to(th.long).to(device),
                th.tensor(sample_triangles).to(th.long).to(device),
                th.tensor(sample_notriangles).to(th.long).to(device),
                th.tensor(sample_squares).to(th.long).to(device),
                th.tensor(sample_nosquares).to(th.long).to(device),
                th.tensor(sample_triangleneighbor).to(th.long).to(device),
                th.tensor(sample_trianglemask).to(th.long).to(device),
                th.tensor(sample_squareneighbor).to(th.long).to(device),
                th.tensor(sample_squaremask).to(th.long).to(device),
            )
        else:
            raise NotImplementedError
        losses = criterion(predictions, sample_label)
        _auc = cal_auc(predictions.detach().cpu().numpy(), sample_label.detach().cpu().numpy(), labelnum)
        _acc = cal_acc(predictions.detach().cpu().numpy(), sample_label.detach().cpu().numpy(), labelnum)
        test_loss.append(losses.item())
        test_auc.append(_auc)
        test_acc.append(_acc)
        # if _iter % 10 == 0:
        #     testwriter.add_scalar("test_loss", losses.item(), _iter)
        #     testwriter.add_scalar("test_acc", _acc, _iter)
        #     testwriter.add_scalar("test_auc", _auc, _iter)
        # gc.collect()
        # with th.cuda.device(device):
        #     th.cuda.empty_cache()
    print(
        "===== TEST LOSS {:.4f} | TEST AUC {:.4f} | TEST ACC {:4f}".format(
            np.mean(test_loss), np.mean(test_auc), np.mean(test_acc)
        )
    )
    logfile.write(
        "TEST LOSS {:.4f} | TEST AUC {:.4f} | TEST ACC {:4f}".format(
            np.mean(test_loss), np.mean(test_auc), np.mean(test_acc)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("-c", "--cuda", type=int, help="cuda number", default=0)
    parser.add_argument("-r", "--learningrate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-w", "--weightdecay", type=float, help="weight decay", default=1e-4)
    parser.add_argument("-e", "--epoch", type=int, help="epoch", default=20)
    parser.add_argument("-d", "--dataname", type=str, help="dataset name", default="demo")
    parser.add_argument(
        "-k", "--task", type=str, help="dataset task: hg, kg, node", default="node"
    )  # decided by dataset
    parser.add_argument("-b", "--batchsize", type=int, help="batch size", default=32)
    parser.add_argument("--no-istriangle", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    # parser.add_argument("-t", "--istriangle", type=bool, help="use triangle data or not", default=True)
    # parser.add_argument("-s", "--issquare", type=bool, help="use square data or not", default=True)
    parser.add_argument("--no-issquare", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    parser.add_argument("-l", "--labelnum", type=int, help="label number", default=1)
    parser.add_argument("-n", "--modelname", type=str, help="loading or saving model name", default="01")
    parser.add_argument("-m", "--embedsize", type=int, help="embedding size", default=32)
    parser.add_argument("-i", "--indropout", type=float, help="dropout of input", default=0.3)
    parser.add_argument("-o", "--outdropout", type=float, help="dropout of output", default=0.3)
    parser.add_argument("--passnum", type=int, help="message passing number", default=1)
    parser.add_argument("--aggregatenum", type=int, help="aggregate number", default=1)
    parser.add_argument("--combinenum", type=int, help="combine number", default=3)
    parser.add_argument("--aggregatetype", type=str, help="graphlet aggregate type: mlp, mean, max, gru", default="max")
    parser.add_argument("--combinetype", type=str, help="logic combine type: mlp, mean, max", default="max")
    args = parser.parse_args()
    print("args:", args.no_istriangle)

    if args.dataname == "lastfm":
        graphfile = open(datadir + "lastfm_graph.pkl", "rb")
        trainfile = open(datadir + "lastfm_train.pkl", "rb")
        testfile = open(datadir + "lastfm_test.pkl", "rb")
        evalfile = open(datadir + "lastfm_eval.pkl", "rb")
        task = "hg"
    elif args.dataname == "demo":
        graphfile = open(datadir + "demo_graph.pkl", "rb")
        trainfile = open(datadir + "demo_" + args.task + "data.pkl", "rb")
        testfile = open(datadir + "demo_" + args.task + "data.pkl", "rb")
        evalfile = open(datadir + "demo_" + args.task + "data.pkl", "rb")
        task = args.task
    elif args.dataname == "movielen":
        graphfile = open(datadir + "movielen_graph.pkl", "rb")
        trainfile = open(datadir + "movielen_train.pkl", "rb")
        testfile = open(datadir + "movielen_test.pkl", "rb")
        evalfile = open(datadir + "movielen_eval.pkl", "rb")
        task = "hg"
    elif args.dataname == "bookmark":
        graphfile = open(datadir + "bookmark_graph.pkl", "rb")
        trainfile = open(datadir + "bookmark_train.pkl", "rb")
        testfile = open(datadir + "bookmark_test.pkl", "rb")
        evalfile = open(datadir + "bookmark_eval.pkl", "rb")
        task = "hg"
    elif args.dataname == "amazon":
        graphfile = open(datadir + "amazon_graph.pkl", "rb")
        trainfile = open(datadir + "amazon_train.pkl", "rb")
        testfile = open(datadir + "amazon_test.pkl", "rb")
        evalfile = open(datadir + "amazon_eval.pkl", "rb")
        task = "hg"
    elif args.dataname == "fb15k237":
        graphfile = open(datadir + "fb15k237_graph.pkl", "rb")
        trainfile = open(datadir + "fb15k237_train.pkl", "rb")
        testfile = open(datadir + "fb15k237_test.pkl", "rb")
        evalfile = open(datadir + "fb15k237_eval.pkl", "rb")
        task = "kg"
    elif args.dataname == "wn18":
        graphfile = open(datadir + "wn18_graph.pkl", "rb")
        trainfile = open(datadir + "wn18_train.pkl", "rb")
        testfile = open(datadir + "wn18_test.pkl", "rb")
        evalfile = open(datadir + "wn18_eval.pkl", "rb")
        task = "kg"
    else:
        raise NotImplementedError
    # graphlet_num: list [trianglelogic_num, squarelogic_num, triangle_num, triangleneighbor_num, square_num, squareneighbor_num, embed_dim]
    graphletnum = pkl.load(graphfile)
    (
        trianglelogic_num,
        squarelogic_num,
        triangle_num,
        triangleneighbor_num,
        square_num,
        squareneighbor_num,
        embed_dim,
    ) = graphletnum
    print(graphletnum)
    # graphdata: DGLGraph
    graphdata = pkl.load(graphfile)
    # graphembedding: nn.Embedding(node_num, embed_size)
    nodeembedding = pkl.load(graphfile)
    graphfile.close()
    # traindata: MyDataset
    traindata = pkl.load(trainfile)
    trainfile.close()
    # testdata: MyDataset
    testdata = pkl.load(testfile)
    testfile.close()
    # evaldata: MyDataset
    evaldata = pkl.load(evalfile)
    evalfile.close()
    device = th.device("cuda:" + str(args.cuda))
    # device = th.device("cpu")

    if nodeembedding is not None:
        node_emb0, node_emb1 = nodeembedding.shape[0], nodeembedding.shape[1]
    else:
        nodeembedding = (th.rand((graphdata.num_nodes(), embed_dim)) * 2 - 1) * math.sqrt(3)
    # nodeembedding.requires_grad_()
    # embedding_table = nn.Embedding.from_pretrained(nodeembedding).requires_grad_() # FIXME

    if task == "hg":
        angel_model = AngelHG
        graphdata_num = len(graphdata.ntypes)
    elif task == "kg":
        angel_model = AngelKG
        graphdata_num = graphdata.edata["etype"].max().item() + 1
    elif task == "node":
        angel_model = AngelNode
        graphdata_num = len(graphdata.ntypes)
    else:
        raise NotImplementedError
    print("#nodes:", graphdata.number_of_nodes())
    print("#edges:", graphdata.number_of_edges())
    print("node/edge type:", graphdata_num)

    model = angel_model(
        graphdata_num,
        args.embedsize,
        not args.no_istriangle,
        not args.no_issquare,
        args.aggregatetype,
        args.combinetype,
        args.embedsize,
        args.embedsize,
        args.embedsize,
        args.passnum,
        args.aggregatenum,
        args.combinenum,
        trianglelogic_num,
        squarelogic_num,
        triangleneighbor_num,
        squareneighbor_num,
        triangle_num,
        square_num,
        embed_dim,
        args.indropout,
        args.outdropout,
        nodeembedding,
        args.labelnum,
    ).to(device)

    def count_parameters(model):
        return sum([p.numel() for p in model.parameters() if p.requires_grad])

    print("#params:", count_parameters(model))

    criterion = nn.BCELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weightdecay)
    logname = "Angel" + task + "_" + args.dataname + "_" + args.modelname
    logname = os.path.join(logdir, logname)
    logfile = open(logname, "w+")
    training_angel(
        task,
        traindata,
        evaldata,
        logname,
        model,
        device,
        args.batchsize,
        optimizer,
        args.epoch,
        args.labelnum,
        savedir,
        args.modelname,
        patience=3,
        istriangle=not args.no_istriangle,
        issquare=not args.no_issquare,
    )
    if os.path.exists(os.path.join(logdir, logname + ".pth")):
        model.load_state_dict(th.load(os.path.join(logdir, logname + ".pth")))
    testing_angel(task, testdata, logname, model, device, args.batchsize, args.labelnum)
    logfile.close()
