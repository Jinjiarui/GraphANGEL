from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch as th
import torch.nn as nn
import os
import sys
import pickle as pkl
import argparse
import numpy as np
import dgl
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basedir, ".."))
from utils.data_loader import DataLoader
from tensorboardX import SummaryWriter
from algo.angel_node import Angel
from utils.base import cal_acc, cal_auc

logdir = os.path.join(basedir, "../log/")
datadir = os.path.join(basedir, "../data/")
savedir = os.path.join(logdir, "model/")

def training_angel(traindata, evaldata, logname, model, device, batchsize, optimizer, epoch, labelnum, savedir, modelname, patience):
    traindir = logname + "_train.txt"
    evaldir = logname + "_eval.txt"
    if not os.path.exists(traindir):
        os.mkdir(traindir)
    if not os.path.exists(evaldir):
        os.mkdir(evaldir)
    trainwriter = SummaryWriter(log_dir=traindir)
    evalwriter = SummaryWriter(log_dir=evaldir)
    train_loss, train_acc, train_auc = [], [], []
    eval_loss, eval_acc, eval_auc = [], [], []
    for _epoch in range(epoch):
        print("===== EPOCH {} =====".format(_epoch))
        logfile.write("Epoch {} >>".format(_epoch))
        for _iter in range(int(len(traindata)/batchsize)):
            model.train()
            batch_index = list(np.random.choice(batchsize, len(traindata)-1))
            optimizer.zero_grad()
            sample_sourceid, sample_targetid, sample_label, sample_triangleid1, sample_triangleid2, sample_trianglemask1, sample_trianglemask2, sample_squareid1, sample_squareid2, sample_squareid3, sample_squareid4, sample_squaremask1, sample_squaremask2, sample_squaremask3, sample_squaremask4, sample_trianglelistid1, sample_trianglelistid2, sample_notrianglelistid1, sample_notrianglelistid2, sample_squarelistid1, sample_squarelistid2, sample_squarelistid3, sample_squarelistid4, sample_nosquarelistid1, sample_nosquarelistid2, sample_nosquarelistid3, sample_nosquarelistid4 = traindata.sample(batch_index)
            predictions = model(th.Tensor(sample_sourceid).to(th.long).to(device), th.Tensor(sample_targetid).to(th.long).to(device), th.Tensor(sample_trianglelistid1).to(th.long).to(device), th.Tensor(sample_trianglelistid2).to(th.long).to(device), th.Tensor(sample_notrianglelistid1).to(th.long).to(device), th.Tensor(sample_notrianglelistid2).to(th.long).to(device), th.Tensor(sample_squarelistid1).to(th.long).to(device), th.Tensor(sample_squarelistid2).to(th.long).to(device), th.Tensor(sample_squarelistid3).to(th.long).to(device), th.Tensor(sample_squarelistid4).to(th.long).to(device), th.Tensor(sample_nosquarelistid1).to(th.long).to(device), th.Tensor(sample_nosquarelistid2).to(th.long).to(device), th.Tensor(sample_nosquarelistid3).to(th.long).to(device), th.Tensor(sample_nosquarelistid4).to(th.long).to(device), th.Tensor(sample_triangleid1).to(th.long).to(device), th.Tensor(sample_triangleid2).to(th.long).to(device), th.Tensor(sample_trianglemask1).to(th.long).to(device), th.Tensor(sample_trianglemask2).to(th.long).to(device), th.Tensor(sample_squareid1).to(th.long).to(device), th.Tensor(sample_squareid2).to(th.long).to(device), th.Tensor(sample_squareid3).to(th.long).to(device), th.Tensor(sample_squareid4).to(th.long).to(device), th.Tensor(sample_squaremask1).to(th.long).to(device), th.Tensor(sample_squaremask2).to(th.long).to(device), th.Tensor(sample_squaremask3).to(th.long).to(device), th.Tensor(sample_squaremask4).to(th.long).to(device))
            # label: batch_size, label_num
            losses = criterion(predictions, th.Tensor(sample_label).to(th.float))
            losses.backward()
            optimizer.step()
            _auc = cal_auc(predictions.detach().cpu().numpy(), np.array(sample_label), labelnum)
            _acc = cal_acc(predictions.detach().cpu().numpy(), np.array(sample_label), labelnum)
            train_loss.append(losses.item())
            train_auc.append(_auc)
            train_acc.append(_acc)
            if _iter % 10 == 0:
                trainwriter.add_scalar("train_loss", losses.item(), _iter)
                trainwriter.add_scalar("train_acc", _acc, _iter)
                trainwriter.add_scalar("train_auc", _auc, _iter)
                if _iter % 100 == 0:
                    print("===== EPOCH {:d} | LOSS {:.4f} | AUC {:.4f} | ACC {:4f}".format(_epoch, losses.item(), _auc, _acc))
                    logfile.write("Epoch {:d} | Loss {:.4f} | AUC {:.4f} | ACC {:4f}".format(_epoch, losses.item(), _auc, _acc))
        # with th.cuda.device(device):
        #     th.cuda.empty_cache()
        print("===== TRAIN LOSS {:.4f} | TRAIN AUC {:.4f} | TRAIN ACC {:4f}".format(np.mean(train_loss), np.mean(train_auc), np.mean(train_acc)))
        logfile.write("TRAIN Loss {:.4f} | TRAIN AUC {:.4f} | TRAIN ACC {:4f}".format(np.mean(train_loss), np.mean(train_auc), np.mean(train_acc)))
        # eval
        model.eval()
        with th.no_grad():
            for _iter in range(int(len(evaldata)/batchsize)):
                _lastloss = 0.0
                _count = 0
                sample_sourceid, sample_targetid, sample_label, sample_triangleid1, sample_triangleid2, sample_trianglemask1, sample_trianglemask2, sample_squareid1, sample_squareid2, sample_squareid3, sample_squareid4, sample_squaremask1, sample_squaremask2, sample_squaremask3, sample_squaremask4, sample_trianglelistid1, sample_trianglelistid2, sample_notrianglelistid1, sample_notrianglelistid2, sample_squarelistid1, sample_squarelistid2, sample_squarelistid3, sample_squarelistid4, sample_nosquarelistid1, sample_nosquarelistid2, sample_nosquarelistid3, sample_nosquarelistid4 = evaldata.sample(batch_index)
                predictions = model(th.Tensor(sample_sourceid).to(th.long).to(device), th.Tensor(sample_targetid).to(th.long).to(device), th.Tensor(sample_trianglelistid1).to(th.long).to(device), th.Tensor(sample_trianglelistid2).to(th.long).to(device), th.Tensor(sample_notrianglelistid1).to(th.long).to(device), th.Tensor(sample_notrianglelistid2).to(th.long).to(device), th.Tensor(sample_squarelistid1).to(th.long).to(device), th.Tensor(sample_squarelistid2).to(th.long).to(device), th.Tensor(sample_squarelistid3).to(th.long).to(device), th.Tensor(sample_squarelistid4).to(th.long).to(device), th.Tensor(sample_nosquarelistid1).to(th.long).to(device), th.Tensor(sample_nosquarelistid2).to(th.long).to(device), th.Tensor(sample_nosquarelistid3).to(th.long).to(device), th.Tensor(sample_nosquarelistid4).to(th.long).to(device), th.Tensor(sample_triangleid1).to(th.long).to(device), th.Tensor(sample_triangleid2).to(th.long).to(device), th.Tensor(sample_trianglemask1).to(th.long).to(device), th.Tensor(sample_trianglemask2).to(th.long).to(device), th.Tensor(sample_squareid1).to(th.long).to(device), th.Tensor(sample_squareid2).to(th.long).to(device), th.Tensor(sample_squareid3).to(th.long).to(device), th.Tensor(sample_squareid4).to(th.long).to(device), th.Tensor(sample_squaremask1).to(th.long).to(device), th.Tensor(sample_squaremask2).to(th.long).to(device), th.Tensor(sample_squaremask3).to(th.long).to(device), th.Tensor(sample_squaremask4).to(th.long).to(device))
                losses = criterion(predictions, th.Tensor(sample_label).to(th.float))
                _auc = cal_auc(predictions.detach().cpu().numpy(), np.array(sample_label), labelnum)
                _acc = cal_acc(predictions.detach().cpu().numpy(), np.array(sample_label), labelnum)
                eval_loss.append(losses.item())
                eval_auc.append(_auc)
                eval_acc.append(_acc)
                # gc.collect()
                # with th.cuda.device(device):
                #     th.cuda.empty_cache()
                if _iter % 10 == 0:
                    evalwriter.add_scalar("eval_loss", losses.item(), _iter)
                    evalwriter.add_scalar("eval_acc", _acc, _iter)
                    evalwriter.add_scalar("eval_auc", _auc, _iter)
                if np.mean(eval_acc) >= np.max(train_acc):
                    logfile.write("===== SAVING BEST ======")
                    if not os.path.exists(savedir):
                        os.makedirs(savedir, exist_ok=True)
                    th.save(model.state_dict(), os.path.join(savedir, modelname+".pth"))
                if np.mean(eval_loss) >= _lastloss:
                    _count += 1
                else:
                    _count = 0
                _lastloss = np.max(eval_loss)
                if _count >= patience:
                    print("===== EARLY STOP =====")
                    break

def testing_angel(testdata, logname, model, device, batchsize, labelnum):
    testdir = logname + "_test.txt"
    if not os.path.exists(testdir):
        os.mkdir(testdir)
    testwriter = SummaryWriter(log_dir=testdir)
    test_loss, test_auc, test_acc = [], [], []
    with th.no_grad():
        for _iter in range(int(len(testdata)/batchsize)):
            batch_index = list(np.random.choice(batchsize, len(testdata)-1))
            sample_sourceid, sample_targetid, sample_label, sample_triangleid1, sample_triangleid2, sample_trianglemask1, sample_trianglemask2, sample_squareid1, sample_squareid2, sample_squareid3, sample_squareid4, sample_squaremask1, sample_squaremask2, sample_squaremask3, sample_squaremask4, sample_trianglelistid1, sample_trianglelistid2, sample_notrianglelistid1, sample_notrianglelistid2, sample_squarelistid1, sample_squarelistid2, sample_squarelistid3, sample_squarelistid4, sample_nosquarelistid1, sample_nosquarelistid2, sample_nosquarelistid3, sample_nosquarelistid4 = testdata.sample(batch_index)
            predictions = model(th.Tensor(sample_sourceid).to(th.long).to(device), th.Tensor(sample_targetid).to(th.long).to(device), th.Tensor(sample_trianglelistid1).to(th.long).to(device), th.Tensor(sample_trianglelistid2).to(th.long).to(device), th.Tensor(sample_notrianglelistid1).to(th.long).to(device), th.Tensor(sample_notrianglelistid2).to(th.long).to(device), th.Tensor(sample_squarelistid1).to(th.long).to(device), th.Tensor(sample_squarelistid2).to(th.long).to(device), th.Tensor(sample_squarelistid3).to(th.long).to(device), th.Tensor(sample_squarelistid4).to(th.long).to(device), th.Tensor(sample_nosquarelistid1).to(th.long).to(device), th.Tensor(sample_nosquarelistid2).to(th.long).to(device), th.Tensor(sample_nosquarelistid3).to(th.long).to(device), th.Tensor(sample_nosquarelistid4).to(th.long).to(device), th.Tensor(sample_triangleid1).to(th.long).to(device), th.Tensor(sample_triangleid2).to(th.long).to(device), th.Tensor(sample_trianglemask1).to(th.long).to(device), th.Tensor(sample_trianglemask2).to(th.long).to(device), th.Tensor(sample_squareid1).to(th.long).to(device), th.Tensor(sample_squareid2).to(th.long).to(device), th.Tensor(sample_squareid3).to(th.long).to(device), th.Tensor(sample_squareid4).to(th.long).to(device), th.Tensor(sample_squaremask1).to(th.long).to(device), th.Tensor(sample_squaremask2).to(th.long).to(device), th.Tensor(sample_squaremask3).to(th.long).to(device), th.Tensor(sample_squaremask4).to(th.long).to(device))
            losses = criterion(predictions, th.Tensor(sample_label).to(th.float))
            _auc = cal_auc(predictions.detach().cpu().numpy(), np.array(sample_label), labelnum)
            _acc = cal_acc(predictions.detach().cpu().numpy(), np.array(sample_label), labelnum)
            test_loss.append(losses.item())
            test_auc.append(_auc)
            test_acc.append(_acc)
            if _iter % 10 == 0:
                testwriter.add_scalar("test_loss", losses.item(), _iter)
                testwriter.add_scalar("test_acc", _acc, _iter)
                testwriter.add_scalar("test_auc", _auc, _iter)
            # gc.collect()
            # with th.cuda.device(device):
            #     th.cuda.empty_cache()
    print("===== TEST LOSS {:.4f} | TEST AUC {:.4f} | TEST ACC {:4f}".format(np.mean(test_loss), np.mean(test_auc), np.mean(test_acc)))
    logfile.write("TEST LOSS {:.4f} | TEST AUC {:.4f} | TEST ACC {:4f}".format(np.mean(test_loss), np.mean(test_auc), np.mean(test_acc)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("-c", "--cuda", type=int, help="cuda number", default=0)
    parser.add_argument("-r", "--learningrate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-w", "--weightdecay", type=float, help="weight decay", default=0.1)
    parser.add_argument("-e", "--epoch", type=int, help="epoch", default=20)
    parser.add_argument("-d", "--dataname", type=str, help="dataset name", default="demo")
    parser.add_argument("-b", "--batchsize", type=int, help="batch size", default=2)
    parser.add_argument("-t", "--istriangle", type=bool, help="use triangle data or not", default=True)
    parser.add_argument("-s", "--issquare", type=bool, help="use square data or not", default=True)
    parser.add_argument("-p", "--passnum", type=int, help="message passing number", default=2)
    parser.add_argument("-l", "--labelnum", type=int, help="label number", default=1)
    parser.add_argument("-n", "--modelname", type=str, help="loading or saving model name", default="01")
    parser.add_argument("-m", "--embedsize", type=int, help="embedding size", default=32)
    parser.add_argument("--passtype", type=str, help="message passing type: mean, project, concat", default="mean")
    parser.add_argument("--aggregatetype", type=str, help="message aggregating type: mean, project, gru, concat", default="concat")
    parser.add_argument("--readouttype", type=str, help="graphlet readout type: softmax, product, concat", default="concat")
    parser.add_argument("--combinetype", type=str, help="graphlet combine type: concat, product", default="concat")
    parser.add_argument("--updatetype", type=str, help="node embedding update type: concat, product", default="concat")
    args = parser.parse_args() 

    if args.dataname == "lasfm":
        graphfile = open(datadir+"lastfm_graph.pkl", "rb")
        trainfile = open(datadir+"lastfm_train.pkl", "rb")
        testfile = open(datadir+"lastfm_test.pkl", "rb")
        evalfile = open(datadir+"lastfm_eval.pkl", "rb")
    elif args.dataname == "demo":
        graphfile = open(datadir+"demo_graph.pkl", "rb")
        trainfile = open(datadir+"demo_data.pkl", "rb")
        testfile = open(datadir+"demo_data.pkl", "rb")
        evalfile = open(datadir+"demo_data.pkl", "rb")
    elif args.dataname == "movielen":
        graphfile = open(datadir+"movielen_graph.pkl", "rb")
        trainfile = open(datadir+"movielen_train.pkl", "rb")
        testfile = open(datadir+"movielen_test.pkl", "rb")
        evalfile = open(datadir+"movielen_eval.pkl", "rb")
    elif args.dataname == "bookmark":
        graphfile = open(datadir+"bookmark_graph.pkl", "rb")
        trainfile = open(datadir+"bookmark_train.pkl", "rb")
        testfile = open(datadir+"bookmark_test.pkl", "rb")
        evalfile = open(datadir+"bookmark_eval.pkl", "rb")
    else:
        raise NotImplementedError
    # graphlet_num: list [triangle_num, triangleneighbor_num, square_num, squareneighbor_num]
    graphletnum = pkl.load(graphfile)
    # graphdata: DGLGraph
    graphdata = pkl.load(graphfile)
    # graphembedding: nn.Embedding(node_num, embed_size)
    graphletembedding = pkl.load(graphfile)
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
    # device = th.device("cuda:" + str(args.cuda))
    device = th.device("cpu")
    model = Angel(graphdata.num_nodes(), args.embedsize, args.istriangle, args.issquare, args.passtype, args.readouttype, args.updatetype, args.aggregatetype, args.combinetype, args.embedsize, args.embedsize, args.embedsize, args.passnum, graphletnum[1], graphletnum[3], graphletnum[0], graphletnum[2], graphletembedding, args.labelnum)
    if os.path.exists(savedir) and args.modelname is not None:
        model.load_state_dict(th.load(os.path.join(savedir, args.modelname+".pth")))
        print("===== MODEL LOADING =====")
    criterion = nn.BCELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weightdecay)

    logname = "AngelEdge" + "_" + args.dataname + "_" + args.modelname 
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logfile = open(os.path.join(logdir, logname+".txt"), "w+")
    training_angel(traindata, evaldata, logname, model, device, args.batchsize, optimizer, args.epoch, args.labelnum, savedir, args.modelname, patience=3)
    if os.path.exists(os.path.join(logdir, logname+".pth")):
        model.load_state_dict(th.load(os.path.join(logdir, logname+".pth")))
    testing_angel(testdata, logname, model, device, args.batchsize, args.labelnum)
    logfile.close()