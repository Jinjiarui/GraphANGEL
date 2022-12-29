import torch as th
import torch.nn as nn
import os
import sys
import gc
import argparse
import pickle as pkl
import random
import numpy as np
from time import time
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basedir, "../"))
from utils.data_loader import DataLoader

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(basedir, "../data")

def running_dataloader(dataname, trianglenum, squarenum, trianglelogicnum, squarelogicnum, triangleneighbornum, squareneighbornum, embedsize):
    data_loader = DataLoader(dataname, datadir, trianglenum, squarenum, trianglelogicnum, squarelogicnum, triangleneighbornum, squareneighbornum, embedsize)
    _ = data_loader.load_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataname", type=str, help="lastfm, amazon, yelp, douban_book, fb15k237, wn18, AIFB, AM", default="demo")
    parser.add_argument("-t", "--trianglenum", type=int, help="graphlet num of triangle shape", default=8)
    parser.add_argument("-s", "--squarenum", type=int, help="graphlet num of square shape", default=8)
    parser.add_argument("-n", "--triangleneighbornum", type=int, help="neighborhood num of triangle shape", default=2)
    parser.add_argument("-m", "--squareneighbornum", type=int, help="neighborhood num of square shape", default=2)
    parser.add_argument("-a", "--trianglelogicnum", type=int, help="logic number of triangle shape", default=2)
    parser.add_argument("-b", "--squarelogicnum", type=int, help="logic number of square shape", default=2)
    parser.add_argument("-e", "--embedsize", type=int, help="embedding size", default=32)
    args = parser.parse_args()
    running_dataloader(args.dataname, args.trianglenum, args.squarenum, args.trianglelogicnum, args.squarelogicnum, args.triangleneighbornum, args.squareneighbornum, args.embedsize)
