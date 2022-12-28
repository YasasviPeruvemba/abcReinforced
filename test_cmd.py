#!/usr/bin/python3.8

from functools import partial
from datetime import datetime
import os
import pandas as pd
import time
import multiprocessing as  mp

import mtlPy
import abcPy
# import reinforce as RF
# from env import EnvGraph as Env
# from env import EnvGraphBalance as EnvBalance
# from env import EnvGraphDch as EnvDch
# from env import EnvGraphMtlDch as EnvMtlDch
# from env import EnvReplica as EnvRep
# from env import EnvGraphDchMap as EnvDchMap
# from env import EnvReplicaExact as EnvRepEx
# from abcR_Survey import reinforced_survey


import numpy as np
from tqdm import tqdm

import sys

def tech_map():
    _abc.read_lib()
    _abc.tech_map()
    _abc.cec()
    return _abc.print_gates()

# def statValue( stat):
#     and_coef = 2
#     level_coef =1
#     # return (float(stat)/float(initStats))
#     return (and_coef*(float(stat.numAnd)/float(initNumAnd)) + level_coef*(float(stat.lev)/float(initLev)))/(and_coef + .level_coef)
    
and_coef = 2
level_coef =1
_abc = abcPy.AbcInterface()
_abc.start()
_abc.backup()
timeSeq = 0

# tech_map()
# initStats = _abc.print_gates()
# print(initStats)
_abc.read('./bench/EPFL/Small/bar.aig')
_abc.balance(l=True)
initStats = _abc.aigStats()
start = tech_map()
_abc.recall()
_abc.compress2rs_balance()
curStats = _abc.aigStats()
boom = tech_map()
_abc.read('./bench/EPFL/Small/bar.aig')
_abc.balance(l=True)
_abc.refactor(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.refactor(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.refactor(l=True)
_abc.resub(k=10, l=True)
_abc.refactor(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.refactor(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.rewrite(l=True, z=True)
_abc.resub(k=6, n=2, l=True)
_abc.rewrite(l=True, z=True)
_abc.refactor(l=True, z=True)
_abc.rewrite(l=True, z=True)
# _abc.aigStats()
cry = tech_map()
print("START", start)
print("INIT:",boom)
print("CUR:",cry)
