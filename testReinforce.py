#!/usr/bin/python3.8

from datetime import datetime
import os
import pandas as pd
import time

import reinforce as RF
from env import EnvGraph as Env
from env import EnvGraphBalance as EnvBalance
from util import writeABC, runABC, extract_data
from abcR_Survey import reinforced_survey

import numpy as np
from tqdm import tqdm
import statistics

import sys

options = ["2_1_without_balance"]#, "2_1_with_balance", "2_3_without_balance", "2_3_without_balance", "2_7_without_balance", "2_7_with_balance", "2_9_without_balance", "2_9_with_balance", "1_0_without_balance", "1_0_with_balance"]

class Logger(object):
    def __init__(self, option):
        self.terminal = sys.stdout
        self.log = open("./logs/"+option + ".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

class AbcReturn:
    def __init__(self, returns, command):
        self.numNodes = float(returns[0])
        self.level = float(returns[1])
        self.command = command
    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numNodes < other.numNodes
        else:
            return self.level < other.level
    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numNodes) == int(self.numNodes)

def getActionSpace(opt=None):
    
    if "without_balance" in option:
        print("Without Balance Run\n\n")
        cmds = ["rewrite -l; ","rewrite -z -l; ","refactor -l; ","refactor -z -l; ","resub -K 6 -l; ","resub -K 6 -N 2 -l; ","resub -K 8 -l; ","resub -K 8 -N 2 -l; ","resub -K 10 -l; ","resub -K 10 -N 2 -l; ","resub -K 12 -l; ","resub -K 12 -N 2 -l; ","resub -k 16 -l; ","resub -k 16 -k 16 -N 2 -l; "]
    else:
        cmds = ["balance -l", "rewrite -l; ","rewrite -z -l; ","refactor -l; ","refactor -z -l; ","resub -K 6 -l; ","resub -K 6 -N 2 -l; ","resub -K 8 -l; ","rs -K 8 -N 2 -l; ","rs -K 10 -l; ","rs -K 10 -N 2 -l; ","rs -K 12 -l; ","rs -K 12 -N 2 -l; ", "resub -k 16 -l; ", "resub -k 16 -N 2 -l; "]
    
    if opt is None:
        for idx, cmd in enumerate(cmds):
            print(str(idx+1)+". "+cmd)
        ids = input("\nSelect the commands that you wish to use : \n(Enter the index numbers in a comma separated manner - eg 1,4,5)\n")
        ids = ids.split(",")
        ids = [int(x)-1 for x in ids]
    else:
        ids = np.arange(len(cmds))
    return ids

benchmarks = []

def testReinforce(filename, ben, option, opt=None):
    now = datetime.now()
    dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time ", dateTime)
    print("###################################\n")
    coefs = [float(option[0]), float(option[2])]
    print(coefs)
    cmds = getActionSpace(opt=opt)
    if "without_balance" in option:
        env = EnvBalance(filename, cmds, coefs)
    else:
        env = Env(filename, cmds, coefs)
    
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 9e-4, RF.FcModelGraph, path="./models/"+option)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel, path="./models/"+option)
    reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline)

    lastTen = []

    for idx in tqdm(range(200), total = 200, ncols = 100, desc ="Episode : "):
        returns, command = reinforce.episode(phaseTrain=True)
        seqLen = reinforce.lenSeq
        line = "Episode : " + str(idx) + " Seq Length : " + str(seqLen)
        if idx >= 180:
            lastTen.append(AbcReturn(returns, command))
        if idx % 10 == 0:
            print(line)
        # reinforce.replay()
    benchmarks.append(ben)
    reinforce._pi.save(benchmarks)
    reinforce._baseline.save(benchmarks)

    resultName = "./results/" + ben + "_"  + option + ".csv"
    #lastfive.sort(key=lambda x : x.level)
    lastTen = sorted(lastTen)
    with open(resultName, 'a') as andLog:
        line = ""
        line += str(lastTen[0].numNodes)
        line += " "
        line += str(lastTen[0].level)
        line += "\n"
        line += lastTen[0].command + "\n" + str(len(lastTen[0].command.split(";"))-1) + "\n"
        andLog.write(line)
    rewards = reinforce.sumRewards
    
    with open('./results/sum_rewards_'+ option +'.csv', 'a') as rewardLog:
        line = ben+","
        for idx in range(len(rewards)):
            line += str(rewards[idx]) 
            if idx != len(rewards) - 1:
                line += ","
        line += "\n"
        rewardLog.write(line)
    with open ('./results/converge_'+ option +'.csv', 'a') as convergeLog:
        line = ben+","
        returns, command = reinforce.episode(phaseTrain=False)
        line += str(returns[0])
        line += ","
        line += str(returns[1])
        line += "\n"
        line += command + "\n" + str(len(command.split(";"))-1) + "\n"
        convergeLog.write(line)
    
    return lastTen[0].command

if __name__ == "__main__":
    
    dir = "./bench/"
    for option in options:
        sys.stdout = Logger(option)
        start_c = time.time()
        for subdir, dirs, files in os.walk(dir,topdown=True):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".aig"):
                    start = time.time()
                    print("Running Reinforce on ",file,".....",sep='')
                    command = testReinforce(filepath, file[:-4], option, opt="All")
                    print("\n",command,"\n")
                    end = time.time()
                    print("Time Elapsed for ", file, " : ", end-start, "seconds\n")
        end_c = time.time()
        print("Total time taken for option "+option+" : ", end_c - start_c)
        # Collect results over ABC and Custim Optimizations
        reinforced_survey("_"+option)

