#!/usr/bin/python3.8

from datetime import datetime
import os
import pandas as pd
import time

import reinforce_inference as RF
from env_inference import EnvGraph as Env
from env_inference import EnvGraphBalance as EnvBalance
from util_inference import writeABC, runABC, extract_data

import numpy as np
from tqdm import tqdm
import statistics

import sys

options = ["2_9_without_balance", "2_9_with_balance", "2_7_without_balance", "2_7_with_balance", "2_5_without_balance", "2_5_with_balance", "1_0_without_balance", "1_0_with_balance", "0_1_without_balance", "0_1_with_balance"]

class Logger(object):
    def __init__(self, option):
        self.terminal = sys.stdout
        self.log = open("./inference_logs/"+option + ".log", "a")

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
        cmds = ["rewrite -l; ","rewrite -z -l; ","refactor -l; ","refactor -z -l; ","resub -K 6 -l; ","resub -K 6 -N 2 -l; ","resub -K 8 -l; ","resub -K 8 -N 2 -l; ","resub -K 10 -l; ","resub -K 10 -N 2 -l; ","resub -K 12 -l; ","resub -K 12 -N 2 -l; ","resub -k 16 -l; ","resub -k 16 -N 2 -l; "]
    else:
        cmds = ["balance -l", "rewrite -l; ","rewrite -z -l; ","refactor -l; ","refactor -z -l; ","resub -K 6 -l; ","resub -K 6 -N 2 -l; ","resub -K 8 -l; ","resub -K 8 -N 2 -l; ","resub -K 10 -l; ","resub -K 10 -N 2 -l; ","resub -K 12 -l; ","resub -K 12 -N 2 -l; ", "resub -k 16 -l; ", "resub -k 16 -N 2 -l; "]
    
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
    coefnum = float(option[0])
    coeflev = float(option[2])
    print(coefnum, coeflev)
    cmds = getActionSpace(opt=opt)
    if "without_balance" in option:
        env = EnvBalance(filename, cmds)
    else:
        env = Env(filename, cmds, [coefnum, coeflev])
    
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 9e-4, RF.FcModelGraph, path="../models/"+option)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel, path="../models/"+option)
    reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline)

    lastTen = []

    for idx in tqdm(range(3), total = 3, ncols = 100, desc ="Episode : "):
        returns, command = reinforce.episode(phaseTrain=False)
        seqLen = reinforce.lenSeq
        line = "Seq Length : " + str(seqLen) + "\nCommand : " + command + "\n"
        lastTen.append(AbcReturn(returns, command))
        print(line)
        # reinforce.replay()

    resultName = "./inference_results/" + ben + "_"  + option + ".csv"
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
    
    return lastTen[0].command

if __name__ == "__main__":
    
    dir = "./inference_bench/"
    for option in options:
        sys.stdout = Logger(option) 
        start_f = time.time()
        for subdir, dirs, files in os.walk(dir,topdown=True):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".aig"):
                    start = time.time()
                    print("Running Reinforce on ",file,".....",sep='')
                    command = testReinforce(filepath, file[:-4], option, opt="All")
                    end = time.time()
                    print("Time Elapsed for ", file, " : ", end-start, "seconds\nCommand :",command,"\n\n")
        end_f = time.time()
        print("\nTotal time elapsed in" + option + " :",end_f - start_f,"seconds\n")
