#!/usr/bin/python3.8

from functools import partial
from datetime import datetime
import os
import pandas as pd
import time
import multiprocessing as  mp

import mtlPy
import reinforce as RF
from env import EnvGraph as Env
from env import EnvGraphBalance as EnvBalance
from env import EnvGraphDch as EnvDch
from env import EnvGraphMtlDch as EnvMtlDch
from env import EnvReplica as EnvRep
from env import EnvGraphDchMap as EnvDchMap
from env import EnvReplicaExact as EnvRepEx
from abcR_Survey import reinforced_survey


import numpy as np
from tqdm import tqdm

import sys

options = ["replica_exact"]#, "with_balance", "without_balance"]
coefs = ["2_1", "2_3", "2_7", "2_9", "1_1", "1_0"]

class Logger(object):
    def __init__(self, option):
        self.flag = True
        self.terminal = sys.stdout
        if not os.path.exists("./logs/" + option[4:]):
            os.system("mkdir ./logs/" + option[4:])
        self.log = open("./logs/" + option[4:] + "/" + option + ".log", "a")

    def write(self, message):
        self.terminal.write(message)
        if self.flag:
            self.log.write(message)  
        else:
            pass

    def flush(self):
        pass

    def close(self):
        self.flag = False
        self.log.close()

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

def getActionSpace(option, opt=None):
    
    if "without_balance" in option:
        print("Without Balance Run\n\n")
        cmds = ["rewrite -l; ","rewrite -z -l; ","refactor -l; ","refactor -z -l; ","resub -K 6 -l; ","resub -K 6 -N 2 -l; ","resub -K 8 -l; ","resub -K 8 -N 2 -l; ","resub -K 10 -l; ","resub -K 10 -N 2 -l; ","resub -K 12 -l; ","resub -K 12 -N 2 -l; ","resub -k 16 -l; ","resub -k 16 -k 16 -N 2 -l; "]        
    elif "dch" in option:
        print("DCH Run\n\n")
        cmds = ["balance -l", "rewrite -l; ","rewrite -z -l; ","refactor -l; ","refactor -z -l; ","resub -K 8 -l; ","resub -K 8 -N 2 -l; ","resub -K 10 -l; ","resub -K 10 -N 2 -l; ","resub -K 12 -l; ","resub -K 12 -N 2 -l; ", "resub -k 16 -l; ", "resub -k 16 -N 2 -l; ", "dch; ", "dc2; "]
    elif "mtl" in option:
        print("MockTurtle Run\n\n")
        cmds = ["rewrite; ", "rewrite azg; ", "rewrite udc; ", "rewrite azg udc; ", "balance; ", "balance crit; "]#, "resub; ", "resub udc; ", "resub pd; ", "resub udc pd; "]
    elif "replica" in option:
        cmds = ["balance; ", "rewrite; ","rewrite -z; ","refactor; ","refactor -z; "]
    else:    
        print("With Balance Run\n\n")
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

def testReinforce(filename, option, opt=None):

    ben = filename.split("/")[-1][:-4]
    start = time.time()
    print("Running Reinforce on ",ben,".....",sep='')

    now = datetime.now()
    dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time ", dateTime)
    print("###################################\n")
    coefs = [float(option[0]), float(option[2])]
    print(coefs)
    cmds = getActionSpace(option, opt=opt)
    if "without_balance" in option:
        env = EnvBalance(filename, cmds, coefs)
    elif "dch_map" in option:
        env = EnvDchMap(filename, cmds, coefs)
    elif "dch" in option:
        env = EnvDch(filename, cmds, coefs)
    elif "mtl" in option:
        env = EnvMtlDch(filename, cmds, coefs)
    elif "replica_exact" in option:
        env = EnvRepEx(filename, cmds, coefs)
    elif "replica" in option:
        env = EnvRep(filename, cmds, coefs)
    else:
        env = Env(filename, cmds, coefs)
    
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 9e-4, RF.FcModelGraph, option, path=None)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel, option, path=None)
    reinforce = RF.Reinforce(env, 0.95, vApprox, vbaseline)

    if not os.path.exists("./results/" + option[4:]):
        os.system("mkdir ./results/" + option[4:])

    lastTen = []
    resultName = "./results/" + option[4:] + "/" + ben + "_"  + option + ".csv"
    andLog =  open(resultName, 'a')

    for idx in tqdm(range(200), total = 200, ncols = 100, desc ="Episode : "):
        returns, command = reinforce.episode(phaseTrain=True)
        seqLen = reinforce.lenSeq
        line = "Episode : " + str(idx) + " Seq Length : " + str(seqLen)
        if idx >= 180:
            lastTen.append(AbcReturn(returns, command))
        if idx % 10 == 0:
            print(line)
        line = ""
        line += str(float(returns[0]))
        line += " "
        line += str(float(returns[1]))
        line += "\n"
        line += command + "\n" + str(len(command.split(";"))-1) + "\n"
        andLog.write(line)

    benchmarks.append(ben)
    reinforce._pi.save(benchmarks)
    reinforce._baseline.save(benchmarks)


    lastTen = sorted(lastTen)
    line = ""
    line += str(lastTen[0].numNodes)
    line += " "
    line += str(lastTen[0].level)
    line += "\n"
    line += lastTen[0].command + "\n" + str(len(lastTen[0].command.split(";"))-1) + "\n"
    andLog.write(line)
    andLog.close()
    
    rewards = reinforce.sumRewards
    
    with open("./results/" + option[4:] + "/sum_rewards_" + option +'.csv', 'a') as rewardLog:
        line = ben+","
        for idx in range(len(rewards)):
            line += str(rewards[idx]) 
            if idx != len(rewards) - 1:
                line += ","
        line += "\n"
        rewardLog.write(line)
    
    with open ("./results/" + option[4:] + "/converge_" + option +'.csv', 'a') as convergeLog:
        line = ben+","
        returns, command = reinforce.episode(phaseTrain=False)
        line += str(returns[0])
        line += ","
        line += str(returns[1])
        line += "\n"
        line += command + "\n" + str(len(command.split(";"))-1) + "\n"
        convergeLog.write(line)
    
    print("\n",lastTen[0].command,"\n")
    end = time.time()
    print("Time Elapsed for ", ben, " : ", end-start, "seconds\n")

    return lastTen[0].command

if __name__ == "__main__":
    
    dir = "./bench/Replica"
    for opt in options:
        for coef in coefs:
            option = coef + "_" + opt
            sys.stdout = Logger(option)
            start_c = time.time()
            for subdir, dirs, files in os.walk(dir,topdown=True):
                for file in files:
                    filepath = subdir + os.sep + file
                    if filepath.endswith(".aig"):
                        start = time.time()
                        command = testReinforce(filepath, option, opt="All")
                        end = time.time()
            end_c = time.time()
            print("Total time taken for option "+option+" : ", end_c - start_c)
            # Collect results over ABC and Custim Optimizations
            # reinforced_survey(opt, coef)
            sys.stdout.close()

    # dir = "./bench/"
    # filepaths = []
    # for subdir, dirs, files in os.walk(dir,topdown=True):
    #     for file in files:
    #         filepath = subdir + os.sep + file
    #         if filepath.endswith(".aig"):
    #             filepaths.append(filepath)

    # pool = mp.Pool(processes=4)
    
    # for opt in options:
    #     for coef in coefs:
    #         option = coef + "_" + opt
    #         sys.stdout = Logger(option)
    #         tempreinforce = partial(testReinforce, opt="All")
    #         treinforce = partial(tempreinforce, option=option)
    #         start_c = time.time()
    #         commands = pool.map(treinforce, filepaths)
    #         print(commands)
    #         end_c = time.time()
    #         print("Total time taken for option "+option+" : ", end_c - start_c)
    #         # Collect results over ABC and Custim Optimizations
    #         # reinforced_survey(opt, coef)
    #         sys.stdout.close()
