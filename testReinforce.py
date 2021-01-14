#!/usr/bin/python3.8

##
# @file testReinforce.py
# @author Keren Zhu
# @date 10/31/2019
# @brief The main for test REINFORCE
#

from datetime import datetime
import os
import pandas as pd
import time

import reinforce as RF
from env import EnvGraph as Env
from util import writeABC, runABC, extract_data

import numpy as np
from tqdm import tqdm
import statistics

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

def testReinforce(filename, ben):
    now = datetime.now()
    dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time ", dateTime)
    env = Env(filename)
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 9e-4, RF.FcModelGraph)
    baseline = RF.Baseline(0)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel)
    reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline)

    lastTen = []

    for idx in tqdm(range(200), total = 200, ncols = 100, desc ="Episode : "):
        returns, command = reinforce.episode(phaseTrain=True)
        seqLen = reinforce.lenSeq
        line = "iter " + str(idx) + " returns "+ str(returns) + " seq Length " + str(seqLen) + "\n"
        if idx >= 190:
            lastTen.append(AbcReturn(returns, command))
        # print(line)
        #reinforce.replay()

    resultName = "./results/" + ben + ".csv"
    #lastfive.sort(key=lambda x : x.level)
    lastTen = sorted(lastTen)
    with open(resultName, 'a') as andLog:
        line = ""
        line += str(lastTen[0].numNodes)
        line += " "
        line += str(lastTen[0].level)
        line += "\n"
        line += lastTen[0].command + "\n" + str(len(lastTen[0].command.split(";"))) + "\n"
        andLog.write(line)
    rewards = reinforce.sumRewards
    
    with open('./results/sum_rewards.csv', 'a') as rewardLog:
        line = ben+","
        for idx in range(len(rewards)):
            line += str(rewards[idx]) 
            if idx != len(rewards) - 1:
                line += ","
        line += "\n"
        rewardLog.write(line)
    with open ('./results/converge.csv', 'a') as convergeLog:
        line = ben+","
        returns, command = reinforce.episode(phaseTrain=False)
        line += str(returns[0])
        line += ","
        line += str(returns[1])
        line += "\n"
        line += command + "\n"
        convergeLog.write(line)
    
    return lastTen[0].command

def visualize(df_area, df_delay):
    hA = df_area.to_pickle()
    fA = open("Reinforced_Survey_Area.html", "w")
    fA.write(hA)
    fA.close()

    hA = df_delay.to_pickle()
    fA = open("Reinforced_Survey_Delay.html", "w")
    fA.write(hA)
    fA.close()

if __name__ == "__main__":
    """
    env = Env("./bench/i10.aig")
    vbaseline = RF.BaselineVApprox(4, 3e-3, RF.FcModel)
    for i in range(10000000):
        with open('log', 'a', 0) as outLog:
            line = "iter  "+ str(i) + "\n"
            outLog.write(line)
        vbaseline.update(np.array([2675.0 / 2675, 50.0 / 50, 2675. / 2675, 50.0 / 50]), 422.5518 / 2675)
        vbaseline.update(np.array([2282. / 2675,   47. / 50, 2675. / 2675,   47. / 50]), 29.8503 / 2675)
        vbaseline.update(np.array([2264. / 2675,   45. / 50, 2282. / 2675,   45. / 50]), 11.97 / 2675)
        vbaseline.update(np.array([2255. / 2675,   44. / 50, 2264. / 2675,   44. / 50]), 3 / 2675)
    """

    # testReinforce("./bench/EPFLBenchmarkSuite/benchmarks/arithmetic/adder.aig", "epflAdder")
    # testReinforce("./bench/EPFLBenchmarkSuite/benchmarks/random_control/dec.aig", "epflDec")
    # testReinforce("./bench/MCNC/Combinational/blif/prom1.blif", "prom1")
    # testReinforce("./bench/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    # testReinforce("./bench/MCNC/Combinational/blif/k2.blif", "k2")
    # testReinforce("./bench/ISCAS/blif/c5315.blif", "c5315")
    # testReinforce("./bench/ISCAS/blif/c6288.blif", "c6288")
    # testReinforce("./bench/MCNC/Combinational/blif/apex1.blif", "apex1")
    # testReinforce("./bench/MCNC/Combinational/blif/bc0.blif", "bc0")
    # testReinforce("./bench/i10.aig", "i10")

    
    dir = "./bench/"

    if os.path.exists("./Reinforced_Survey_Area.pkl"):
        df_area = pd.read_pickle("Reinforced_Survey_Area.pkl")
    else:
        df_area = pd.DataFrame(columns=["Benchmark","Compress2rs Area","Reinforced Area"])

    if os.path.exists("./Reinforced_Survey_Delay.pkl"):
        df_delay = pd.read_pickle("Reinforced_Survey_Delay.pkl")
    else:
        df_delay = pd.DataFrame(columns=["Benchmark","Compress2rs Delay","Reinforced Delay"])

    for subdir, dirs, files in os.walk(dir,topdown=True):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".aig"):
                start = time.time()
                print("Running Reinforce on ",file,".....",sep='')
                command = testReinforce(filepath, file[:-4])
                # Comparing the runs of compress2rs and new sequence
                print("Running ABC on ",file,".....",sep='')
                # Find compress2rs stats
                writeABC(filepath, command, opt=0)
                runABC()
                c_area, c_delay = extract_data()
                # Find custom script stats
                writeABC(filepath, command, opt=1)
                runABC()
                r_area, r_delay = extract_data()
                # Aggregate the results
                df_area.loc[0 if pd.isnull(df_area.index.max()) else df_area.index.max() + 1] = [file, c_area, r_area]
                df_delay.loc[0 if pd.isnull(df_delay.index.max()) else df_delay.index.max() + 1] = [file, c_delay, r_delay]
                print("\n", df_area.loc[df_area.index.max()], "\n")
                print("\n", df_delay.loc[df_delay.index.max()], "\n")
                # Update the pickle files
                df_area.to_pickle("Reinforced_Survey_Area.pkl")
                df_delay.to_pickle("Reinforced_Survey_Delay.pkl")            
                end = time.time()
                print("Time Elapsed for ", file, " : ", end-start, "seconds\n")
    
    visualize(df_area, df_delay)

