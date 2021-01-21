##
# @file env.py
# @author Keren Zhu
# @date 10/25/2019
# @brief The environment classes
#

import abc_py as abcPy
import numpy as np
import graphExtractor as GE
import torch
import pandas as pd
from dgl.nn.pytorch import GraphConv
import dgl



class EnvNaive2(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        self.lenSeq = 0
        self._abc.read(self._aigfile)
        initStats = self._abc.aigStats() # The initial AIG statistics
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)
        self.resyn2() # run a compress2rs as target
        self.resyn2()
        resyn2Stats = self._abc.aigStats()
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / 20.0 # 18 is the length of compress2rs sequence
        print("baseline num AND ", resyn2Stats.numAnd, " total reward ", totalReward )
    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
    def reset(self):
        self.lenSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats() # The initial AIG statistics
        self._curStats = self._lastStats # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= 20):
            done = True
        return nextState,reward,done,0
    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K      10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        #self.actsTaken[actionIdx] += 1
        self.lenSeq += 1
        """
        # Compress2rs actions
        if actionIdx == 0:
            self._abc.balance(l=True) # b -l
        elif actionIdx == 1:
            self._abc.resub(k=6, l=True) # rs -K 6 -l
        #elif actionIdx == 2:
        #    self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        #elif actionIdx == 3:
        #    self._abc.resub(k=8, l=True) # rs -K 8 -l
        #elif actionIdx == 4:
        #    self._abc.resub(k=10, l=True) # rs -K 10 -l
        #elif actionIdx == 5:
        #    self._abc.resub(k=12, l=True) # rs -K 12 -l
        #elif actionIdx == 6:
        #    self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif actionIdx == 2:
            self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif actionIdx == 3:
            self._abc.rewrite(l=True) # rw -l
        #elif actionIdx == 3:
        #    self._abc.rewrite(l=True, z=True) # rwz -l
        elif actionIdx == 4:
            self._abc.refactor(l=True) # rf -l
        #elif actionIdx == 4:
        #    self._abc.refactor(l=True, z=True) # rfz -l
        elif actionIdx == 5: # terminal
            self._abc.end()
            return True
        else:
            assert(False)
        """
        if actionIdx == 0:
            self._abc.balance(l=False) # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False) # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False) # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif actionIdx == 4:
            self._abc.refactor(l=False, z=True) #rs
        elif actionIdx == 5:
            self._abc.end()
            return True
        else:
            assert(False)
        """
        elif actionIdx == 3:
            self._abc.rewrite(z=True) #rwz
        elif actionIdx == 4:
            self._abc.refactor(z=True) #rfz
        """


        # update the statitics
        self._lastStats = self._curStats
        self._curStats = self._abc.aigStats()
        return False
    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions())
        np.put(oneHotAct, self.lastAct, 1)
        lastOneHotActs  = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1/3
        lastOneHotActs[self.lastAct3] += 1/3
        lastOneHotActs[self.lastAct] += 1/3
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev,
            self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.lenSeq) / 20.0])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        #combined = np.expand_dims(combined, axis=0)
        #return stateArray.astype(np.float32)
        return torch.from_numpy(combined.astype(np.float32)).float()

    def reward(self):
        if self.lastAct == 5: #term
            return 0
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 5
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    def statValue(self, stat):
        return float(stat.numAnd)  / float(self.initNumAnd) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()






class EnvGraph(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile, cmds):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        self._actionSpace = cmds
        self.timeSeq = 0
        self._readtime = self._abc.read(self._aigfile)
        initStats = self._abc.aigStats() # The initial AIG statistics
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)
        self._runtimeBaseline = self.compress2rs()
        compress2rsStats = self._abc.aigStats()
        totalReward = self.statValue(initStats) - self.statValue(compress2rsStats)
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", compress2rsStats.numAnd, "Baseline Level ", compress2rsStats.lev, " Total Reward ", totalReward)

    def getRuntimeBaseline(self):
        return self._runtimeBaseline    
    
    def resyn2(self):
        t = 0
        t += self._abc.balance(l=False)
        t += self._abc.rewrite(l=False)
        t += self._abc.refactor(l=False)
        t += self._abc.balance(l=False)
        t += self._abc.rewrite(l=False)
        t += self._abc.rewrite(l=False, z=True)
        t += self._abc.balance(l=False)
        t += self._abc.refactor(l=False, z=True)
        t += self._abc.rewrite(l=False, z=True)
        t += self._abc.balance(l=False)
        return t
    
    def reset(self):
        self.timeSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats() # The initial AIG statistics
        self._curStats = self._lastStats # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    
    def close(self):
        self.reset()
    
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.timeSeq >= self._runtimeBaseline):
            done = True
        return nextState, reward, done, 0

    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        action = self._actionSpace[actionIdx] # Map User Action Space to the Complete. 
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        t = 0
        if action == 0:
            t = self._abc.balance(l=False) # b
        elif action == 1:
            t = self._abc.balance(l=True) # b -l
        elif action == 2:
            t = self._abc.rewrite(l=False) # rw
        elif action == 3:
            t = self._abc.rewrite(l=True) # rw -l
        elif action == 4:
            t = self._abc.rewrite(l=False, z=True) # rw -z
        elif action == 5:
            t = self._abc.rewrite(l=True, z=True) # rw -z -l
        elif action == 6:
            t = self._abc.refactor(l=False) # rf
        elif action == 7:
            t = self._abc.refactor(l=True) # rf -l
        elif action == 8:
            t = self._abc.refactor(l=False, z=True) # rf -z
        elif action == 9:
            t = self._abc.refactor(l=True, z=True) # rf - z -l
        elif action == 10:
            t = self._abc.resub(k=4, l=True) # rs -k 4 -l
        elif action == 11:
            t = self._abc.resub(k=5, l=True) # rs -k 5 -l
        elif action == 12:
            t = self._abc.resub(k=6, l=True) # rs -k 6 -l
        elif action == 13:
            self._abc.end()
            return True, -1.0
        else:
            assert(False)

        # update the statitics
        self.timeSeq += t
        self._lastStats = self._curStats
        self._curStats = self._abc.aigStats()
        return False, t
    
    def getCommand(self, actions):
        """
        @brief Returns abc command for a list of actions
        """
        cmd = ""
        for action in actions:
            act = self._actionSpace[action]
            if act == 0:
                cmd += "balance; "
            elif act == 1:
                cmd += "balance -l; "
            elif act == 2:
                cmd += "rewrite; "
            elif act == 3:
                cmd += "rewrite -l; "
            elif act == 4:
                cmd += "rewrite -z; "
            elif act == 5:
                cmd += "rewrite -z -l; "
            elif act == 6:
                cmd += "refactor; "
            elif act == 7:
                cmd += "refactor -l; "
            elif act == 8:
                cmd += "refactor -z; "
            elif act == 9:
                cmd += "refactor -z -l; "
            elif act == 10:
                cmd += "resub -K 4 -l; "
            elif act == 11:
                cmd += "resub -K 5 -l; "
            elif act == 12:
                cmd += "resub -K 6 -l; "
        return cmd

    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions())
        np.put(oneHotAct, self.lastAct, 1)
        lastOneHotActs  = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1/3
        lastOneHotActs[self.lastAct3] += 1/3
        lastOneHotActs[self.lastAct] += 1/3
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev,
            self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.timeSeq) / self._runtimeBaseline])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        #combined = np.expand_dims(combined, axis=0)
        #return stateArray.astype(np.float32)
        combined_torch =  torch.from_numpy(combined.astype(np.float32)).float()
        graph = GE.extract_dgl_graph(self._abc)
        return (combined_torch, graph)
    
    def reward(self):
        if self.lastAct == self.numActions(): #terminate
            return 0
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        # return float(stat.lev)  / float(self.initLev)
        return 2*(float(stat.numAnd)/float(self.initNumAnd)) + 5*(float(stat.lev)/float(self.initLev))
        #return stat.numAnd + stat.lev * 10
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def seed(self, sd):
        pass
    
    def compress2rs(self):
        return self._abc.compress2rs()

