import sys
#sys.path.append("/afs/pd.inf.tu-dresden.de/users/yape863c/.local/lib/python3.8/site-packages/abc_py-0.0.1-py3.8-linux-x86_64.egg")
import abc_py as abcPy
import numpy as np
import graphExtractor as GE
import torch
import pandas as pd
from dgl.nn.pytorch import GraphConv
import dgl



class EnvGraphBalance(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile, cmds, coefs):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self.and_coef = coefs[0]
        self.level_coef = coefs[1]
        self._abc.start()
        self._actionSpace = cmds
        self.timeSeq = 0
        self._readtime = self._abc.read(self._aigfile)
        self._abc.balance(l=True)
        self.initStats = self._abc.aigStats() # The initial AIG statistics
        self.initNumAnd = float(self.initStats.numAnd)
        self.initLev = float(self.initStats.lev)
        self._runtimeBaseline = self.compress2rs_balance()
        compress2rsStats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(compress2rsStats)# Accounting for 18 steps 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", compress2rsStats.numAnd, "Baseline Level ", compress2rsStats.lev, " Total Reward ", totalReward)

    def getRuntimeBaseline(self):
        return self._runtimeBaseline    
    
    def reset(self):
        self.timeSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._abc.balance(l=True)
        self._lastStats = self._abc.aigStats() # The initial AIG statistics
        self._curStats = self._abc.aigStats() # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        self.lenSeq = 0
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
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K 10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        t = 0
        if action == 0:
            t += self._abc.rewrite(l=True) # rw -l
        elif action == 1:
            t += self._abc.rewrite(l=True, z=True) # rw -z -l
        elif action == 2:
            t += self._abc.refactor(l=True) # rf -l
        elif action == 3:
            t += self._abc.refactor(l=True, z=True) # rf - z -l
        # elif action == 4:
        #    t = self._abc.resub(k=4, l=True) # rs -k 4 -l
        # elif action == 5:
        #    t = self._abc.resub(k=5, l=True) # rs -k 5 -l
        elif action == 4:
            t += self._abc.resub(k=6, l=True) # rs -k 6 -l
        elif action == 5:
            t += self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        elif action == 6:
            t += self._abc.resub(k=8, l=True) # rs -K 8 -l
        elif action == 7:
            t += self._abc.resub(k=8, n=2, l=True) # rs -K 8 -N 2 -l
        elif action == 8:
            t += self._abc.resub(k=10, l=True) # rs -K 10 -l
        elif action == 9:
            t += self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif action == 10:
            t += self._abc.resub(k=12, l=True) # rs -K 12 -l
        elif action == 11:
            t += self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif action == 12:
            t += self._abc.resub(k=16, l=True) # rs -K 16 -l
        elif action == 13:
            t += self._abc.resub(k=16, n=2, l=True) # rs - K 16 -N 2 -l
        elif action == 14:
            self._abc.end()
            return True, -1.0
        else:
            assert(False)

        # update the statitics
        self.lenSeq += 1
        self.timeSeq += t
        self.lastActionTime = t
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
                cmd += "rewrite -l; "
            elif act == 1:
                cmd += "rewrite -z -l; "
            elif act == 2:
                cmd += "refactor -l; "
            elif act == 3:
                cmd += "refactor -z -l; "
            # elif act == 4:
            #    cmd += "resub -K 4 -l; "
            # elif act == 5:
            #    cmd += "resub -K 5 -l; "
            elif act == 4:
                cmd += "resub -K 6 -l; "
            elif act == 5:
                cmd += "resub -K 6 -N 2 -l; "
            elif act == 6:
                cmd += "resub -K 8 -l; "
            elif act == 7:
                cmd += "resub -K 8 -N 2 -l; "
            elif act == 8:
                cmd += "resub -K 10 -l; "
            elif act == 9:
                cmd += "resub -K 10 -N 2 -l; "
            elif act == 10:
                cmd += "resub -K 12 -l; "
            elif act == 11:
                cmd += "resub -K 12 -N 2 -l; "
            elif act == 12:
                cmd += "resub -K 16 -l; "
            elif act == 13:
                cmd += "resub -K 16 -N 2 -l; "
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
        return (self.statValue(self._lastStats) - self.statValue(self._curStats))/self.lastActionTime - self._rewardBaseline
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
        return self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev))
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def seed(self, sd):
        pass
    
    def compress2rs_balance(self):
        return self._abc.compress2rs_balance()



class EnvGraph(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile, cmds, coefs):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self.and_coef = coefs[0]
        self.level_coef = coefs[1]
        self._abc.start()
        self._actionSpace = cmds
        self.timeSeq = 0
        self._readtime = self._abc.read(self._aigfile)
        self.initStats = self._abc.aigStats() # The initial AIG statistics
        self.initNumAnd = float(self.initStats.numAnd)
        self.initLev = float(self.initStats.lev)
        self._runtimeBaseline = self.compress2rs()
        compress2rsStats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(compress2rsStats)# Accounting for 18 steps 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", compress2rsStats.numAnd, "Baseline Level ", compress2rsStats.lev, " Total Reward ", totalReward)

    def getRuntimeBaseline(self):
        return self._runtimeBaseline    
    
    def reset(self):
        self.timeSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats() # The initial AIG statistics
        self._curStats = self._abc.aigStats() # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        self.lenSeq = 0
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
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K 10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        t = 0
        if action == 0:
            t = self._abc.balance(l=True) # b -l
        elif action == 1:
            t = self._abc.rewrite(l=True) # rw -l
        elif action == 2:
            t = self._abc.rewrite(l=True, z=True) # rw -z -l
        elif action == 3:
            t = self._abc.refactor(l=True) # rf -l
        elif action == 4:
            t = self._abc.refactor(l=True, z=True) # rf - z -l
        # elif action == 5:
        #    t = self._abc.resub(k=4, l=True) # rs -k 4 -l
        # elif action == 6:
        #    t = self._abc.resub(k=5, l=True) # rs -k 5 -l
        elif action == 5:
            t = self._abc.resub(k=6, l=True) # rs -k 6 -l
        elif action == 6:
            t = self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        elif action == 7:
            t = self._abc.resub(k=8, l=True) # rs -K 8 -l
        elif action == 8:
            t = self._abc.resub(k=8, n=2, l=True) # rs -K 8 -N 2 -l
        elif action == 9:
            t = self._abc.resub(k=10, l=True) # rs -K 10 -l
        elif action == 10:
            t = self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif action == 11:
            t = self._abc.resub(k=12, l=True) # rs -K 12 -l
        elif action == 12:
            t = self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif action == 13:
            t = self._abc.resub(k=16, l=True) # rs -K 16 -l
        elif action == 14:
            t = self._abc.resub(k=16, n=2, l=True) # rs - K 16 -N 2 -l
        elif action == 15:
            self._abc.end()
            return True, -1.0
        else:
            assert(False)

        # update the statitics
        self.lenSeq += 1
        self.timeSeq += t
        self.lastActionTime = t
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
                cmd += "balance -l; "
            elif act == 1:
                cmd += "rewrite -l; "
            elif act == 2:
                cmd += "rewrite -z -l; "
            elif act == 3:
                cmd += "refactor -l; "
            elif act == 4:
                cmd += "refactor -z -l; "
            # elif act == 5:
            #    cmd += "resub -K 4 -l; "
            # elif act == 6:
            #    cmd += "resub -K 5 -l; "
            elif act == 5:
                cmd += "resub -K 6 -l; "
            elif act == 6:
                cmd += "resub -K 6 -N 2 -l; "
            elif act == 7:
                cmd += "resub -K 8 -l; "
            elif act == 8:
                cmd += "resub -K 8 -N 2 -l; "
            elif act == 9:
                cmd += "resub -K 10 -l; "
            elif act == 10:
                cmd += "resub -K 10 -N 2 -l; "
            elif act == 11:
                cmd += "resub -K 12 -l; "
            elif act == 12:
                cmd += "resub -K 12 -N 2 -l; "
            elif act == 13:
                cmd += "resub -K 16 -l; "
            elif act == 14:
                cmd += "resub -K 16 -N 2 -l; "
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
        return (self.statValue(self._lastStats) - self.statValue(self._curStats))/self.lastActionTime - self._rewardBaseline
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
        return self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev))
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def seed(self, sd):
        pass
    
    def compress2rs(self):
        return self._abc.compress2rs()

