import sys
#sys.path.append("/afs/pd.inf.tu-dresden.de/users/yape863c/.local/lib/python3.8/site-packages/abc_py-0.0.1-py3.8-linux-x86_64.egg")
import abcPy
import mtlPy
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
        if totalReward < 0:
            totalReward = 0
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
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
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
        if totalReward < 0:
            totalReward = 0
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
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def seed(self, sd):
        pass
    
    def compress2rs(self):
        return self._abc.compress2rs()


class EnvGraphDch(object):
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
        self._runtimeBaseline = 2*(self.compress2rs() + self.dch() + self._abc.balance(l=True)) 
        targetStats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(targetStats)# Accounting for 18 steps
        if totalReward < 0:
            totalReward = 0 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", targetStats.numAnd, "Baseline Level ", targetStats.lev, " Total Reward ", totalReward)

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
        elif action == 5:
            t = self._abc.resub(k=8, l=True) # rs -K 8 -l
        elif action == 6:
            t = self._abc.resub(k=8, n=2, l=True) # rs -K 8 -N 2 -l
        elif action == 7:
            t = self._abc.resub(k=10, l=True) # rs -K 10 -l
        elif action == 8:
            t = self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif action == 9:
            t = self._abc.resub(k=12, l=True) # rs -K 12 -l
        elif action == 10:
            t = self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif action == 11:
            t = self._abc.resub(k=16, l=True) # rs -K 16 -l
        elif action == 12:
            t = self._abc.resub(k=16, n=2, l=True) # rs - K 16 -N 2 -l
        elif action == 13:
            t = self._abc.dch() + self._abc.balance(l=True) # dch
        elif action == 14:
            t = self._abc.dc2() # dc2
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
            elif act == 5:
                cmd += "resub -K 8 -l; "
            elif act == 6:
                cmd += "resub -K 8 -N 2 -l; "
            elif act == 7:
                cmd += "resub -K 10 -l; "
            elif act == 8:
                cmd += "resub -K 10 -N 2 -l; "
            elif act == 9:
                cmd += "resub -K 12 -l; "
            elif act == 10:
                cmd += "resub -K 12 -N 2 -l; "
            elif act == 11:
                cmd += "resub -K 16 -l; "
            elif act == 12:
                cmd += "resub -K 16 -N 2 -l; "
            elif action == 13:
                cmd += "dch; balance -l; "
            elif action == 14:
                cmd += "dc2; "
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
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def compress2rs(self):
        return self._abc.compress2rs()

    def dch(self):
        return self._abc.dch()


class ReducedEnvGraphDch(object):
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
        self._runtimeBaseline = 2*(self.compress2rs() + self.dch() + self._abc.balance(l=True)) 
        targetStats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(targetStats)# Accounting for 18 steps
        if totalReward < 0:
            totalReward = 0 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", targetStats.numAnd, "Baseline Level ", targetStats.lev, " Total Reward ", totalReward)

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
            t = self._abc.rewrite(l=True, z=True) + self._abc.balance(l=True) # rewrite -z -l; balance -l;
        elif action == 1:
            t = self._abc.dc2() # dc2;
        elif action == 2:
            t = self._abc.dch() + self._abc.balance(l=True) # dch; balance -l;
        elif action == 3:
            t = self._abc.resub(k=8, l=True) + self._abc.refactor(l=True, z=True) + self._abc.resub(k=8, l=True) # resub -K 8 -l; refactor -z -l; resub -K 8 -N 2 -l;
        elif action == 4:
            t = self._abc.resub(k=10, l=True) + self._abc.refactor(l=True, z=True) + self._abc.resub(k=10, l=True) # resub -K 10 -l; refactor -z -l; resub -K 10 -N 2 -l;
        elif action == 5:
            t = self._abc.resub(k=12, l=True) + self._abc.refactor(l=True, z=True) + self._abc.resub(k=12, l=True) # resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l;
        elif action == 6:
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
                cmd += "rewrite -z -l; balance -l; "
            elif act == 1:
                cmd += "dc2; "
            elif act == 2:
                cmd += "dch; balance -l; "
            elif act == 3:
                cmd += "resub -K 8 -l; refactor -z -l; resub -K 8 -N 2 -l; "
            elif act == 4:
                cmd += "resub -K 10 -l; refactor -z -l; resub -K 10 -N 2 -l; "
            elif act == 5:
                cmd += "resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; "
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
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def compress2rs(self):
        return self._abc.compress2rs()

    def dch(self):
        return self._abc.dch()


class EnvGraphMtlDch(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile, cmds, coefs):
        self._mtl = mtlPy.MtlInterface()
        self._aigfile = aigfile
        self.and_coef = coefs[0]
        self.level_coef = coefs[1]
        self._mtl.start()
        self._actionSpace = cmds
        self.timeSeq = 0
        self._readtime = self._mtl.read(self._aigfile)
        self.initStats = self._mtl.migStats() # The initial MIG statistics
        self.initNumAnd = float(self.initStats.numMigNodes)
        self.initLev = float(self.initStats.lev)
        self._runtimeBaseline = 2 * self.baseline() 
        targetStats = self._mtl.migStats()
        totalReward = self.statValue(self.initStats) - self.statValue(targetStats)# Accounting for 18 steps
        if totalReward < 0:
            totalReward = 0 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", targetStats.numMigNodes, "Baseline Level ", targetStats.lev, " Total Reward ", totalReward)

    def getRuntimeBaseline(self):
        return self._runtimeBaseline    

    def baseline(self):
        t = 0
        t += self._mtl.balance()
        t += self._mtl.rewrite(allow_zero_gain = True, use_dont_cares = True)
        t += self._mtl.rewrite()
        t += self._mtl.balance(crit=True)
        t += self._mtl.rewrite(allow_zero_gain = True)
        t += self._mtl.balance()
        t += self._mtl.rewrite(use_dont_cares = True)
        t += self._mtl.balance(crit=True)
        t += self._mtl.balance()
        return t

    def reset(self):
        self.timeSeq = 0
        self._mtl.end()
        self._mtl.start()
        self._mtl.read(self._aigfile)
        self._mtl.balance()
        self._lastStats = self._mtl.migStats() # The initial AIG statistics
        self._curStats = self._mtl.migStats() # the current AIG statistics
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
            t = self._mtl.rewrite() # rewrite;
        elif action == 1:
            t = self._mtl.rewrite(allow_zero_gain=True) # rewrite -l;
        elif action == 2:
            t = self._mtl.rewrite(use_dont_cares=True) # rewrite -z;
        elif action == 3:
            t = self._mtl.rewrite(allow_zero_gain=True, use_dont_cares=True) # rewrite -l -z;
        elif action == 4:
            t = self._mtl.balance() # balance;
        elif action == 5:
            t = self._mtl.balance(crit=True) # balance crit;
        # elif action == 6:
        #     t = self._mtl.resub() # resub;
        # elif action == 7:
        #     t = self._mtl.resub(use_dont_cares=True) # resub -z;
        # elif action == 8:
        #     t = self._mtl.resub(preserve_depth=True) # resub -p;
        # elif action == 9:
        #     t = self._mtl.resub(use_dont_cares=True, preserve_depth=True) # resub;    
        # elif action == 10:
        #     t = self._mtl.refactor() # refactor;
        # elif action == 11:
        #     t = self._mtl.refactor(allow_zero_gain=True) # refactor;
        # elif action == 12:
        #     t = self._mtl.refactor(use_dont_cares=True) # refactor;    
        elif action == 6:
            self._mtl.end()
            return True, -1.0
        else:
            assert(False)

        # update the statitics
        self.lenSeq += 1
        self.timeSeq += t
        self.lastActionTime = t
        self._lastStats = self._curStats
        self._curStats = self._mtl.migStats()
        return False, t
    
    def getCommand(self, actions):
        """
        @brief Returns abc command for a list of actions
        """
        cmd = ""
        for action in actions:
            act = self._actionSpace[action]
            if act == 0:
                cmd += "rewrite; "
            elif act == 1:
                cmd += "rewrite azg; "
            elif act == 2:
                cmd += "rewrite udc; "
            elif act == 3:
                cmd += "rewrite azg udc; "
            elif act == 4:
                cmd += "balance; "
            elif act == 5:
                cmd += "balance crit; "
            # elif act == 6:
            #     cmd += "resub; "
            # elif act == 7:
            #     cmd += "resub udc; "
            # elif act == 8:
            #     cmd += "resub pd; "
            # elif act == 9:
            #     cmd += "resub udc pd; "
            # elif action == 10:
                # cmd += "refactor; " 
            # elif action == 11:
                # cmd += "refactor azg; " 
            # elif action == 12:
                # cmd += "refactor udc; " 
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
        stateArray = np.array([self._curStats.numMigNodes / self.initNumAnd, self._curStats.lev / self.initLev,
            self._lastStats.numMigNodes / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.timeSeq) / self._runtimeBaseline])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        #combined = np.expand_dims(combined, axis=0)
        #return stateArray.astype(np.float32)
        combined_torch =  torch.from_numpy(combined.astype(np.float32)).float()
        graph = GE.extract_dgl_graph_mtl(self._mtl)
        return (combined_torch, graph)
    
    def reward(self):
        if self.lastAct == self.numActions(): #terminate
            return 0
        return (self.statValue(self._lastStats) - self.statValue(self._curStats))/self.lastActionTime - self._rewardBaseline
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numMigNodes , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numMigNodes)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)


 
class EnvReplica(object):
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
        self._runtimeBaseline = 2*(self.resyn2() + self.resyn2()) 
        targetStats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(targetStats)# Accounting for 18 steps
        if totalReward < 0:
            totalReward = 0 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", targetStats.numAnd, "Baseline Level ", targetStats.lev, " Total Reward ", totalReward)

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
        t = 0
        if action == 0:
            t = self._abc.balance(l=False) # b 
        elif action == 1:
            t = self._abc.rewrite(l=False) # rw 
        elif action == 2:
            t = self._abc.rewrite(l=False, z=True) # rw -z 
        elif action == 3:
            t = self._abc.refactor(l=False) # rf 
        elif action == 4:
            t = self._abc.refactor(l=False, z=True) # rf - z 
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
                cmd += "balance; "
            elif act == 1:
                cmd += "rewrite; "
            elif act == 2:
                cmd += "rewrite -z; "
            elif act == 3:
                cmd += "refactor; "
            elif act == 4:
                cmd += "refactor -z; "
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
        combined_torch =  torch.from_numpy(combined.astype(np.float32)).float()
        graph = GE.extract_dgl_graph(self._abc)
        return (combined_torch, graph)
    
    def reward(self):
        if self.lastAct == self.numActions(): #terminate
            return 0
        return (self.statValue(self._lastStats) - self.statValue(self._curStats))/self.lastActionTime - self._rewardBaseline
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def compress2rs(self):
        return self._abc.compress2rs()

    def dch(self):
        return self._abc.dch()

class EnvGraphDchMap(object):
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
        self.initStats = self._abc.map() # The initial AIG statistics
        self.initArea = float(self.initStats.area)
        self.initDepth = float(self.initStats.depth)
        self._runtimeBaseline = 2*(self.compress2rs() + self.dch() + self._abc.balance(l=True)) 
        targetStats = self._abc.map()
        totalReward = self.statValue(self.initStats) - self.statValue(targetStats)# Accounting for 18 steps
        if totalReward < 0:
            totalReward = 0 
        self._rewardBaseline = totalReward / self._runtimeBaseline # Baseline time of compress2rs sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Area ", targetStats.area, "Baseline Depth ", targetStats.depth, " Total Reward ", totalReward)

    def getRuntimeBaseline(self):
        return self._runtimeBaseline    
    
    def reset(self):
        self.timeSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.map() # The initial AIG statistics
        self._curStats = self._abc.map() # the current AIG statistics
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
        elif action == 5:
            t = self._abc.resub(k=8, l=True) # rs -K 8 -l
        elif action == 6:
            t = self._abc.resub(k=8, n=2, l=True) # rs -K 8 -N 2 -l
        elif action == 7:
            t = self._abc.resub(k=10, l=True) # rs -K 10 -l
        elif action == 8:
            t = self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif action == 9:
            t = self._abc.resub(k=12, l=True) # rs -K 12 -l
        elif action == 10:
            t = self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif action == 11:
            t = self._abc.resub(k=16, l=True) # rs -K 16 -l
        elif action == 12:
            t = self._abc.resub(k=16, n=2, l=True) # rs - K 16 -N 2 -l
        elif action == 13:
            t = self._abc.dch() + self._abc.balance(l=True) # dch
        elif action == 14:
            t = self._abc.dc2() # dc2
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
        self._curStats = self._abc.map()
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
            elif act == 5:
                cmd += "resub -K 8 -l; "
            elif act == 6:
                cmd += "resub -K 8 -N 2 -l; "
            elif act == 7:
                cmd += "resub -K 10 -l; "
            elif act == 8:
                cmd += "resub -K 10 -N 2 -l; "
            elif act == 9:
                cmd += "resub -K 12 -l; "
            elif act == 10:
                cmd += "resub -K 12 -N 2 -l; "
            elif act == 11:
                cmd += "resub -K 16 -l; "
            elif act == 12:
                cmd += "resub -K 16 -N 2 -l; "
            elif action == 13:
                cmd += "dch; balance -l; "
            elif action == 14:
                cmd += "dc2; "
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
        stateArray = np.array([self._curStats.area / self.initArea, self._curStats.depth / self.initDepth,
            self._lastStats.area / self.initArea, self._lastStats.depth / self.initDepth])
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
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.area , self._curStats.depth]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.area)/float(self.initArea)) + self.level_coef*(float(stat.depth)/float(self.initDepth)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def compress2rs(self):
        return self._abc.compress2rs()

    def dch(self):
        return self._abc.dch()


class EnvReplicaExact(object):
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
        self._runtimeBaseline = 2*(self.resyn2() + self.resyn2()) 
        targetStats = self._abc.aigStats()
        totalReward = self.statValue(self.initStats) - self.statValue(targetStats)
        if totalReward < 0:
            totalReward = 0 
        self._rewardBaseline = totalReward / 20 # Baseline steps of 2 resyn2 sequence
        print("Baseline Time Taken", self._runtimeBaseline, " Baseline Nodes ", targetStats.numAnd, "Baseline Level ", targetStats.lev, " Total Reward ", totalReward)

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
        t = 0
        if action == 0:
            t = self._abc.balance(l=False) # b 
        elif action == 1:
            t = self._abc.rewrite(l=False) # rw 
        elif action == 2:
            t = self._abc.rewrite(l=False, z=True) # rw -z 
        elif action == 3:
            t = self._abc.refactor(l=False) # rf 
        elif action == 4:
            t = self._abc.refactor(l=False, z=True) # rf - z 
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
                cmd += "balance; "
            elif act == 1:
                cmd += "rewrite; "
            elif act == 2:
                cmd += "rewrite -z; "
            elif act == 3:
                cmd += "refactor; "
            elif act == 4:
                cmd += "refactor -z; "
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
        stepArray = np.array([float(self.lenSeq) / 20])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        combined_torch =  torch.from_numpy(combined.astype(np.float32)).float()
        graph = GE.extract_dgl_graph(self._abc)
        return (combined_torch, graph)
    
    def reward(self):
        if self.lastAct == self.numActions(): #terminate
            return 0
        return (self.statValue(self._lastStats) - self.statValue(self._curStats)) - self._rewardBaseline
    
    def numActions(self):
        return len(self._actionSpace)
    
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    
    def statValue(self, stat):
        return (self.and_coef*(float(stat.numAnd)/float(self.initNumAnd)) + self.level_coef*(float(stat.lev)/float(self.initLev)))/(self.and_coef + self.level_coef)
    
    def curStatsValue(self):
        return self.statValue(self._curStats)
    
    def compress2rs(self):
        return self._abc.compress2rs()

    def dch(self):
        return self._abc.dch()