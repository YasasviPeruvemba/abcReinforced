import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import bisect
import random
from dgl.nn.pytorch import GraphConv
import dgl

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, out_len)

    def forward(self, g):
        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return torch.squeeze(hg)


class FcModel(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModel, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs, option):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32-4)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)
        if "mtl" in option:
            self.gcn = GCN(7, 12, 4)
        else:
            self.gcn = GCN(6, 12, 4)

    def forward(self, x, graph):
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        return x


class PiApprox(object):
    """
    n dimensional continous states
    m discret actions
    """
    def __init__(self, dimStates, numActs, alpha, network, option, path=None):
        """
        @brief approximate policy pi(. | st)
        @param dimStates: Number of dimensions of state space
        @param numActs: Number of the discret actions
        @param alpha: learning rate
        @param network: a pytorch model
        """
        self._path = path
        self._option = option
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs, option)
        #self._network.cuda()
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        self.tau = 0.5 # temperature for gumbel_softmax
        
        if self._path is not None:
            if os.path.exists(self._path + "/" + self._option + "_pi.pth"): 
                checkpoint = torch.load(self._path + "/" + self._option + "_pi.pth")
                self._network.load_state_dict(checkpoint['model_state_dict'])
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self._network.train()

        print("Input Feat Dim of Graph : ",dimStates," Output Channel Dim of Graph : ", numActs)

    def __call__(self, s, graph, phaseTrain=True):
        self._network.eval()
        #s = torch.from_numpy(s).float() #.cuda()
        out = self._network(s, graph)
        #interval = (out.max() - out.min()).data
        #out = (out - out.min().data) / interval
        #normal = self.normalizeLogits(out)
        #probs = F.gumbel_softmax(out, dim=-1, tau = self.tau, hard=True)
        probs = F.softmax(out, dim=-1)
        #w = list(self._network.parameters())
        """
        with open('log', 'a', 0) as outLog:
            line = "logits " + str(out) + "\n" + "action prob " + str(probs) + "\n" 
            outLog.write(line)
        """
        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
            #action = torch.argmax(probs)
        else:
            action = torch.argmax(out)
            #action = torch.argmax(probs)

        return action.data.item()

    def update(self, s, graph, a, gammaT, delta):
        self._network.train()
        prob = self._network(s, graph)#.cuda())
        #logProb = -F.gumbel_softmax(prob, dim=-1, tau = self.tau, hard=True)
        logProb = torch.log_softmax(prob, dim=-1)
        loss = -gammaT * delta * logProb
        """
        with open('log', 'a', 0) as outLog:
            line = "\n\n\nlogProb " + str(logProb) + '\n' 
            line += "prob " + str(prob) + '\n'
            line += "loss " +str(loss) + '\n'
            line += "action "+ str(a) + '\n'
            line += "gammaT " + str(gammaT) + '\n'
            line += "delta " + str(delta) + '\n'
            outLog.write(line)
        """
        self._optimizer.zero_grad()
        loss[a].backward()
        self._optimizer.step()

    def episode(self):
        #self._tau = self._tau * 0.98
        pass

    def save(self, benchmarks):
        if self._path is not None:
            if not os.path.exists(self._path):
                os.system("mkdir " + self._path)
            torch.save({
                'model_state_dict' : self._network.state_dict(),
                'optimizer_state_dict' : self._optimizer.state_dict(),
                'benchmarks' : benchmarks
            }, self._path + "/" + self._option + "_pi.pth")
        else: return




class Baseline(object):
    """
    The dumbest baseline: a constant for every state
    """
    def __init__(self, b):
        self.b = b

    def __call__(self, s):
        return self.b

    def update(self, s, G):
        pass

class BaselineVApprox(object):
    """
    The baseline with approximation of state value V
    """
    def __init__(self, dimStates, alpha, network, option, path=None):
        """
        @brief approximate policy pi(. | st)
        @param dimStates: Number of dimensions of state space
        @param numActs: Number of the discret actions
        @param alpha: learning rate
        @param network: a pytorch model
        """
        self._path = path
        self._option = option
        self._dimStates = dimStates
        self._alpha = alpha
        self._network = network(dimStates, 1)
        #self._network.cuda()
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        """
        def initZeroWeights(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(0.0)
        self._network.apply(initZeroWeights)
        """
        if self._path is not None:
            if os.path.exists(self._path + "/" + self._option + "_baseline.pth"): 
                checkpoint = torch.load(self._path + "/" + self._option + "_baseline.pth")
                self._network.load_state_dict(checkpoint['model_state_dict'])
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self._network.train()

    def __call__(self, state):
        self._network.eval()
        return self.value(state).data
    def value(self, state):
        #state = torch.from_numpy(state).float()
        out = self._network(state)
        return out
    def update(self, state, G):
        self._network.train()
        vApprox = self.value(state)
        loss = (torch.tensor([G]) - vApprox[-1]) ** 2 / 2  
        """
        with open('log', 'a', 0) as outLog:
            line = "loss" + str(loss) + "\n"
            line += "state " + str(state) + "\n"
            line += "approximate " + str(vApprox) + "\n"
            line += "G " + str(torch.tensor([G])) + "\n"
            outLog.write(line)
        """
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
    
    def save(self, benchmarks):
        if self._path is not None:
            if not os.path.exists(self._path):
                os.system("mkdir " + self._path)
            torch.save({
                'model_state_dict' : self._network.state_dict(),
                'optimizer_state_dict' : self._optimizer.state_dict(),
                'benchmarks' : benchmarks
            }, self._path + "/" + self._option + "_baseline.pth")
        else: return


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """
    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value
    def __lt__(self, other):
        return self.value < other.value

class Reinforce(object):
    def __init__(self, env, gamma, pi, baseline):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.memTrajectory = [] # the memorized trajectories. sorted by value
        self.memLength = 4
        self.sumRewards = []
    
    def genTrajectory(self, phaseTrain=True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        time_elapsed = 0
        runtimeBaseline = self._env.getRuntimeBaseline()

        while not term:
            action = self._pi(state[0], state[1], phaseTrain)
            term, t = self._env.takeAction(action)
            time_elapsed += t
            nextState = self._env.state()
            nextReward = self._env.reward()
            states.append(state)
            rewards.append(nextReward)
            actions.append(action)
            state = nextState
            if len(actions) > 20:
                term = True
        return Trajectory(states, rewards, actions, self._env.curStatsValue())
    
    def episode(self, phaseTrain=True):
        trajectory = self.genTrajectory(phaseTrain=phaseTrain) # Generate a trajectory of episode of states, actions, rewards
        self.updateTrajectory(trajectory, phaseTrain)
        self._pi.episode()
        return self._env.returns(), self._env.getCommand(trajectory.actions)
    
    def updateTrajectory(self, trajectory, phaseTrain=True):
        states = trajectory.states
        self.lenSeq = len(states) # Length of the episode
        rewards = trajectory.rewards
        self.sumRewards.append(sum(rewards))
        actions = trajectory.actions
        if not phaseTrain:
            return
        bisect.insort(self.memTrajectory, trajectory) # memorize this trajectory
        for tIdx in range(self.lenSeq):
            G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
            state = states[tIdx]
            action = actions[tIdx]
            baseline = self._baseline(state[0])
            delta = G - baseline
            """
            with open('log', 'a', 0) as outLog:
                line = "update " + str(tIdx) + "\n"
                line += "G " + str(G) + "\n"
                line += "baseline " + str(baseline) + "\n"
                outLog.write(line)
            """
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
        # print("\nEpisode Reward : ",sum(rewards))
