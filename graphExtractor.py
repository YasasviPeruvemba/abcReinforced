import sys
#sys.path.append("/afs/pd.inf.tu-dresden.de/users/yape863c/.local/lib/python3.8/site-packages/abc_py-0.0.1-py3.8-linux-x86_64.egg")
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import numpy as np
import dgl
import torch

def symmetricLaplacian(abc):
    numNodes = abc.numNodes()
    L = np.zeros((numNodes, numNodes))
    print("numNodes", numNodes)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        degree = float(aigNode.numFanouts())
        if (aigNode.hasFanin0()):
            degree += 1.0
            fanin = aigNode.fanin0()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        if (aigNode.hasFanin1()):
            degree += 1.0
            fanin = aigNode.fanin1()
            L[nodeIdx][fanin] = -1.0
            L[fanin][nodeIdx] = -1.0
        L[nodeIdx][nodeIdx] = degree
    return L

def symmetricLapalacianEigenValues(abc):
    L = symmetricLaplacian(abc)
    print("L", L)
    eigVals = np.real(LA.eigvals(L))
    print("eigVals", eigVals)
    return eigVals

def extract_dgl_graph(abc):
    numNodes = abc.numNodes()
    G = dgl.DGLGraph()
    G.add_nodes(numNodes)
    features = torch.zeros(numNodes, 6)
    for nodeIdx in range(numNodes):
        aigNode = abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        if nodeType == 6: continue
        features[nodeIdx][nodeType] = 1.0
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            G.add_edge(fanin, nodeIdx)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            G.add_edge(fanin, nodeIdx)
    G = dgl.add_self_loop(G)
    G.ndata['feat'] = features
    return G


def extract_dgl_graph_mtl(mtl):
    numNodes = mtl.migStats().numMigNodes
    G = dgl.DGLGraph()
    G.add_nodes(numNodes)
    features = torch.zeros(numNodes, 7)
    for node in range(numNodes):
        migNode = mtl.migNode(node)
        nodeType = migNode.nodeType()
        if nodeType == 9: continue
        elif nodeType == 8:
            features[node][0] = 1.0
            features[node][2] = 1.0
        elif nodeType == 7:
            features[node][1] = 1.0
            features[node][2] = 1.0
        else:
            features[node][nodeType] = 1.0
        if migNode.hasFanin0():
            f = migNode.fanin0()
            G.add_edge(f, node)
        elif migNode.hasFanin1():
            f = migNode.fanin1()
            G.add_edge(f, node)
        elif migNode.hasFanin2():
            f = migNode.fanin2()
            G.add_edge(f, node)
    G = dgl.add_self_loop(G)
    G.ndata['feat'] = features
    return G