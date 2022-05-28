

from genericpath import exists
from typing import Any, Dict, List
import pickle
import gc

class TableNode:
    table: List
    action: int = -1
    visit: int = 0
    scores: List = []
    parent: str = None
    childs: List[Dict[str,int]] = []
    def print(self):
        print("table")
        for row in range(0,4):
            print(self.table[row])
        print(f"action: {str(self.action)}")
        print(f"visit: {str(self.visit)}")
        print(f"scores: {str(self.scores)}")
        print(f"parent: {self.parent}")
        print(f"childs: {str(self.childs)}")



class TreeRepository(object):
    nodes = {}
    savePath = "app/data/nodes.pickle_6"
    loadPath = "app/data/nodes.pickle_6"
    def __init__(self, idx) -> None:
        self.loadPath = self.loadPath[:-1] + str(idx)
        self.savePath = self.savePath[:-1] + str(idx + 1)
        print(self.loadPath)
        print(self.savePath)
        self.loadNodes()

    def saveNodes(self):
        print("save...")
        with open(self.savePath, "wb") as fw:
            pickle.dump(self.nodes, fw)
        print(len(self.nodes.keys()))
        print("saveComplete")
        del self.nodes
        gc.collect()
    
    def loadNodes(self):
        print("load...")
        if exists(self.loadPath):
            with open(self.loadPath, "rb") as fr:
                self.nodes = pickle.load(fr)
        else:
            self.nodes = {}
        print(len(self.nodes.keys()))

    def updateNodes(self, node: TableNode):
        self.nodes[str(node.table)] = node

    def getNode(self, table: List):
        if str(table) in self.nodes:
            node = self.nodes[str(table)]
        else:
            node = TableNode()
            node.table = table
            node.childs = []
            node.scores = []
        return node
    
    def getCopyTree(self):
        return {key: value[:] for key, value in self.nodes.items()}

    def getExistNode(self, table: str):
        node = None
        if table in self.nodes:
            node = self.nodes[table]
        return node

    def printNodes(self):
        for node in self.nodes:
            self.nodes[node].print()
    