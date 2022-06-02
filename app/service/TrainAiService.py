from typing import List
from repo.tensor import TensorModelRepository
from repo.game import GameRepository
from repo.table import TableRepository
from repo.tree import TableNode, TreeDbRepository

import random
import numpy as np
import sys

class TrainAiService(object):
    def __init__(self) -> None:
        self.tableRepo = TableRepository()
        self.gameRepo = GameRepository()
        self.tensorModelRepo = TensorModelRepository()
        self.treeRepo = TreeDbRepository()
        self.scores = []
        self.turns = []

    def run(self):
        self.gameRepo.initGame()
        while True:
            dirList = self.tableRepo.getPossibleDirList()
            print(dirList)
            if len(dirList) <= 0:
                break
            for _ in range(100):
                self.simulate()
            losses = self.tensorModelRepo.losses
            if len(losses) > 0:
                print(sum(losses)/len(losses))
            self.tensorModelRepo.losses = []
            self.tensorModelRepo.updateTargetModel()
            action = self.selection(self.tableRepo.table, dirList, True)
            if action == -1:
                break
            print(f"action: {str(action)}")
            data = self.tableRepo.moveTable(action)
            
            self.gameRepo.nextTurn(data[1])
            self.tableRepo.genRandom()

            self.gameRepo.printGame()
            self.tableRepo.printTable()
            sys.stdout.flush()
        print("end")
        self.tableRepo.printTable()
        self.tensorModelRepo.saveModel()

        averageScores = sum(self.scores) / len(self.scores)
        averageTurns = sum(self.turns ) / len(self.turns)
        self.treeRepo.addGameInfo(self.gameRepo.turn, self.gameRepo.score, "TrainAiService")
        self.treeRepo.addGameInfo(averageTurns, averageScores, "TrainAiServiceAverage")
        unique, count = np.unique(np.array(self.tensorModelRepo.predictedQValue), return_counts=True)
        print(dict(zip(unique, count)))
    

    def simulate(self):
        tableRepo = TableRepository()
        gameRepo = GameRepository()
        tableRepo.table = self.tableRepo.getCopyTable()
        gameRepo.turn = self.gameRepo.turn
        gameRepo.score = self.gameRepo.score
        rootScore = self.gameRepo.score
        while True:
            node = self.tensorModelRepo.getNode(tableRepo.getCopyTable())
            node.rootScore = rootScore
            dirList = tableRepo.getPossibleDirList()
            if len(dirList) <= 0:
                break
            val = random.randrange(0,2)
            if val == 0:
                idx = random.randrange(0,len(dirList))
                data = tableRepo.moveTable(dirList[idx])
                action = dirList[idx]
            else:
                action = self.selection(tableRepo.getCopyTable(), dirList)
                if action == -1:
                    idx = random.randrange(0,len(dirList))
                    data = tableRepo.moveTable(dirList[idx])
                    action = dirList[idx]
                else:
                    data = tableRepo.moveTable(action)
            
            if not data[0]:
                print(tableRepo.table)
                print(action)
                print("same move")
                node.print()
                break

            gameRepo.nextTurn(data[1])
            tableRepo.genRandom()

            childNode = self.tensorModelRepo.getNode(tableRepo.getCopyTable())
            childNode.action = action
            childNode.rootScore = rootScore
            isDup = False
            for d in node.childs:
                if str(childNode.table) == str(d[0]):
                    isDup = True
            if not isDup:
                node.childs.append([childNode.table, action])
            childNode.parent = node.table
            self.tensorModelRepo.updateNodes(node)
            self.tensorModelRepo.updateNodes(childNode)
        self.backPropagation(tableRepo.table, gameRepo.score)
        self.scores.append(gameRepo.score)
        self.turns.append(gameRepo.turn)

    
    def selection(self, table: List, dirList, isPrint=False):
        current: TableNode = self.tensorModelRepo.getExistNode(str(table))
        if current == None:
            print("nodeNone")
            return -1
        maxUCT = 0
        nextChildQValue = -1
        if len(current.childs) <= 0:
            return -1
        for childData in current.childs:
            childKey = childData[0]
            child: TableNode = self.tensorModelRepo.getExistNode(str(childKey))
            if len(child.scores) <= 0:
                continue
            qValue = self.tensorModelRepo.getActionPredicted(child.table)[0]
            UCT = np.max(qValue)
            if maxUCT < UCT:
                maxUCT = UCT
                nextChildQValue = qValue       # print(f"maxUCT: {maxUCT}")
        action = -1
        if isPrint:
            print(np.argmax(np.argsort(-nextChildQValue)))
        sortValue = list(np.argsort(-nextChildQValue))
        for i in reversed(range(0,4)):
            a = sortValue.index(i)
            if a in dirList:
                action = a
                break
        return action


    def backPropagation(self, table: List, score: int):
        nodeList = []
        node: TableNode = self.tensorModelRepo.getExistNode(str(table))
        while True:
            nodeList.append(node)
            node.visit = node.visit + 1
            # node.scores.append(score)
            if node.parent is None:
                break
            node = self.tensorModelRepo.getExistNode(str(node.parent))
        self.tensorModelRepo.appendSamples(nodeList)
        self.tensorModelRepo.trainModel()


