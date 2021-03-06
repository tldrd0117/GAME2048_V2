from multiprocessing import Lock
from typing import List
from repo.tensor_multi import TensorMultitModelRepository
from repo.game import GameRepository
from repo.table import TableRepository, Direction
from repo.tree import TableNode, TreeDbRepository

import random
import numpy as np
import sys

class TrainMultiAiService(object):
    def __init__(self, weight) -> None:
        self.tableRepo = TableRepository()
        self.gameRepo = GameRepository()
        self.treeRepo = TreeDbRepository()
        self.tensorModelRepo = TensorMultitModelRepository()
        if weight is not None:
            self.tensorModelRepo.updateTargetModel(weight)
        self.maxQ = []
        self.simulateMaxQ = []
        self.predictedActions = []
        self.simulatePredictedActions = []
        self.turns = []
        self.scores = []
        
    

    def averageList(li):
        if len(li) <= 0:
            return 0
        return sum(li) / len(li)

    def run(self):
        print("run")
        self.gameRepo.initGame()
        simulateCount = 0
        while True:
            dirList = self.tableRepo.getPossibleDirList()
            print(dirList)
            if len(dirList) <= 0:
                break
            for i in range(10):
                simulateCount = simulateCount + 1
                self.simulate()
            action, maxQ = self.selection(self.tableRepo.table, dirList, True)
            if action == -1:
                print("-1")
                idx = random.randrange(0,len(dirList))
                action = dirList[idx]
            else:
                self.maxQ.append(maxQ)
            print(f"action: {str(action)}")
            data = self.tableRepo.moveTable(action)
            
            self.gameRepo.nextTurn(data[1])
            self.tableRepo.genRandom()

            self.gameRepo.printGame()
            self.tableRepo.printTable()
            sys.stdout.flush()
        print("end")
        self.tableRepo.printTable()

        averageMaxQ = sum(self.maxQ) / len(self.maxQ)
        unique1, count1 = np.unique(np.array(self.predictedActions), return_counts=True)
        actions = dict(zip(unique1, count1))
        averageScores = sum(self.scores) / len(self.scores)
        averageTurns = sum(self.turns) / len(self.turns)

        averageSimulateMaxQ = sum(self.simulateMaxQ) / len(self.simulateMaxQ) if len(self.simulateMaxQ) > 0 else 0
        unique2, count2 = np.unique(np.array(self.simulatePredictedActions), return_counts=True)
        simulateActions = dict(zip(unique2, count2))

        self.treeRepo.addGameInfo(averageTurns, averageScores, averageSimulateMaxQ, "TrainMultiAiServiceAverageLRChange", simulateActions, simulateCount + 1)
        self.treeRepo.addGameInfo(self.gameRepo.turn, self.gameRepo.score, averageMaxQ, "TrainMultiAiServiceLRChange", actions, 1)
        return self.tensorModelRepo.memory
    

    def simulate(self):
        tableRepo = TableRepository()
        gameRepo = GameRepository()
        tableRepo.table = self.tableRepo.getCopyTable()
        gameRepo.turn = self.gameRepo.turn
        gameRepo.score = self.gameRepo.score
        rootScore = self.gameRepo.score
        simulateMaxQ = []
        while True:
            node = self.tensorModelRepo.getNode(tableRepo.getCopyTable())
            node.rootScore = rootScore
            dirList = tableRepo.getPossibleDirList()
            if len(dirList) > 0:
                if gameRepo.turn / 2 > len(simulateMaxQ):
                    action, maxQ = self.selection(tableRepo.getCopyTable(), dirList)
                    if action == -1:
                        idx = random.randrange(0,len(dirList))
                        data = tableRepo.moveTable(dirList[idx])
                        action = dirList[idx]
                    else:
                        data = tableRepo.moveTable(action)
                        simulateMaxQ.append(maxQ)
                else:
                    val = random.randrange(0,2)
                    if val == 0:
                        idx = random.randrange(0,len(dirList))
                        data = tableRepo.moveTable(dirList[idx])
                        action = dirList[idx]
                    else:
                        action, maxQ = self.selection(tableRepo.getCopyTable(), dirList)
                        if action == -1:
                            idx = random.randrange(0,len(dirList))
                            data = tableRepo.moveTable(dirList[idx])
                            action = dirList[idx]
                        else:
                            data = tableRepo.moveTable(action)
                            simulateMaxQ.append(maxQ)
                
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
                childNode.score = 1 if data[1] > 0 else 0

                isDup = False
                for d in node.childs:
                    if str(childNode.table) == str(d[0]):
                        isDup = True
                if not isDup:
                    node.childs.append([childNode.table, action])
                childNode.parent = node.table
                self.tensorModelRepo.updateNodes(node)
                self.tensorModelRepo.updateNodes(childNode)

            # ???????????? ????????? ??????????????? 0?????? ??????
            if len(dirList) < 4:
                li = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
                impossibleAction = []
                for dir in li:
                    if dir not in dirList:
                        impossibleAction.append(dir)
                for action in impossibleAction:
                    childNode = self.tensorModelRepo.newNode()
                    childNode.action = action
                    childNode.rootScore = rootScore
                    childNode.score = 0

                    isDup = False
                    for d in node.childs:
                        if str(childNode.table) == str(d[0]):
                            isDup = True
                    if not isDup:
                        node.childs.append([childNode.table, action])
                    childNode.parent = node.table
                    self.tensorModelRepo.appendSamples([childNode])
            if len(dirList) <= 0:
                break


        self.backPropagation(tableRepo.table, gameRepo.score)
        self.turns.append(gameRepo.turn)
        self.scores.append(gameRepo.score)
        self.simulateMaxQ = self.simulateMaxQ + simulateMaxQ


    def selection(self, table: List, dirList, isMain=False):
        nextChildQValue = self.tensorModelRepo.getActionPredicted(table)[0]
        maxQ = np.amax(nextChildQValue)
        action = -1
        if isMain:
            print(f"max:{np.argmax(nextChildQValue)} q: {str(nextChildQValue)}")
            self.predictedActions.append(np.argmax(nextChildQValue))
        else:
            self.simulatePredictedActions.append(np.argmax(nextChildQValue))
        sortValue = list(np.argsort(nextChildQValue))
        for i in reversed(range(0,4)):
            a = sortValue[i]
            if a in dirList:
                action = a
                break
        return action, maxQ


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
        self.tensorModelRepo.resetNodes()

