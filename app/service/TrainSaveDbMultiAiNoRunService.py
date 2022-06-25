from multiprocessing import Lock
from typing import List

from repo.tensor_multi_db import TensorMultitModelDbRepository
from repo.game import GameRepository
from repo.table import TableRepository, Direction
from repo.tree import TableNode, TreeDbRepository

import random
import numpy as np
import sys
import datetime
import time

class TrainSaveDbMultiAiService(object):
    def __init__(self, episodeCount, currentCount) -> None:
        self.tableRepo = TableRepository()
        self.gameRepo = GameRepository()
        self.treeRepo = TreeDbRepository()
        self.tensorModelRepo = TensorMultitModelDbRepository()
        self.tensorModelRepo.loadModel()
        self.maxQ = []
        self.simulateMaxQ = []
        self.predictedActions = []
        self.simulatePredictedActions = []
        self.turns = []
        self.scores = []
        self.episodeCount = episodeCount
        self.currentCount = currentCount


    def averageList(li):
        if len(li) <= 0:
            return 0
        return sum(li) / len(li)

    def run(self):
        self.startDate = datetime.datetime.now()
        print("run")
        self.gameRepo.initGame()
        simulateCount = 0

        # dirList = self.tableRepo.getPossibleDirList()
        # print(dirList)
        # if len(dirList) <= 0:
        #     break
        self.tableRepo.genRandom()
        self.tableRepo.printTable()
        length = 0
        while True:
            simulateCount = simulateCount + 1
            self.simulate()
            
            if self.tensorModelRepo.memorySize > length + 1000:
                print(str(self.tensorModelRepo.memorySize))
                length = self.tensorModelRepo.memorySize

            if self.tensorModelRepo.memorySize > 50000:
                break
        # action, maxQ = self.selection(self.tableRepo.table, dirList, True)
        # if action == -1:
        #     break
        #     # print("-1")
        #     # idx = random.randrange(0,len(dirList))
        #     # action = dirList[idx]
        # else:
        #     self.maxQ.append(maxQ)
        # print(f"action: {str(action)}")
        # data = self.tableRepo.moveTable(action)
        
        # self.gameRepo.nextTurn(data[1])
        # self.tableRepo.genRandom()

        # self.gameRepo.printGame()
        # self.tableRepo.printTable()
        # sys.stdout.flush()
        
        print("end")
        print(str(self.tensorModelRepo.memorySize))
        self.tableRepo.printTable()
        averageScores = sum(self.scores) / len(self.scores) if len(self.scores) > 0 else 0
        averageTurns = sum(self.turns) / len(self.turns) if len(self.turns) > 0 else 0

        averageSimulateMaxQ = sum(self.simulateMaxQ) / len(self.simulateMaxQ) if len(self.simulateMaxQ) > 0 else 0
        unique2, count2 = np.unique(np.array(self.simulatePredictedActions), return_counts=True)
        simulateActions = dict(zip(unique2, count2))

        self.treeRepo.addGameInfo(averageTurns, averageScores, averageSimulateMaxQ, "TrainMultiAiServiceAverageLRChange", simulateActions, simulateCount + 1)
        # self.treeRepo.addGameInfo(self.gameRepo.turn, self.gameRepo.score, averageMaxQ, "TrainMultiAiServiceLRChange", actions, 1)
        return self.startDate, datetime.datetime.now()
    

    def simulate(self):
        tableRepo = TableRepository()
        gameRepo = GameRepository()
        tableRepo.table = self.tableRepo.getCopyTable()
        gameRepo.turn = self.gameRepo.turn
        gameRepo.score = self.gameRepo.score
        simulateMaxQ = []
        while True:
            node = self.tensorModelRepo.getNode(tableRepo.getCopyTable())
            dirList = tableRepo.getPossibleDirList()
            # if len(dirList) > 0:
            if random.random() < self.currentCount / self.episodeCount:
                action, maxQ, isAction = self.selection(tableRepo.getCopyTable(), dirList)
                if isAction == False:
                    childNode = self.tensorModelRepo.newNode()
                    if action < 2:
                        childNode.action = action
                        childNode.parent = node.table
                        childNode.score = -100
                        self.tensorModelRepo.appendSamples([childNode])
                    else:
                        childNode.action = action - 2
                        childNode.parent = self.tableRepo.getRotateTableCounterClockWise(tableRepo.getCopyTable())
                        childNode.score = -100
                        self.tensorModelRepo.appendSamples([childNode])

                    break
                data = tableRepo.moveTable(action)
                simulateMaxQ.append(maxQ)
            else:
                action = random.randrange(0,4)
                if action not in dirList:
                    childNode = self.tensorModelRepo.newNode()
                    if action < 2:
                        childNode.action = action
                        childNode.parent = node.table
                        childNode.score = -100
                        self.tensorModelRepo.appendSamples([childNode])
                    else:
                        childNode.action = action - 2
                        childNode.parent = self.tableRepo.getRotateTableCounterClockWise(tableRepo.getCopyTable())
                        childNode.score = -100
                        self.tensorModelRepo.appendSamples([childNode])
                    break
                
                data = tableRepo.moveTable(action)
            if not data[0]:
                print(tableRepo.table)
                print(action)
                print("same move")
                node.print()
                break

            gameRepo.nextTurn(data[1])
            tableRepo.genRandom()
                
            if action < 2:
                childNode = self.tensorModelRepo.getNode(tableRepo.getCopyTable())
                childNode.action = action
                childNode.parent = node.table
                # childNode.rootScore = rootScore
                childNode.score = 1 if data[1] > 0 else 0

                self.tensorModelRepo.appendSamples([childNode])
            else:
                childNode = self.tensorModelRepo.getNode(self.tableRepo.getRotateTableCounterClockWise(tableRepo.getCopyTable()))
                childNode.action = action - 2
                childNode.parent = self.tableRepo.getRotateTableCounterClockWise(node.table)
                # childNode.rootScore = rootScore
                childNode.score = 1 if data[1] > 0 else 0

                self.tensorModelRepo.appendSamples([childNode])


                # isDup = False
                # for d in node.childs:
                #     if str(childNode.table) == str(d[0]):
                #         isDup = True
                # if not isDup:
                #     node.childs.append([childNode.table, action])
                # childNode.parent = node.table
                # self.tensorModelRepo.updateNodes(node)
                # self.tensorModelRepo.updateNodes(childNode)

            # 불가능한 액션은 최종보상을 0으로 세팅
            # if len(dirList) < 4:
            #     li = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
            #     impossibleAction = []
            #     for dir in li:
            #         if dir not in dirList:
            #             impossibleAction.append(dir)
            #     for action in impossibleAction:
            #         newNode = self.tensorModelRepo.newNode()
            #         newNode.action = action
            #         # childNode.rootScore = rootScore
            #         newNode.score = 0
            #         newNode.parent = node.table
            #         self.tensorModelRepo.appendSamples([newNode])
            # if len(dirList) <= 0:
            #     break
        # self.backPropagation(tableRepo.table, gameRepo.score)
        self.turns.append(gameRepo.turn)
        self.scores.append(gameRepo.score)
        self.simulateMaxQ = self.simulateMaxQ + simulateMaxQ
        self.tensorModelRepo.resetNodes()


    def selection(self, table: List, dirList, isMain=False):
        nextChildQValue1 = self.tensorModelRepo.getActionPredicted(table)[0]
        nextChildQValue2 = self.tensorModelRepo.getActionPredicted(self.tableRepo.getRotateTableCounterClockWise(table))[0]
        nextChildQValue = np.concatenate((nextChildQValue1, nextChildQValue2))
        maxQ = np.amax(nextChildQValue)
        action = -1
        if isMain:
            print(f"max:{np.argmax(nextChildQValue)} q: {str(nextChildQValue)}")
            self.predictedActions.append(np.argmax(nextChildQValue))
        else:
            self.simulatePredictedActions.append(np.argmax(nextChildQValue))
        sortValue = list(np.argsort(-nextChildQValue))
        if sortValue[0] in dirList:
            action = sortValue[0]
        return sortValue[0], maxQ, action != -1
        # sortValue = list(np.argsort(nextChildQValue))
        # for i in reversed(range(0,4)):
        #     a = sortValue[i]
        #     if a in dirList:
        #         action = a
        #         break
        # return action, maxQ


    def backPropagation(self, table: List, score: int):
        nodeList = []
        node: TableNode = self.tensorModelRepo.getExistNode(str(table))
        while True:
            nodeList.append(node)
            # node.visit = node.visit + 1
            # node.scores.append(score)
            if node.parent is None:
                break
            node = self.tensorModelRepo.getExistNode(str(node.parent))
        self.tensorModelRepo.appendSamples(nodeList)
        self.tensorModelRepo.resetNodes()

