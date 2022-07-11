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
import traceback
import os


class TrainSaveDbMultiAiService(object):
    def __init__(self, episodeCount, currentCount, predictPercent) -> None:
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
        self.predictPercent = predictPercent


    def averageList(li):
        if len(li) <= 0:
            return 0
        return sum(li) / len(li)

    def run(self):
        try:
            self.startDate = datetime.datetime.now()
            print("run")
            self.gameRepo.initGame()
            simulateCount = 0

            # dirList = self.tableRepo.getPossibleDirList()
            # print(dirList)
            # if len(dirList) <= 0:
            #     break
            self.tableRepo.genRandom()
            print(f"pid: {str(os.getpid())}")
            print(f"episodeCount: {str(self.episodeCount)}")
            print(f"currentCount: {str(self.currentCount)}")
            print(f"predictPercent: {str(self.predictPercent)}")
            length = 0
            while True:
                simulateCount = simulateCount + 1
                self.simulate()
                
                if self.tensorModelRepo.memorySize > length + 1000:
                    print(str(self.tensorModelRepo.memorySize))
                    length = self.tensorModelRepo.memorySize

                if self.tensorModelRepo.memorySize > 7000:
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
            # self.tableRepo.printTable()
            averageScores = sum(self.scores) / len(self.scores) if len(self.scores) > 0 else 0
            averageTurns = sum(self.turns) / len(self.turns) if len(self.turns) > 0 else 0

            averageSimulateMaxQ = sum(self.simulateMaxQ) / len(self.simulateMaxQ) if len(self.simulateMaxQ) > 0 else 0
            unique2, count2 = np.unique(np.array(self.simulatePredictedActions), return_counts=True)
            simulateActions = dict(zip(unique2, count2))
            print(str(averageTurns))
            print(str(averageScores))
            print(str(simulateActions))
            print(f"{str(self.currentCount)}/{str(self.episodeCount)}")

            self.treeRepo.addGameInfo(averageTurns, averageScores, averageSimulateMaxQ, "TrainMultiAiServiceAverageLRChange", simulateActions, simulateCount + 1)
            # self.treeRepo.addGameInfo(self.gameRepo.turn, self.gameRepo.score, averageMaxQ, "TrainMultiAiServiceLRChange", actions, 1)
            return self.startDate, datetime.datetime.now()
        except Exception:
            print(traceback.format_exc())
    
    def appendWrongAction(self, action, parentTable, dirList):
        childNode = self.tensorModelRepo.newNode()
        if action < 2:
            childNode.action = action
            childNode.parent = parentTable
            childNode.score = (-0.1 * len(dirList)) if len(dirList) > 0 else -0.1
            self.tensorModelRepo.appendSamples([childNode])
        else:
            childNode.action = action - 2
            childNode.parent = self.tableRepo.getRotateTableCounterClockWise(parentTable)
            childNode.score = (-0.1 * len(dirList)) if len(dirList) > 0 else -0.1
            self.tensorModelRepo.appendSamples([childNode])
    
    def appendSuccessAction(self, action, parentTable, table, isReward, gameRepo: GameRepository):
        if action < 2:
            childNode = self.tensorModelRepo.getNode(table)
            childNode.action = action
            childNode.parent = parentTable
            # childNode.rootScore = rootScore
            childNode.score = (0.01 * gameRepo.turn) if isReward else 0.01
            self.tensorModelRepo.appendSamples([childNode])
        else:
            childNode = self.tensorModelRepo.getNode(self.tableRepo.getRotateTableCounterClockWise(table))
            childNode.action = action - 2
            childNode.parent = self.tableRepo.getRotateTableCounterClockWise(parentTable)
            # childNode.rootScore = rootScore
            childNode.score = (0.01 * gameRepo.turn) if isReward else 0.01
            self.tensorModelRepo.appendSamples([childNode])


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
            if random.random() < self.predictPercent:
                action, maxQ, isAction = self.selection(tableRepo.getCopyTable(), dirList)
                if isAction == False:
                    self.appendWrongAction(action, tableRepo.getCopyTable(), dirList)
                    break
                data = tableRepo.moveTable(action)
                simulateMaxQ.append(maxQ)
            else:
                action = random.randrange(0,4)
                if action not in dirList:
                    self.appendWrongAction(action, tableRepo.getCopyTable(), dirList)
                    break
                data = tableRepo.moveTable(action)
            if not data[0]:
                print("same move")
                break

            gameRepo.nextTurn(data[1])
            tableRepo.genRandom()
            self.appendSuccessAction(action, node.table, tableRepo.getCopyTable(), data[1]>0, gameRepo)
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
