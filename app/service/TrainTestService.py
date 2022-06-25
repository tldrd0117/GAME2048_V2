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

class TrainTestService(object):
    def __init__(self) -> None:
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


    def averageList(li):
        if len(li) <= 0:
            return 0
        return sum(li) / len(li)

    def run(self):
        self.startDate = datetime.datetime.now()
        gameCount = 100
        for _ in range(gameCount):
            tableRepo = TableRepository()
            gameRepo = GameRepository()

            print("run")
            gameRepo.initGame()
            while True:
                dirList = tableRepo.getPossibleDirList()
                print(dirList)
                if len(dirList) <= 0:
                    break
                action, maxQ, isAction = self.selection(tableRepo.table, dirList, True)
                if isAction == False:
                    break
                else:
                    self.maxQ.append(maxQ)
                print(f"action: {str(action)}")
                data = tableRepo.moveTable(action)
                
                gameRepo.nextTurn(data[1])
                tableRepo.genRandom()

                gameRepo.printGame()
                tableRepo.printTable()
                sys.stdout.flush()
            print("end")
            tableRepo.printTable()
            self.scores.append(gameRepo.score)
            self.turns.append(gameRepo.turn)

        averageMaxQ = sum(self.maxQ) / len(self.maxQ) if len(self.maxQ) > 0 else 0
        averageScore = sum(self.scores) / len(self.scores) if len(self.scores) > 0 else 0
        averageTurns = sum(self.turns) / len(self.turns) if len(self.turns) > 0 else 0
        unique1, count1 = np.unique(np.array(self.predictedActions), return_counts=True)
        actions = dict(zip(unique1, count1))

        self.treeRepo.addGameInfo(averageTurns, averageScore, averageMaxQ, "TrainMultiAiServiceLRChange", actions, gameCount)
        return self.startDate, datetime.datetime.now()


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
    
    

