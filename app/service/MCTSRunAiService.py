import random
from typing import List
from repo.tree import TableNode
from repo.table import TableRepository
from repo.game import GameRepository
from repo.tree import TreeRepository, TreeDbRepository
import math

class MCTSRunAiService(object):
    def __init__(self) -> None:
        self.tableRepo = TableRepository()
        self.gameRepo = GameRepository()
        self.treeRepo = TreeDbRepository()
        # self.treeRepo = TreeRepository(idx)
        self.existAction = 0
        self.nonExistAction = 0
    

    def run(self):
        self.gameRepo.initGame()
        while True:
            dirList = self.tableRepo.getPossibleDirList()
            print(dirList)
            if len(dirList) <= 0:
                break
            # action = self.selection(self.tableRepo.table)
            action = self.selectOrSimulate(True)
            if action == -1:
                print("no action")
                idx = random.randrange(0,len(dirList))
                data = self.tableRepo.moveTable(dirList[idx])
                action = dirList[idx]
            else:
                data = self.tableRepo.moveTable(action)
            print(f"action: {str(action)}")
            data = self.tableRepo.moveTable(action)
            
            self.gameRepo.nextTurn(data[1])
            self.tableRepo.genRandom()

            self.gameRepo.printGame()
            self.tableRepo.printTable()
        print("end")
        print(f"existActions: {str(self.existAction)} nonExistActions: {str(self.nonExistAction)}")
        self.tableRepo.printTable()
    

    def selectOrSimulate(self, isFirst = False):
        action = self.selection(self.tableRepo.table, self.gameRepo.score)
        if action == -1:
            if isFirst:
                self.nonExistAction = self.nonExistAction + 1
            print("action None")
            for _ in range(30):
                self.simulate()
            return self.selectOrSimulate()
        if isFirst:
            self.existAction = self.existAction + 1
        return action
    

    def simulate(self):
        tableRepo = TableRepository()
        gameRepo = GameRepository()
        tableRepo.table = self.tableRepo.getCopyTable()
        gameRepo.turn = self.gameRepo.turn
        gameRepo.score = self.gameRepo.score
        while True:
            node = self.treeRepo.getNode(tableRepo.getCopyTable())
            dirList = tableRepo.getPossibleDirList()
            if len(dirList) <= 0:
                break
            action = self.selection(tableRepo.getCopyTable(), gameRepo.score)
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

            childNode = self.treeRepo.getNode(tableRepo.getCopyTable())
            childNode.action = action
            isDup = False
            for d in node.childs:
                if str(childNode.table) == d[0]:
                    isDup = True
            if not isDup:
                node.childs.append([str(childNode.table), action])
            childNode.parent = str(node.table)
            self.treeRepo.updateNodes(node)
            self.treeRepo.updateNodes(childNode)
        self.backPropagation(tableRepo.table, gameRepo.score)


    def selection(self, table: List, currentScore: int):
        current: TableNode = self.treeRepo.getExistNode(str(table))
        if current == None:
            print("nodeNone")
            return -1
        maxUCT = 0
        nextChildAction = -1
        if len(current.childs) <= 10:
            return -1
        for childData in current.childs:
            childKey = childData[0]
            child: TableNode = self.treeRepo.getExistNode(childKey)
            if len(child.scores) <= 0:
                continue
            UCT = sum(child.scores) / len(child.scores) - currentScore #+ math.sqrt(2) * math.sqrt(math.log(current.visit) / child.visit)
            if maxUCT < UCT:
                maxUCT = UCT
                nextChildAction = childData[1]
        # print(f"maxUCT: {maxUCT}")
        return nextChildAction


    def backPropagation(self, table: List, score: int):
        node: TableNode = self.treeRepo.getExistNode(str(table))
        while True:
            node.visit = node.visit + 1
            node.scores.append(score)
            self.treeRepo.updateNodes(node)
            if node.parent is None:
                break
            node = self.treeRepo.getExistNode(node.parent)


