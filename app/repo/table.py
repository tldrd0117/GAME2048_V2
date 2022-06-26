from enum import IntEnum
import random
from typing import List
import itertools
import os

class Direction(IntEnum):
    LEFT=0
    RIGHT=1
    UP=2
    DOWN=3


class TableRepository(object):
    table = []
    def __init__(self) -> None:
        self.initTable()    

    def initTable(self):
        self.table = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        self.genRandom(True)
        self.genRandom(True)
    
    def getCopyTable(self):
        return self.copyTable(self.table)

    def printTable(self):
        print(f"pid: {str(os.getpid())}")
        print("table:")
        for row in range(0,4):
            print(self.table[row])
    
    def getPossibleDirList(self):
        dir = []
        before = self.copyTable(self.table)
        if self.moveTable(Direction.LEFT)[0]:
            dir.append(Direction.LEFT)
        self.table = self.copyTable(before)
        if self.moveTable(Direction.UP)[0]:
            dir.append(Direction.UP)
        self.table = self.copyTable(before)
        if self.moveTable(Direction.RIGHT)[0]:
            dir.append(Direction.RIGHT)
        self.table = self.copyTable(before)
        if self.moveTable(Direction.DOWN)[0]:
            dir.append(Direction.DOWN)
        self.table = self.copyTable(before)
        return dir
    

    def genRandom(self, isFirst = False):
        zeroCoord = []
        for row in range(0,4):
            for col in range(0,4):
                if self.table[row][col] == 0:
                    zeroCoord.append([row, col])
        if len(zeroCoord) > 0:
            ran = random.randrange(0, len(zeroCoord))
            coord = zeroCoord[ran]
            if isFirst:
                self.table[coord[0]][coord[1]] = 2
            else:
                num = random.randrange(0,11)
                self.table[coord[0]][coord[1]] = 2 if num < 10 else 4
            return True
        return False
    
    def copyTable(self, table: List):
        return [item[:] for item in table]
    

    def sumTable(self, table: List):
        return sum(list(itertools.chain(*table)))
    
    
    def moveTable(self, dir: Direction):
        mergeCount = 0
        before = self.copyTable(self.table)
        if Direction.UP == dir:
            self.rotateTableCounterClockWise()
        elif Direction.DOWN == dir:
            self.rotateTableClockWise()
        elif Direction.RIGHT == dir:
            self.rotate180Degree()
        orderSet = []
        for row in range(0,4):
            for col in range(1,4):
                orderSet.append([row, col])        
        for xy in orderSet:
            x = xy[0]
            y = xy[1]
            t = y
            for comp in range(1,y+1):
                compValue = self.table[x][y-comp]
                value = self.table[x][t]
                isMove = False
                if value <= 0:
                    continue
                if compValue == 0:
                    self.table[x][y-comp] = self.table[x][t]
                    self.table[x][t] = 0
                    t = y-comp
                    isMove = True
                    continue
                if compValue == value:
                    mergeCount = mergeCount + 1
                    self.table[x][y-comp] = self.table[x][y-comp] * 2
                    self.table[x][t] = 0
                    t = y-comp
                    isMove = True
                    break
                if not isMove:
                    break
        if Direction.UP == dir:
            self.rotateTableClockWise()
        elif Direction.DOWN == dir:
            self.rotateTableCounterClockWise()
        elif Direction.RIGHT == dir:
            self.rotate180Degree()
        # print(dir)
        # print(before)
        # print(self.table)
        # print(str(str(before) != str(self.table)))
        return [str(before) != str(self.table), mergeCount]
        
    def getRotateTableCounterClockWise(self, table):
        result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for row in range(0,4):
            for col in range(0,4):
                result[row][col]=table[col][3-row]
        return result

    def rotateTableClockWise(self):
        result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for row in range(0,4):
            for col in range(0,4):
                result[row][col]=self.table[3-col][row]
        self.table = result

    def rotateTableCounterClockWise(self):
        result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for row in range(0,4):
            for col in range(0,4):
                result[row][col]=self.table[col][3-row]
        self.table = result
    
    def rotate180Degree(self):
        result = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for row in range(0,4):
            for col in range(0,4):
                result[row][col]=self.table[3-row][3-col]
        self.table = result


    

        
    
    