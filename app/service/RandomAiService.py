import random
from repo.table import TableRepository
from repo.game import GameRepository

class RandomAiService(object):
    def __init__(self) -> None:
        self.tableRepo = TableRepository()
        self.gameRepo = GameRepository()

    def run(self):
        self.gameRepo.initGame()
        while True:
            dirList = self.tableRepo.getPossibleDirList()
            print(dirList)
            if len(dirList) <= 0:
                break
            idx = random.randrange(0,len(dirList))
            print(f"move: {str(dirList[idx])}")
            self.tableRepo.moveTable(dirList[idx])
            self.gameRepo.nextTurn()
            self.tableRepo.genRandom()
            self.gameRepo.printGame()
            self.tableRepo.printTable()

        print("end")
        self.tableRepo.printTable()

