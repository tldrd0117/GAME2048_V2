import os

class GameRepository(object):
    turn = 0
    scroe = 0
    def __init__(self) -> None:
        self.initGame()

    def initGame(self):
        self.turn = 0
        self.score = 0
    
    def nextTurn(self, score: int):
        self.turn = self.turn + 1
        self.score = self.score + score
    
    def printGame(self):
        print(f"pid: {str(os.getpid())}")
        print(f"turn : {self.turn}")
        print(f"score : {self.score}")
        