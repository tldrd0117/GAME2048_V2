from statistics import mean
from repo.actor_critic import ActorCriticRepository

class ActorCriticTrainService(object):
    # def __init__(self, startDate, endDate) -> None:
    #     self.startDate = startDate
    #     self.endDate = endDate
    #     self.actorCritic = ActorCriticRepository()
    def __init__(self) -> None:
        self.actorCritic = ActorCriticRepository()
    
    def loadModel(self):
        self.actorCritic.loadModel()
    
    def saveModel(self, avgLoss):
        self.actorCritic.saveModel(avgLoss)
    
    def setTrainData(self, gradsList, losses):
        self.gradsList = gradsList
        self.losses = losses
    
    def applyGrads(self):
        self.actorCritic.applyGrads(self.gradsList)
        # self.actorCritic.getEpisodes(self.startDate, self.endDate)
        # avgLoss = self.actorCritic.train_step(0.99)
        # avgLoss = mean(self.losses)
        # print(avgLoss)
        # self.actorCritic.saveModel(avgLoss)
