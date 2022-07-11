from datetime import datetime
from statistics import mean
from repo.actor_critic import ActorCriticRepository

class ActorCriticTrainService(object):
    # def __init__(self, startDate, endDate) -> None:
    #     self.startDate = startDate
    #     self.endDate = endDate
    #     self.actorCritic = ActorCriticRepository()
    def __init__(self) -> None:
        self.actorCritic = ActorCriticRepository()
    
    def getMinMaxDates(self, dates):
        minDates = datetime.max
        maxDates = datetime.min
        for date in dates:
            for one in date:
                if one < minDates:
                    minDates = one
                if one > maxDates:
                    maxDates = one
        return minDates, maxDates
    

    def run(self, dates) -> None:
        self.actorCritic.loadModel()
        startDate, endDate = self.getMinMaxDates(dates)
        print(f"startDate:{str(startDate)} endDate:{str(endDate)}")
        li = self.actorCritic.getGradientsBetween(startDate, endDate)
        actorLosses = []
        criticLosses = []
        for item in li:
            name, grads, loss = item
            if name == "actor":
                actorLosses.append(loss)
            elif name == "critic":
                criticLosses.append(loss)
            self.actorCritic.applyGrads(name, grads)
            print(f"applyGrads: {str(loss)} ({name})")
        self.actorCritic.saveModel(mean(actorLosses), mean(criticLosses))

    
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
