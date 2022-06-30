from statistics import mean
from service.ActorCriticTrainService import ActorCriticTrainService
from service.ActorCriticService import ActorCriticService
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Lock, Process, freeze_support, Queue, Manager, Pool, TimeoutError

def runEpisode(episodeCount):
    service = ActorCriticService(episodeCount)
    grads, losses = service.run()
    return grads, losses

def train(data):
    trainService = ActorCriticTrainService()
    trainService.loadModel()
    totalLosses = []
    for each in data:
        grads, losses = each
        totalLosses = totalLosses + losses
        trainService.setTrainData(grads, losses)
        trainService.applyGrads()
    trainService.saveModel(mean(totalLosses))
    

if __name__=='__main__':
    loopCount = 100
    episodeCount = 5
    processCount = 3

    for i in range(loopCount):
        with Pool(processes=processCount) as p:
            result = p.map_async(runEpisode, [episodeCount]*processCount)
            try:
                data = result.get(timeout=4800)
            except TimeoutError:
                p.terminate()
                p.join()
                continue
            p.map(train, [data])
# poetry run python app/main_actor_critic_multi.py
