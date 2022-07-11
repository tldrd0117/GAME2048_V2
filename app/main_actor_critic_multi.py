from statistics import mean
from service.ActorCriticTrainService import ActorCriticTrainService
from service.ActorCriticService import ActorCriticService
import sys
import multiprocessing as mp

sys.setrecursionlimit(10000)
from multiprocessing import Lock, Process, freeze_support, Queue, Manager, Pool, TimeoutError

def runEpisode(episodeCount):
    service = ActorCriticService(episodeCount)
    startDate, endDate = service.run()
    return startDate, endDate

def train(dates):
    trainService = ActorCriticTrainService()
    trainService.run(dates)

if __name__=='__main__':
    loopCount = 1000
    episodeCount = 10
    processCount = 3
    mp.set_start_method('spawn')

    for i in range(loopCount):
        with Pool(processes=processCount) as p:
            result = p.map_async(runEpisode, [episodeCount]*processCount)
            try:
                data = result.get(timeout=4800)
                print(len(data))
            except TimeoutError:
                p.terminate()
                p.join()
                continue
            p.map(train, [data])
# poetry run python app/main_actor_critic_multi.py
