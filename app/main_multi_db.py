from collections import deque
from functools import reduce
import multiprocessing

from service.RunAiService import RunAiService
from service.RandomAiService import RandomAiService
from service.MCTSAiService import MCTSAiService
from service.MCTSRunAiService import MCTSRunAiService
from service.TrainAiService import TrainAiService
from service.TrainSaveDbMultiAiService import TrainSaveDbMultiAiService
from repo.tensor_multi_db import TensorMultitModelDbRepository
import sys
import gc
from multiprocessing import Lock, Process, freeze_support, Queue, Manager, Pool
import time
import tensorflow as tf
import tracemalloc
import datetime

sys.setrecursionlimit(10000)

def work(d):
    ai = TrainSaveDbMultiAiService()
    return ai.run()

def train(dates):
    print("train")
    startDate = dates[0]
    endDate = dates[1]

    tensor = TensorMultitModelDbRepository()
    tensor.loadModel()
    tensor.updateMemoryFromDb(startDate, endDate)
    
    trainCount = int(len(tensor.memory) / (tensor.batch_size * 2) )
    print(len(tensor.memory))
    for i in range(trainCount):
        print(f"train: {i}")
        tensor.trainModel()
    losses = tensor.losses
    avgLoss = str(sum(losses)/len(losses)) if len(losses) > 0 else 0
    if len(losses) > 0:
        print(f"loss: {avgLoss}")
    tensor.saveModel("multi_db_init", avgLoss)

def train_old(dates):
    print("train old")
    startDate = dates[0]
    endDate = dates[1]

    tensor = TensorMultitModelDbRepository()
    tensor.loadModel()
    tensor.updateMemoryFromDbRandom(startDate, 500000)
    if len(tensor.memory) < 400000:
        return
    
    trainCount = int(len(tensor.memory) / (tensor.batch_size * 2) )
    print(len(tensor.memory))
    for i in range(trainCount):
        print(f"train: {i}")
        tensor.trainModel()
    losses = tensor.losses
    avgLoss = str(sum(losses)/len(losses)) if len(losses) > 0 else 0
    if len(losses) > 0:
        print(f"loss: {avgLoss}")
    tensor.saveModel("multi_db_init_old", avgLoss)

def calDate(dateList):
    if len(dateList) <= 0:
        return
    dateList = reduce(lambda x,y: x+y, dateList)
    return [min(dateList), max(dateList)]

if __name__=='__main__':
    # freeze_support()
    start = int(time.time())
    procs = []
    processCount = 3
    weight = None
    for _ in range(1000):
        with Pool(processes=processCount) as p:
            result = p.map_async(work, [None]*processCount)
            try:
                dates = result.get(timeout=4800)
            except multiprocessing.TimeoutError:
                p.terminate()
                p.join()
                continue
            print(len(dates))
            result = deque([])
            d = calDate(dates)
            print(d)
            p.map(train, [d])
            for _ in range(1):
                p.map(train_old, [d])
            result = None
            dates = None
        
    print("***run time(sec) :", int(time.time()) - start)
# poetry run python app/main_multi_db.py
