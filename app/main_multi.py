from collections import deque
from service.RunAiService import RunAiService
from service.RandomAiService import RandomAiService
from service.MCTSAiService import MCTSAiService
from service.MCTSRunAiService import MCTSRunAiService
from service.TrainAiService import TrainAiService
from service.TrainMultiAiService import TrainMultiAiService
from repo.tensor_multi import TensorMultitModelRepository
import sys
import gc
from multiprocessing import Lock, Process, freeze_support, Queue, Manager, Pool
import time
import tensorflow as tf
from work import work
import tracemalloc

sys.setrecursionlimit(10000)

def work(weight):
    ai = TrainMultiAiService(weight)
    return ai.run()

def train(memory):
    tensor = TensorMultitModelRepository()
    tensor.loadModel()
    tensor.updateMemory(memory)
    
    trainCount = int(len(tensor.memory) / (tensor.batch_size * 2) )
    print(len(tensor.memory))
    for i in range(trainCount):
        print(f"train: {i}")
        tensor.trainModel()
    losses = tensor.losses
    print(f"loss: {str(sum(losses)/len(losses))}")
    weight = tensor.model.get_weights()
    tensor.saveModel()
    return weight

def loadModel(d):
    tensor = TensorMultitModelRepository()
    tensor.loadModel()
    weight = tensor.model.get_weights()
    return weight


if __name__=='__main__':
    # freeze_support()
    start = int(time.time())
    procs = []
    processCount = 6
    weight = None
    for _ in range(1000):
        with Pool(processes=processCount) as p:
            if weight is None:
                weight = p.map(loadModel, [None])[0]
            memories = p.map(work, [weight]*processCount)
            print(len(memories))
            result = deque([])
            for memory in memories:
                result = result + memory
            weight = p.map(train, [result])[0]
            result = None
            memories = None
        
    print("***run time(sec) :", int(time.time()) - start)
# poetry run python app/main_multi.py
