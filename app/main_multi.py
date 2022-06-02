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

sys.setrecursionlimit(10000)
def work(weight):
    ai = TrainMultiAiService(weight)
    return ai.run()

def main():
    start = int(time.time())
    procs = []
    processCount = 1
    tensor = TensorMultitModelRepository()
    tensor.loadModel()
    weight = tensor.model.get_weights()
    for _ in range(10):
        with Pool(processes=processCount) as p:
            memories = p.map(work, [weight]*processCount)
            print(len(memories))
            result = deque([])
            for memory in memories:
                result = result + memory
            tensor.updateMemory(result)
        trainCount = int(len(tensor.memory) / (tensor.batch_size * 2) )
        print(len(tensor.memory))
        for i in range(trainCount):
            print(f"train: {i}")
            tensor.trainModel()
        losses = tensor.losses
        print(f"loss: {str(sum(losses)/len(losses))}")
        weight = tensor.model.get_weights()
        tensor.saveModel()
    print("***run time(sec) :", int(time.time()) - start)

if __name__=='__main__':
    freeze_support()
    main()
# poetry run python app/main_multi.py
