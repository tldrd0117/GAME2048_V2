from service.RunAiService import RunAiService
from service.RandomAiService import RandomAiService
from service.MCTSAiService import MCTSAiService
from service.MCTSRunAiService import MCTSRunAiService
from service.TrainAiService import TrainAiService
import sys
import gc
from multiprocessing import Process, freeze_support
import time

sys.setrecursionlimit(10000)
def work():
    ai = TrainAiService()
    ai.run()

def main():
    work()
    # start = int(time.time())
    # procs = []  
    # for num in range(8):
    #     proc = Process(target=work, args=())
    #     procs.append(proc)
    #     proc.start()

    # for proc in procs:
    #     proc.join()
    # print("***run time(sec) :", int(time.time()) - start)

if __name__=='__main__':
    freeze_support()
    main()
# for _ in range(10):
#     ai = TrainAiService()
#     ai.run()
# ai = RunAiService()
# ai.run()
# print("main")
# poetry run python app/main.py
