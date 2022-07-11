from service.RunAiService import RunAiService
from service.RandomAiService import RandomAiService
from service.MCTSAiService import MCTSAiService
from service.MCTSRunAiService import MCTSRunAiService
from service.TrainAiService import TrainAiService
import sys
import gc
from multiprocessing import Lock, Process, freeze_support
import time
import signal

sys.setrecursionlimit(10000)
def work(lock):
    ai = TrainAiService(lock)
    ai.run()


def main():
    start = int(time.time())
    procs = []
    lock = Lock()
    for num in range(1):
        proc = Process(target=work, args=(lock,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    print("***run time(sec) :", int(time.time()) - start)

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
