from service.RandomAiService import RandomAiService
from service.MCTSAiService import MCTSAiService
from service.TensorAiService import TensorAiService
import sys
import gc
sys.setrecursionlimit(10000)
# for idx in range(10, 11):
#     gc.collect()
#     ai = MCTSAiService(idx)
#     ai.run()
for _ in range(10):
    ai = TensorAiService()
    ai.run()
print("main")
# poetry run python app/main.py
