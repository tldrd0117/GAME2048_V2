
from service.TrainMultiAiService import TrainMultiAiService

def work(weight):
    ai = TrainMultiAiService(weight)
    return ai.run()