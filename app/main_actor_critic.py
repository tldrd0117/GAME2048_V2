from service.ActorCriticTrainService import ActorCriticTrainService
from service.ActorCriticService import ActorCriticService


for i in range(100):
    service = ActorCriticService(10)
    grads, losses = service.run()

    trainService = ActorCriticTrainService(grads, losses)
    trainService.run()