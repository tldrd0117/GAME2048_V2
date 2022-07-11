
from service.TrainMultiAiService import TrainMultiAiService
from repo.tensor_multi_db import TensorMultitModelDbRepository
from datasource.mongo import MongoDataSource
import datetime

mongo = MongoDataSource()
samples = mongo.getSamplesRandomByAction(datetime.datetime.now(), 1, 50000)
print(samples)
print(len(samples))
# poetry run python app/test.py