
from service.TrainMultiAiService import TrainMultiAiService
from repo.tensor_multi_db import TensorMultitModelDbRepository
from datasource.mongo import MongoDataSource
import datetime

mongo = MongoDataSource()
samples = mongo.getSamplesRandom(datetime.datetime.now(),10)
print(samples)
# poetry run python app/test.py