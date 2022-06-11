
from dotenv import dotenv_values
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from bson.binary import Binary
import pickle
import datetime

config = dotenv_values('.env')

class MongoDataSource(object):
    client: MongoClient
    def __init__(self) -> None:
        host = config["mongodbHost"]
        port = config["mongodbPort"]
        userName = config["mongodbUserName"]
        password = config["mongodbPassword"]
        path = f'mongodb://{userName}:{password}@{host}:{port}'
        self.client = MongoClient(path)
        self.nodes: Collection = self.client["game2048"]["nodes"]
        self.nodes.create_index([("key", ASCENDING)], unique=True, name="nodeIndex")
        self.gameinfo: Collection = self.client["game2048"]["gameinfo"]

        self.samples: Collection = self.client["game2048"]["samples"]
        self.samples.create_index([("key", ASCENDING)], unique=True, name="sampleIndex")
        self.samples.create_index([("createdAt", DESCENDING)], unique=False, name="sampleIndexDate")

        self.weights: Collection = self.client["game2048"]["weights"]
        self.weights.create_index([("createdAt", DESCENDING)], unique=True, name="weightIndex")
            
    
    def updateNodes(self, node: "TableNode"):
        data = pickle.dumps(node)
        self.nodes.update_one({
            "key": str(node.table),
        }, {
            "$set": {
                "key": str(node.table),
                "data": Binary(data)
            },
        }, upsert=True)
    
    def getNode(self, table: str):
        cursor = self.nodes.find_one({"key": table})
        if cursor is not None:
            cursor["data"] = pickle.loads(cursor["data"])
            return cursor["data"]
        return None
    
    def addGameInfo(self, turn, score, averageMaxQ, serviceName, actionDicts, gameCount):
        self.gameinfo.insert_one({
            "turn": turn,
            "score": score,
            "maxQ": averageMaxQ,
            "serviceName": serviceName,
            "createdAt": datetime.datetime.now(),
            "left": int(actionDicts[0]) if 0 in actionDicts else 0,
            "right": int(actionDicts[1]) if 1 in actionDicts else 0,
            "up": int(actionDicts[2]) if 2 in actionDicts else 0,
            "down": int(actionDicts[3]) if 3 in actionDicts else 0,
            "gameCount": gameCount
        })
    
    def getGameInfo(self, serviceName):
        cursor = self.gameinfo.find({
            "serviceName": serviceName
        })
        return list(cursor)
    
    
    def updateSamples(self, sample):
        data = pickle.dumps(sample)
        self.samples.update_one({
            "key": str(sample),
        }, {
            "$set": {
                "key": str(sample),
                "data": Binary(data),
                "createdAt": datetime.datetime.now()
            },
        }, upsert=True)
        
    
    def getSamplesAfter(self, date):
        cursor = self.samples.find({ \
            "createdAt": {
                "$gte": date
            }
        })
        return list(map(lambda d : pickle.loads(d["data"]), list(cursor)))
    
    def getSamplesBefore(self, date):
        cursor = self.samples.find({ \
            "createdAt": {
                "$lte": date
            }
        })
        return list(map(lambda d : pickle.loads(d["data"]), list(cursor)))
    
    def getSamplesBetween(self, startDate, endDate):
        cursor = self.samples.find({ \
            "createdAt": {
                "$gte": startDate,
                "$lte": endDate
            }
        })
        return list(map(lambda d : pickle.loads(d["data"]), list(cursor)))
    
    def saveWeight(self, weight, loss):
        data = pickle.dumps(weight)
        self.weights.insert_one({
            "data": Binary(data),
            "createdAt": datetime.datetime.now(),
            "loss": loss
        })
    
    def getLastWeight(self):
        cursor = self.weights.find()
        if len(list(cursor)) > 0:
            limit = self.weights.find().sort([('createdAt', -1)]).limit(1)
            return pickle.loads(list(limit)[0]["data"])
        return None





