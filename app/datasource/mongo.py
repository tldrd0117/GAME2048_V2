
from dotenv import dotenv_values
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from bson.binary import Binary
import pickle
import datetime
import hashlib
import random
import time
import gridfs

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

        self.gridFs = self.client["gridFs"]
        self.weightsFs = gridfs.GridFS(self.gridFs, "weightsFs")

        self.nodes: Collection = self.client["game2048"]["nodes"]
        self.nodes.create_index([("key", ASCENDING)], unique=True, name="nodeIndex")
        self.gameinfo: Collection = self.client["game2048"]["gameinfo"]

        self.samples: Collection = self.client["game2048"]["samples"]
        self.samples.create_index([("key", ASCENDING)], unique=True, name="sampleIndex")
        self.samples.create_index([("createdAt", DESCENDING)], unique=False, name="sampleIndexDate")
        self.samples.create_index([("createdAt", DESCENDING), ("reward", ASCENDING), ("action", ASCENDING)], unique=False, name="sampleIndexDateRewardAction")

        self.weights: Collection = self.client["game2048"]["weights"]
        self.weights.create_index([("createdAt", DESCENDING)], unique=True, name="weightIndex")

        self.episodes: Collection = self.client["game2048"]["episodes"]
        self.episodes.create_index([("createdAt", DESCENDING)], unique=False, name="episodeIndexDate")
            
    
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
        key = str(sample).encode('utf-8')
        md5 = hashlib.new("md5")
        md5.update(key)
        key = md5.hexdigest()
        self.samples.update_one({
            "key": key,
        }, {
            "$set": {
                "key": key,
                "history": pickle.dumps(sample[0]),
                "action": sample[1],
                "reward": sample[2],
                "nextHistory": pickle.dumps(sample[3]),
                "nextHistoryCounterClockwise": pickle.dumps(sample[4]),
                "createdAt": datetime.datetime.now()
            },
        }, upsert=True)
        
    
    def getSamplesAfter(self, date):
        cursor = self.samples.find({ \
            "createdAt": {
                "$gte": date
            }
        })
        return list(map(lambda d : (pickle.loads(d["history"]), d["action"], d["reward"], pickle.loads(d["nextHistory"]), pickle.loads(d["nextHistoryCounterClockwise"])), list(cursor)))
    
    def getSamplesBefore(self, date):
        cursor = self.samples.find({ \
            "createdAt": {
                "$lte": date
            }
        })
        return list(map(lambda d : (pickle.loads(d["history"]), d["action"], d["reward"], pickle.loads(d["nextHistory"]), pickle.loads(d["nextHistoryCounterClockwise"])), list(cursor)))
    
    def getSamplesBetween(self, startDate, endDate):
        cursor = self.samples.find({ \
            "createdAt": {
                "$gte": startDate,
                "$lte": endDate
            }
        })
        return list(map(lambda d : (pickle.loads(d["history"]), d["action"], d["reward"], pickle.loads(d["nextHistory"]), pickle.loads(d["nextHistoryCounterClockwise"])), list(cursor)))
    
    def getSamplesRandom(self, startDate, size):
        cursor = self.samples.aggregate([
            {"$match": { "createdAt": { "$lte": startDate }}},
            {"$sample": { "size": size } }
        ], allowDiskUse=True)
    
    def getSamplesRandomByAction(self, startDate, action, size):
        cursor = self.samples.aggregate([
            {"$match": { "action": action ,"createdAt": { "$lte": startDate }}},
            {"$sample": { "size": size } }
        ], allowDiskUse=True)
        
        return list(map(lambda d : (pickle.loads(d["history"]), d["action"], d["reward"], pickle.loads(d["nextHistory"]), pickle.loads(d["nextHistoryCounterClockwise"])), list(cursor)))
    
    
    def getSamplesRandomByActionAndReward(self, startDate, action, reward, size):
        start = int(time.time())
        print("***find start")
        cursor = self.samples.find({
            "action": action,
            "reward": reward,
            "createdAt": { "$lte": startDate }
        })
        sample = random.sample(list(cursor), size)
        print("***getSamplesRandomByActionAndReward time(sec) :", int(time.time()) - start)
        # cursor = self.samples.aggregate([
        #     {"$match": { "action": action, "reward": reward ,"createdAt": { "$lte": startDate }}},
        #     {"$sample": { "size": size } }
        # ], allowDiskUse=True)
        
        return list(map(lambda d : (pickle.loads(d["history"]), d["action"], d["reward"], pickle.loads(d["nextHistory"]), pickle.loads(d["nextHistoryCounterClockwise"])), sample))
    
    
    def getSamplesRandomByActionAndReward(self, startDate, action, reward, size):
        start = int(time.time())
        print("***find start")
        cursor = self.samples.find({
            "action": action, 
            "reward": reward ,
            "createdAt": { "$lte": startDate }
        })
        sample = random.sample(list(cursor), size)
        print("***getSamplesRandomByActionAndReward time(sec) :", int(time.time()) - start)
        # cursor = self.samples.aggregate([
        #     {"$match": { "action": action, "reward": reward ,"createdAt": { "$lte": startDate }}},
        #     {"$sample": { "size": size } }
        # ], allowDiskUse=True)
        
        return list(map(lambda d : (pickle.loads(d["history"]), d["action"], d["reward"], pickle.loads(d["nextHistory"])), sample))
    
    def saveWeight(self, name, weight, loss):
        data = pickle.dumps(weight)
        id = self.weightsFs.put(data)
        self.weights.insert_one({
            "name": name,
            "dataId": id,
            "createdAt": datetime.datetime.now(),
            "loss": loss
        })
    
    def getLastWeight(self):
        cursor = self.weights.find()
        if len(list(cursor)) > 0:
            c = self.weights.find().sort([('createdAt', -1)]).limit(1)
            fileId = list(c)[0]["dataId"]
            return pickle.loads(self.weightsFs.get(fileId).read())
        return None
    
    def getLastWeightByName(self, modelName):
        cursor = self.weights.find()
        if len(list(cursor)) > 0:
            c = self.weights.find({
                "name": modelName
            }).sort([('createdAt', -1)]).limit(1)
            fileId = list(c)[0]["dataId"]
            return pickle.loads(self.weightsFs.get(fileId).read())
        return None
    
    def getLosses(self):
        return self.weights.find({},{"createdAt":1, "loss":1, "name":1})
    
    def updateEpisodeData(self, action_probs, values, rewards, turn, score):
        self.episodes.insert_one({
            "action_probs": pickle.dumps(action_probs),
            "values": pickle.dumps(values),
            "rewards": pickle.dumps(rewards),
            "turn": turn,
            "score": score,
            "createdAt": datetime.datetime.now()
        })
    
    def getEpisodesBetween(self, startDate, endDate):
        cursor = self.episodes.find({ \
            "createdAt": {
                "$gte": startDate,
                "$lte": endDate
            }
        })
        return list(map(lambda d : (pickle.loads(d["action_probs"]), pickle.loads(d["values"]), pickle.loads(d["rewards"]), d["turn"], d["score"]), list(cursor)))
  
