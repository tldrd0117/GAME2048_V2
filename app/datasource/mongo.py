
from dotenv import dotenv_values
from pymongo import MongoClient, ASCENDING
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
    
    def addGameInfo(self, turn, score, averageMaxQ, serviceName, actionDicts):
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
        })
