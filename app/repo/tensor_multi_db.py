from genericpath import exists
from statistics import mode
from typing import Deque, Dict, List
from pexpect import ExceptionPexpect
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Dropout, Add, concatenate
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import RMSprop
import random
import pickle
from collections import deque
import os
from tensorflow.python.framework.ops import disable_eager_execution
from filelock import FileLock
from datasource.mongo import MongoDataSource

class TableNode:
    table: List
    action: int = -1
    parent: str = None
    score: int = 0
    def print(self):
        print("table")
        for row in range(0,4):
            print(self.table[row])
        print(f"action: {str(self.action)}")
        print(f"parent: {self.parent}")

class TensorMultitModelDbRepository(object):
    nodes = {}
    losses = []
    memory: Deque = deque(maxlen=500000)
    def __init__(self) -> None:
        self.db = MongoDataSource()
        disable_eager_execution()
        self.state_size = (4,4,16,)
        self.action_size = 4
        self.batch_size = 2048
        self.discount_factor = 0.99

        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        self.model = self.buildModel()
        self.targetModel = self.buildModel()
        self.optimizer = self.buildOptimizer()

        # 텐서보드 설정
        graph = tf.compat.v1.get_default_graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # config.gpu_options.allow_growth = True
        # config.log_device_placement = True

        self.sess = tf.compat.v1.Session (config=config, graph= graph)
        K.set_session(self.sess)
        # self.sess = tf.compat.v1.InteractiveSession()
        # K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.compat.v1.summary.FileWriter(
            'app/data/summary', self.sess.graph)
        self.sess.run(tf.compat.v1.global_variables_initializer())
    

    def updateNodes(self, node: TableNode):
        self.nodes[str(node.table)] = node
    
    def resetNodes(self):
        self.nodes = {}
    
    def getNode(self, table: List):
        # if str(table) in self.nodes:
        #     node = self.nodes[str(table)]
        # else:
        node = TableNode()
        node.table = table
        return node
    

    def newNode(self):
        node = TableNode()
        node.table = []
        return node
    
    def getCopyTree(self):
        return {key: value[:] for key, value in self.nodes.items()}

    def getExistNode(self, table: str):
        node = None
        if table in self.nodes:
            node = self.nodes[table]
        return node

    def updateMemory(self, memory):
        self.memory = memory
    
    def updateMemoryFromDb(self, startDate, endDate):
        self.memory = self.db.getSamplesBetween(startDate, endDate)
        print(f"memory length: {str(len(self.memory))}")
    
    def updateTargetModel(self, weight):
        self.model.set_weights(weight)
        self.targetModel.set_weights(weight)
        print(self.model.get_weights())


    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.01) , input_shape=self.state_size))
        model.add(Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation="relu"))
        model.summary()
        return model
    
    def getLoss(self, prediction):
        return 
    
    def buildOptimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        rms = RMSprop(lr=0.0001, epsilon=0.01)
        updates = rms.get_updates(loss, self.model.trainable_weights)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # # 입실론 탐욕 정책으로 행동 선택
    # def get_action(self, history, predict_percent=0.5):
    #     if np.random.rand() < predict_percent:
    #         li = [0,1,2,3]
    #         random.shuffle(li)
    #         return np.array(li), False
    #     else:
    #         q_value = self.model.predict(history)
    #         return np.argsort(-q_value[0]), True
    
    def getActionPredicted(self, history):
        histTable = self.convertTable(history)
        hist = np.array(histTable).flatten().reshape(1,4,4,16)
        q_value = self.model.predict(hist)
        histTable = None
        return q_value

    def saveDeque(self):
        with open('data/memory.h5', 'wb') as fw:
            pickle.dump(self.memory, fw)

    def loadDeque(self):
        for fileName in os.listdir('data'):
            if fileName.startswith('memory.h5'):
                with open('data/memory.h5', 'rb') as fr:
                    data = pickle.load(fr)
                    if data:
                        return data
        return deque(maxlen=500000)
    
    def loadModel(self):
        weight = self.db.getLastWeight()
        if weight is not None:
            self.model.set_weights(weight)
            self.updateTargetModel(weight)
        else:
        # with FileLock("app/data/model/game2048_dqn.h5.lock", timeout=100):
            if exists("app/data/model/game2048_dqn_multi.h5"):
                self.model.load_weights("app/data/model/game2048_dqn_multi.h5")
                self.updateTargetModel(self.model.get_weights())
                self.saveModel("multi_db_init",0)

    def saveModel(self, modelName, loss):
        self.db.saveWeight(modelName, self.model.get_weights(), loss)
        # with FileLock("app/data/model/game2048_dqn.h5.lock", timeout=100):
        # self.model.save_weights("app/data/model/game2048_dqn_multi.h5")

    def getReward(self, reward):
        if reward > 100:
            return 100
        else:
            return reward
    

    def convertTable(self, table):
        tab = [item[:] for item in table]
        if len(tab) <= 0:
            return []
        for i in range(len(tab)):
            for j in range(len(tab[0])):
                tab[i][j] = [int(k) for k in list(bin(tab[i][j])[2:].zfill(16))]
        return tab
    

    def appendSamples(self, tableNodes: List[TableNode]):
        for tableNode in tableNodes:
            if tableNode.parent is None:
                continue
            historyTable = self.convertTable(tableNode.parent)
            history = np.array(historyTable).flatten().reshape(4,4,16)
            action = tableNode.action
            reward = tableNode.score
            if len(tableNode.table) > 0:
                nextHisotryTable = self.convertTable(tableNode.table)
                nextHistory = np.array(nextHisotryTable).flatten().reshape(4,4,16)
            else:
                nextHistory = None
            self.db.updateSamples((history, int(action), reward, nextHistory))
            historyTable = None
            nextHisotryTable = None

    def trainModel(self):
        if len(self.memory) < 10000:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, 4,4,16))
        next_history = np.zeros((self.batch_size, 4,4,16))
        target = np.zeros((self.batch_size,))
        action, reward = [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value = self.targetModel.predict(next_history)

        for i in range(self.batch_size):
            if next_history[i] is None:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                    np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]
        self.losses.append(loss[0])
        return loss

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.compat.v1.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.compat.v1.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
