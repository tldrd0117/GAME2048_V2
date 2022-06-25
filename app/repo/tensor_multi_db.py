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
from repo.table import TableRepository
from datasource.mongo import MongoDataSource
import time

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
    memory: List = []
    def __init__(self) -> None:
        self.db = MongoDataSource()
        self.tableRepo = TableRepository()
        disable_eager_execution()
        self.state_size = (64,4,1,)
        self.action_size = 2
        self.batch_size = 2048
        self.discount_factor = 0.99
        self.memorySize = 0

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
        config.gpu_options.per_process_gpu_memory_fraction = 0.125
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
    
    def updateMemoryFromDbRandom(self, startDate, size):
        self.memory = self.db.getSamplesRandom(startDate, size)
    
    def updateMemoryFromDbRandomByAction(self, startDate, action, size):
        self.memory = self.db.getSamplesRandomByAction(startDate, action, size)
    
    def updateSamplesRandomByActionAndReward(self, startDate, action, reward, size):
        self.memory = self.memory + self.db.getSamplesRandomByActionAndReward(startDate, action, reward, size)
    
    def getLosses(self):
        return self.db.getLosses()
    
    def updateTargetModel(self, weight):
        self.model.set_weights(weight)
        self.targetModel.set_weights(weight)
        print(self.model.get_weights())

    def buildModel(self):
        model1 = Sequential()
        model1.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model1.add(Conv2D(64, (4, 4), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model1.add(Flatten())

        model2 = Sequential()
        model2.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model2.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model2.add(Flatten())

        model3 = Sequential()
        model3.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model3.add(Conv2D(64, (2, 2), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model3.add(Flatten())

        model4 = Sequential()
        model4.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model4.add(Conv2D(64, (2, 1), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model4.add(Flatten())

        model5 = Sequential()
        model5.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model5.add(Conv2D(64, (1, 2), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model5.add(Flatten())

        model6 = Sequential()
        model6.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model6.add(Conv2D(64, (3, 1), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model6.add(Flatten())

        model7 = Sequential()
        model7.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model7.add(Conv2D(64, (1, 3), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model7.add(Flatten())

        model8 = Sequential()
        model8.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model8.add(Conv2D(64, (1, 4), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model8.add(Flatten())

        model9 = Sequential()
        model9.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model9.add(Conv2D(64, (4, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model9.add(Flatten())

        modelConcat = concatenate([model1.output, model2.output, model3.output, model4.output, model5.output, model6.output, model7.output, model8.output, model9.output])

        modelConcat = Flatten()(modelConcat)
        modelConcat = Dense(1024, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeUniform())(modelConcat)
        modelConcat = Dense(128, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeUniform())(modelConcat)
        modelConcat = Dense(2, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeUniform())(modelConcat)

        model = Model(inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input, model7.input, model8.input, model9.input], outputs=[modelConcat])
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
        hist = np.array(histTable).flatten().reshape(1,64,4,1)
        q_value = self.model.predict([hist]*9)
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
            history = np.array(historyTable).flatten().reshape(64,4,1)
            action = tableNode.action
            reward = tableNode.score
            if len(tableNode.table) > 0:
                nextHisotryTable = self.convertTable(tableNode.table)
                nextHistoryCounterClockwiseTable = self.convertTable(self.tableRepo.getRotateTableCounterClockWise(tableNode.table))
                nextHistory = np.array(nextHisotryTable).flatten().reshape(64,4,1)
                nextHistoryCounterClockwise = np.array(nextHistoryCounterClockwiseTable).flatten().reshape(64,4,1)
            else:
                nextHistory = None
                nextHistoryCounterClockwise = None
            self.memorySize = self.memorySize + 1
            self.db.updateSamples((history, int(action), int(reward), nextHistory, nextHistoryCounterClockwise))
            historyTable = None
            nextHisotryTable = None

    def trainModel(self):
        if len(self.memory) < 10000:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, 64,4,1))
        next_history = np.zeros((self.batch_size, 64,4,1))
        nextHistoryClockWise = np.zeros((self.batch_size, 64,4,1))
        target = np.zeros((self.batch_size,))
        action, reward = [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3])
            nextHistoryClockWise[i] = np.float32(mini_batch[i][4])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value1 = self.targetModel.predict([next_history]*9)
        target_value2 = self.targetModel.predict([nextHistoryClockWise]*9)
        target_value = target_value1 + target_value2

        for i in range(self.batch_size):
            if next_history[i] is None and nextHistoryClockWise[i] is None:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                    np.amax(target_value[i])

        loss = self.optimizer([[history]*9, action, target])
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
