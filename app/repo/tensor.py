from genericpath import exists
from typing import Deque, Dict, List
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import RMSprop
import random
import pickle
from collections import deque
import os
from tensorflow.python.framework.ops import disable_eager_execution

class TableNode:
    table: List
    action: int = -1
    visit: int = 0
    scores: List = []
    parent: str = None
    rootScore: int = 0
    childs: List[Dict[str,int]] = []
    def print(self):
        print("table")
        for row in range(0,4):
            print(self.table[row])
        print(f"action: {str(self.action)}")
        print(f"visit: {str(self.visit)}")
        print(f"scores: {str(self.scores)}")
        print(f"parent: {self.parent}")
        print(f"childs: {str(self.childs)}")

class TensorModelRepository(object):
    nodes = {}
    predictedQValue = []
    losses = []
    memory: Deque = deque(maxlen=500000)
    def __init__(self) -> None:
        disable_eager_execution()
        self.state_size = (4,4,1,)
        self.action_size = 4
        self.batch_size = 4096
        self.discount_factor = 0.99

        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        self.model = self.buildModel()
        self.targetModel = self.buildModel()
        self.updateTargetModel()
        self.optimizer = self.buildOptimizer()

        # 텐서보드 설정
        self.sess = tf.compat.v1.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.compat.v1.summary.FileWriter(
            'app/data/summary', self.sess.graph)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        if exists("app/data/model/game2048_dqn.h5"):
            self.model.load_weights("app/data/model/game2048_dqn.h5")
            self.updateTargetModel()
            print(self.model.get_weights())
    

    def updateNodes(self, node: TableNode):
        self.nodes[str(node.table)] = node
    
    def getNode(self, table: List):
        if str(table) in self.nodes:
            node = self.nodes[str(table)]
        else:
            node = TableNode()
            node.table = table
            node.childs = []
            node.scores = []
        return node
    
    def getCopyTree(self):
        return {key: value[:] for key, value in self.nodes.items()}

    def getExistNode(self, table: str):
        node = None
        if table in self.nodes:
            node = self.nodes[table]
        return node
    
    def updateTargetModel(self):
        self.targetModel.set_weights(self.model.get_weights())
    

    def saveModel(self):
        self.model.save_weights("app/data/model/game2048_dqn.h5")

    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu'))
        model.add(Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(4))
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

        rms = RMSprop(lr=0.00001, epsilon=0.01)
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
        hist = np.array(history).flatten().reshape(1,4,4,1)
        q_value = self.model.predict(hist)
        self.predictedQValue.append(np.argmax(np.argsort(-q_value)))
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
        return deque(maxlen=400000)
    

    def getReward(self, reward):
        if reward > 100:
            return 100
        else:
            return reward
    

    def appendSamples(self, tableNodes: List[TableNode]):
        for tableNode in tableNodes:
            if tableNode.parent is None:
                continue
            history = np.array(tableNode.parent).flatten().reshape(4,4,1)
            action = tableNode.action
            reward = self.getReward(sum(tableNode.scores) / len(tableNode.scores) - tableNode.rootScore)
            nextHistory = np.array(tableNode.table).flatten().reshape(4,4,1)
            self.memory.append((history, int(action), reward, nextHistory))

    def trainModel(self):
        if len(self.memory) < self.batch_size:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, 4,4,1))
        next_history = np.zeros((self.batch_size, 4,4,1))
        target = np.zeros((self.batch_size,))
        action, reward = [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value = self.targetModel.predict(next_history)

        for i in range(self.batch_size):
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
