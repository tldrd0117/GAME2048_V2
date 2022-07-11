from multiprocessing import Lock
from typing import List

from repo.tensor_multi_db import TensorMultitModelDbRepository
from repo.game import GameRepository
from repo.table import TableRepository, Direction
from repo.tree import TableNode, TreeDbRepository
from repo.actor_critic import ActorCriticRepository

import random
import numpy as np
import sys
import datetime
import time
import traceback
import os
import tensorflow as tf

class ActorCriticService(object):
    def __init__(self, episodeCount = 1) -> None:
        self.tableRepo = TableRepository()
        self.gameRepo = GameRepository()
        self.treeRepo = TreeDbRepository()
        self.actorCritic = ActorCriticRepository()
        self.turns = []
        self.scores = []
        self.simulatedActions = []
        self.values = []
        self.rewards = []
        self.episodeCount = episodeCount
    
    def run(self):
        startDate = datetime.datetime.now()

        self.actorCritic.loadModel()
        for i in range(self.episodeCount):
            self.runEpisode()
        averageScores = sum(self.scores) / len(self.scores) if len(self.scores) > 0 else 0
        averageTurns = sum(self.turns) / len(self.turns) if len(self.turns) > 0 else 0
        averageValues = sum(self.values) / len(self.values) if len(self.values) > 0 else 0
        averageRewards = sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else 0
        unique2, count2 = np.unique(np.array(self.simulatedActions), return_counts=True)
        simulateActions = dict(zip(unique2, count2))
        print(str(averageTurns))
        print(str(averageScores))
        print(str(simulateActions))

        self.treeRepo.addGameInfo(averageTurns, averageScores, averageRewards, averageValues, "ActorCriticService", simulateActions, self.episodeCount)
        
        # return self.grads, self.losses
        return startDate, datetime.datetime.now()
    
    def convertTable(self, table):
        tab = [item[:] for item in table]
        if len(tab) <= 0:
            return []
        for i in range(len(tab)):
            for j in range(len(tab[0])):
                tab[i][j] = [int(k) for k in list(bin(tab[i][j])[2:].zfill(16))]
        return tab
    
    def selection(self, state):
        state2 = self.tableRepo.getRotateTableCounterClockWise(state)

        newTable = self.convertTable(state)
        newTable2 = self.convertTable(state2)

        newTable = np.array(newTable).flatten().reshape(1,64,4,1)
        newTable2 = np.array(newTable2).flatten().reshape(1,64,4,1)

        action_logits_t, value = self.actorCritic.model(newTable)
        action_logits_t2, value2 = self.actorCritic.model(newTable2)

        if value >= value2:
            print(f"1:{str(action_logits_t)}")
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            self.simulatedActions.append(action)
            self.values.append(tf.get_static_value(tf.squeeze(value)))
            return action, action_probs_t, value
        else:
            print(f"2:{str(action_logits_t2)}")
            action = tf.random.categorical(action_logits_t2, 1)[0, 0]
            action_probs_t2 = tf.nn.softmax(action_logits_t2)
            self.simulatedActions.append(action+2)
            self.values.append(tf.get_static_value(tf.squeeze(value2)))
            return action+2, action_probs_t2, value2


    def runEpisode(self):
        tableRepo = TableRepository()
        gameRepo = GameRepository()
        tableRepo.initTable()
        gameRepo.initGame()
        simulateMaxQ = []

        with tf.GradientTape(persistent=True) as tape:
            action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            while True:
                action, action_probs_t, value = self.selection(tableRepo.getCopyTable())
                values = values.write(gameRepo.turn, tf.squeeze(value))
                action_probs = action_probs.write(gameRepo.turn, action_probs_t[0, action if action < 2 else action-2])
                # values.append(tf.get_static_value(tf.squeeze(value)))
                # action_probs.append(tf.get_static_value(action_probs_t[0, action]))
                data = tableRepo.moveTable(action)
                if not data[0]:
                    dirlist = tableRepo.getPossibleDirList()
                    if len(dirlist) > 0:
                    # rewards = rewards.write(gameRepo.turn, -len(dirlist)/(gameRepo.turn+1))
                    # rewards = rewards.write(gameRepo.turn, -len(dirlist)/3)
                        rewards = rewards.write(gameRepo.turn, 0)
                    else:
                        print("################END")
                        rewards = rewards.write(gameRepo.turn, 0)
                    print("end")
                    break
                else:
                    rewards = rewards.write(gameRepo.turn, 1 if data[1]>0 else 0.5)
                    # rewards = rewards.write(gameRepo.turn, (gameRepo.turn+1) * 0.01)
                # rewards.append(data[1])
                gameRepo.nextTurn(data[1])
                tableRepo.genRandom()
            simulateMaxQ.append(tf.squeeze(value))
            self.turns.append(gameRepo.turn)
            self.scores.append(gameRepo.score)
            action_probs = action_probs.stack()
            values = values.stack()
            rewards = rewards.stack()
            print(f"{gameRepo.turn} / {gameRepo.score}" )
            # self.actorCritic.updateEpisode(action_probs, values, rewards, gameRepo.turn, gameRepo.score)
            # gradient를 구해야함
            actor_loss, critic_loss, returns = self.actorCritic.getGradient(0.99, (action_probs, values, rewards, gameRepo.turn, gameRepo.score), tape)
            actorGrads = tape.gradient(actor_loss, self.actorCritic.model.actor.trainable_variables)
            criticGrads = tape.gradient(critic_loss, self.actorCritic.model.critic.trainable_variables)
            
            self.actorCritic.insertGradients("actor", actorGrads, tf.get_static_value(actor_loss).tolist())
            self.actorCritic.insertGradients("critic", criticGrads, tf.get_static_value(critic_loss).tolist())

            self.rewards = self.rewards + tf.get_static_value(returns).tolist()
            # print(rewards)
            # self.grads.append(grads)
            # self.losses.append(tf.get_static_value(loss).tolist())
    