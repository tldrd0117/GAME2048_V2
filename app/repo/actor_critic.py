from genericpath import exists
from statistics import mode
from typing import Deque, Dict, List, Tuple
from pexpect import ExceptionPexpect
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Dropout, Add, concatenate
from keras.regularizers import l2
import random
import pickle
from collections import deque
import os
from tensorflow.python.framework.ops import disable_eager_execution
from filelock import FileLock
from repo.table import TableRepository
from datasource.mongo import MongoDataSource
import time

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(
        self):
        """Initialize."""
        super().__init__()
        self.state_size = (64,4,1,)
        actor, critic = self.buildModel()
        self.actor = actor
        self.critic = critic

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.actor([inputs]*9), self.critic([inputs]*9)

    def buildModel(self):
        model1 = Sequential()
        model1.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model1.add(Conv2D(128, (4, 4), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model1.add(Flatten())

        model2 = Sequential()
        model2.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model2.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model2.add(Flatten())

        model3 = Sequential()
        model3.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model3.add(Conv2D(128, (2, 2), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model3.add(Flatten())

        model4 = Sequential()
        model4.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model4.add(Conv2D(128, (2, 1), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model4.add(Flatten())

        model5 = Sequential()
        model5.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model5.add(Conv2D(128, (1, 2), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model5.add(Flatten())

        model6 = Sequential()
        model6.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model6.add(Conv2D(128, (3, 1), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model6.add(Flatten())

        model7 = Sequential()
        model7.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model7.add(Conv2D(128, (1, 3), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model7.add(Flatten())

        model8 = Sequential()
        model8.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model8.add(Conv2D(128, (1, 4), padding='valid', strides=(1, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model8.add(Flatten())

        model9 = Sequential()
        model9.add(Conv2D(64, (16, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() , input_shape=self.state_size))
        model9.add(Conv2D(128, (4, 1), padding='valid', strides=(16, 1), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeNormal() ))
        model9.add(Flatten())

        modelConcat = concatenate([model1.output, model2.output, model3.output, model4.output, model5.output, model6.output, model7.output, model8.output, model9.output])

        modelConcat = Flatten()(modelConcat)
        modelConcat = Dense(2048, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeUniform())(modelConcat)
        actorOut = Dense(4, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeUniform())(modelConcat)
        criticOut = Dense(1, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.HeUniform())(modelConcat)

        actor = Model(inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input, model7.input, model8.input, model9.input], outputs=[actorOut])
        critic = Model(inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input, model7.input, model8.input, model9.input], outputs=[criticOut])
        return actor, critic

class TableNode:
    table: List
    action: int = -1
    parent: str = None
    score: float = 0
    def print(self):
        print("table")
        for row in range(0,4):
            print(self.table[row])
        print(f"action: {str(self.action)}")
        print(f"parent: {self.parent}")

class ActorCriticRepository(object):

    def __init__(self) -> None:
        self.model = ActorCritic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        self.db = MongoDataSource()
        self.episodes = []
        self.eps = np.finfo(np.float32).eps.item()
        
    
    def get_expected_return(
        self,
        rewards: tf.Tensor, 
        gamma: float, 
        standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                    (tf.math.reduce_std(returns) + self.eps))

        return returns
    

    def updateEpisode(self, action_probs, values, rewards, turn, score):
        self.db.updateEpisodeData(action_probs, values, rewards, turn, score)
    
    def getEpisodes(self, startDate, endDate):
        self.episodes = self.db.getEpisodesBetween(startDate, endDate)

    def compute_loss(
        self,
        action_probs: tf.Tensor,  
        values: tf.Tensor,  
        returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss
    

    def listToTensorArray(self, li, tensorArray: tf.TensorArray):
        for i, item in enumerate(li):
            tensorArray.write(i, tf.constant(item))
        return tensorArray.stack()
    

    def getGradient(self, gamma, e, tape: tf.GradientTape):
        avgLoss = []

        action_probs, values, rewards, turn, score = e
        # action_probs = self.listToTensorArray(action_probs, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))
        # values = self.listToTensorArray(values, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))
        # rewards = self.listToTensorArray(rewards, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))
        # Run the model for one episode to collect training data
        # rewards = np.array(rewards, np.float32)
        # action_probs = self.listToTensorArray(action_probs, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)).stack()
        # values = self.listToTensorArray(values, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)).stack()
        # Calculate expected returns
        returns = self.get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        # action_probs, values, returns = [
        #     tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

        # Calculating loss values to update our network
        loss = self.compute_loss(action_probs, values, returns)
        print(f"loss: {loss}")
        avgLoss.append(loss)

        # Compute the gradients from the loss
        return loss
    
    def applyGrads(self, gradList):
        for grads in gradList:
            print("apply_gradients")
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def train_step(
        self,
        gamma: float, 
        ) -> tf.Tensor:
        """Runs a model training step."""

        avgLoss = []

        for e in self.episodes:
            action_probs, values, rewards, turn, score = e
            action_probs = self.listToTensorArray(action_probs, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))
            values = self.listToTensorArray(values, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))
            rewards = self.listToTensorArray(rewards, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True))
            with tf.GradientTape() as tape:
                # Run the model for one episode to collect training data
                # rewards = np.array(rewards, np.float32)
                print(action_probs)
                print(values)
                print(rewards)
                # action_probs = self.listToTensorArray(action_probs, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)).stack()
                # values = self.listToTensorArray(values, tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)).stack()
                # Calculate expected returns
                returns = self.get_expected_return(rewards, gamma)
                print("returns")
                print(returns)

                # Convert training data to appropriate TF tensor shapes
                # action_probs, values, returns = [
                #     tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

                # Calculating loss values to update our network
                loss = self.compute_loss(action_probs, values, returns)
                print(f"loss: {loss}")
                avgLoss.append(loss)

            # Compute the gradients from the loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            # Apply the gradients to the model's parameters
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # episode_reward = tf.math.reduce_sum(rewards)
        
        return sum(avgLoss) / len(avgLoss) if len(avgLoss) > 0 else 0
    
    def loadModel(self):
        actorWeight = self.db.getLastWeightByName("actorCriticActor")
        criticWeight = self.db.getLastWeightByName("actorCriticCritic")
        if actorWeight is not None:
            print("actor load Complete")
            self.model.actor.set_weights(actorWeight)
        if criticWeight is not None:
            print("critic load Complete")
            self.model.critic.set_weights(criticWeight)
    

    def saveModel(self, loss):
        print("saveModel")
        self.db.saveWeight("actorCriticActor", self.model.actor.get_weights(), loss)
        self.db.saveWeight("actorCriticCritic", self.model.critic.get_weights(), loss)
        print("saveModel Complete")
        
    