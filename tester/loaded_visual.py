# Import Libraries
import math
import datetime
import random
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import torch

import relearn.pies.dqn as DQN
from relearn.explore import EXP, MEM
from relearn.pies.utils import compare_weights
from relearn.pies.utils import RMSprop_update

import modman

from queue import Queue
import gym

from copy import deepcopy

debug = True

now = datetime.datetime.now

# Logging CSV String
LOG_CSV = 'epoch,reward,tr,up\n'

##############################################
# SETUP Hyperparameters
##############################################
ALIAS = 'experiment_01'
ENV_NAME = 'CartPole-v0'

# API endpoint
URL = "http://localhost:3000/api/model/"


class INFRA:
    """ Dummy empty class"""

    def __init__(self):
        pass


EXP_PARAMS = INFRA()
EXP_PARAMS.MEM_CAP = 50000
EXP_PARAMS.EPST = (0.95, 0.05, 0.95)  # (start, min, max)
EXP_PARAMS.DECAY_MUL = 0.99999
EXP_PARAMS.DECAY_ADD = 0


PIE_PARAMS = INFRA()
PIE_PARAMS.LAYERS = [128, 128, 128]
PIE_PARAMS.OPTIM = torch.optim.RMSprop  # SGD
PIE_PARAMS.LOSS = torch.nn.MSELoss
PIE_PARAMS.LR = 0.001
PIE_PARAMS.DISCOUNT = 0.999999
PIE_PARAMS.DOUBLE = False
PIE_PARAMS.TUF = 4
PIE_PARAMS.DEV = 'cpu'

TRAIN_PARAMS = INFRA()
TRAIN_PARAMS.EPOCHS = 50000
TRAIN_PARAMS.MOVES = 10
TRAIN_PARAMS.EPISODIC = False
TRAIN_PARAMS.MIN_MEM = 30
TRAIN_PARAMS.LEARN_STEPS = 1
TRAIN_PARAMS.BATCH_SIZE = 50
TRAIN_PARAMS.TEST_FREQ = 10

TEST_PARAMS = INFRA()
TEST_PARAMS.CERF = 100
TEST_PARAMS.RERF = 100


P = print


def F(fig, file_name): return plt.close()  # print('FIGURE ::',file_name)


def T(header, table): return print(header, '\n', table)


P('#', ALIAS)

##############################################
# Setup ENVS
##############################################

# Test ENV
venv = gym.make(ENV_NAME)

# Policy and Exploration
pie = DQN.PIE(
    venv.observation_space.shape[0],
    LL=PIE_PARAMS.LAYERS,
    action_dim=venv.action_space.n,
    device=PIE_PARAMS.DEV,
    opt=PIE_PARAMS.OPTIM,
    cost=PIE_PARAMS.LOSS,
    lr=PIE_PARAMS.LR,
    dis=PIE_PARAMS.DISCOUNT,
    mapper=lambda x: x,
    double=PIE_PARAMS.DOUBLE,
    tuf=PIE_PARAMS.TUF,
    seed=None)


##############################################
# Load Parameters
##############################################
pie.load('model.pt')


##############################################
# Testing
##############################################
obs = venv.reset()
r = 0
print("step:", venv._max_episode_steps)
for i in range(venv._max_episode_steps):
    action = pie.predict(obs)
    obs, rewards, dones, info = venv.step(action)
    if(dones):
        print("#step ", i, "-> ", dones)
        break
    r += rewards
    venv.render()
print("score: ", r)
