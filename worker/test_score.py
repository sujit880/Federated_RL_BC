# Import Libraries
import math
import datetime
import random
from time import sleep

import numpy as np
# import matplotlib.pyplot as plt
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
ENV_NAME = 'LunarLander-v2'
# API endpoint
URL = "http://localhost:3001/api/model/"


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
PIE_PARAMS.LAYERS = [80, 68]
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
TEST_PARAMS.CERF = 10
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
# Testing
##############################################
P('#', 'Val[1]')
def Test_Params(params, client_key):
    pie.Q.load_state_dict(modman.convert_list_to_tensor(params))
    pie.Q.eval()
    results = []
    for ce in range(TEST_PARAMS.CERF):
        cs = venv.reset()  # <--- validation environment
        R = 0
        A = []
        ts, done = 0, False
        while not done and ts < venv._max_episode_steps:
            ts += 1
            a = pie.predict(cs)
            cs, r, done, _ = venv.step(a)
            R += r
            A.append(a)

        results.append(R)

        P('Reward:', R)

    effs = np.array(results)
    P(client_key, ': Mean Reward:', np.mean(effs))
    P('MAX Reward:', np.max(effs))
    P('MIN Reward:', np.min(effs))
    return client_key, np.mean(effs);
    # fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    # ax[0].plot(effs, color='green')
    # ax[0].set_ylabel('Reward')
    # ax[1].hist(effs, range=(0, 200.1), bins=25, color='green')
    # ax[1].set_xlabel('Reward Distribution')
    # plt.savefig(f'./logs/{ALIAS}.png')
    # F(fig, 'val_CERF')
    # plt.close()