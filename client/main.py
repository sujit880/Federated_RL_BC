# Import Libraries
import random
import datetime
from os import getpid
import numpy as np
import matplotlib.pyplot as plt
import torch

# from relearn.pies.ppo import PPO
from sb_wrapper.ppo import PPO
from relearn.utils import test_model

import modman

from queue import Queue
import gym
import os
from time import sleep
import socket

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)


now = datetime.datetime.now

##############################################
# SETUP Hyperparameters
##############################################
ALIAS = 'ppo_sb__100_3_1'
ENV_NAME = 'CartPole-v0'


##############################################
# SETUP Seeds
##############################################
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# For test locally -> ..
# API endpoint
# URL = "http://localhost:5500/api/model/"  # Un comment this line if you wanna test locally
# ..

# For test in the server and sepertade clients ...

ip_address = "172.16.26.15"  # server macine ip address
# API endpoint
# URL = "http://"+ip_address+":5500/api/model/"
# URL = "http://localhost:5500/api/model/"

URL = "http://localhost:3000/api/model/"

# ..


class INFRA:
    """ Dummy empty class"""

    def __init__(self):
        pass


# EXP_PARAMS = INFRA()
# EXP_PARAMS.MEM_CAP = 1
# EXP_PARAMS.EPST = (0.95, 0.05, 0.95)  # (start, min, max)
# EXP_PARAMS.DECAY_MUL = 0.99999
# EXP_PARAMS.DECAY_ADD = 0


PIE_PARAMS = INFRA()
PIE_PARAMS.LAYERS = [64, 64, 64]
PIE_PARAMS.OPTIM = torch.optim.Adam  # 1. RMSprop, 2. Adam, 3. SGD
PIE_PARAMS.LOSS = torch.nn.MSELoss
PIE_PARAMS.LR = 0.001
PIE_PARAMS.DISCOUNT = 0.99
PIE_PARAMS.DOUBLE = False
PIE_PARAMS.TUF = 4
PIE_PARAMS.DEV = 'cpu'

TRAIN_PARAMS = INFRA()
TRAIN_PARAMS.EPOCHS = 50000
TRAIN_PARAMS.MOVES = 1
TRAIN_PARAMS.EPISODIC = False
TRAIN_PARAMS.MIN_MEM = 1
TRAIN_PARAMS.LEARN_STEPS = 64
TRAIN_PARAMS.BATCH_SIZE = 100
TRAIN_PARAMS.TEST_FREQ = 10

TEST_PARAMS = INFRA()
TEST_PARAMS.CERF = 100
TEST_PARAMS.RERF = 100


MODEL_HYPARAMS = f'LR-{PIE_PARAMS.LR}_TF-{TRAIN_PARAMS.TEST_FREQ}_BS-{TRAIN_PARAMS.BATCH_SIZE}_LS-{TRAIN_PARAMS.LEARN_STEPS}'


P = print


def F(fig, file_name): return plt.close()  # print('FIGURE ::',file_name)


def T(header, table): return print(header, '\n', table)


P('#', ALIAS)

##############################################
# Setup ENVS
##############################################

# Train ENV
env = gym.make(ENV_NAME)

# Test ENV
venv = gym.make(ENV_NAME)


# SET ENV SEEDS
# env.seed(0)
# venv.seed(0)

# Policy and Exploration
# exp = EXP(env=env, cap=EXP_PARAMS.MEM_CAP, epsilonT=EXP_PARAMS.EPST)

# txp = EXP(env=venv, cap=math.inf, epsilonT=(0, 0, 0))


# def decayF(epsilon, moves, isdone):
#     global eps
#     new_epsilon = epsilon*EXP_PARAMS.DECAY_MUL + \
#         EXP_PARAMS.DECAY_ADD  # random.random()
#     eps.append(new_epsilon)
#     return new_epsilon


# pie = PPO(
#     env,
#     env.observation_space.shape[0],
#     PIE_PARAMS.LAYERS,
#     env.action_space.n,
#     device=PIE_PARAMS.DEV,
#     # opt=PIE_PARAMS.OPTIM,
#     # cost=PIE_PARAMS.LOSS,
#     learning_rate=PIE_PARAMS.LR,
#     gamma=PIE_PARAMS.DISCOUNT,
#     # mapper=lambda x: x,
#     # double=PIE_PARAMS.DOUBLE,
#     # tuf=PIE_PARAMS.TUF,
#     # seed=None
# )

pie = PPO(env)


log_data = []
##############################################
# Fetch Initial Model Params (If Available)
##############################################

# reply = modman.send_model_params(
#         URL, modman.convert_tensor_to_list(pie.Q.state_dict()), PIE_PARAMS.LR, ALIAS)
# print("Response:",reply)

while modman.get_model_lock(URL, ALIAS):  # wait if model updation is going on
    print("Waiting for Model Lock Release1.")
    sleep(0.02)
print("before fetch params")
global_params, n_push, log_id, is_available = modman.fetch_params(URL, ALIAS)
# print("global params", global_params)
n_steps = n_push
print("After fetch Params")


if is_available:
    P("Model exist")
    P("Loading Q params .....")
    P("Number Push: ", n_push)
    log_data.append(
        [f'Log Data for Client IPAddres: {IPAddr} Pid: {getpid()}'])
    log_data.append(["Model exist"])
    log_data.append(['Loading Q params .......'])
    log_data.append(["Number Push: ", n_push])
    pie.load_state_dict(modman.convert_list_to_tensor(global_params))
else:
    P("Setting model for server")
    global_params, n_push, log_id, Iteration = modman.send_model_params(
        URL, modman.convert_tensor_to_list(pie.state_dict()), PIE_PARAMS.LR, ALIAS)
    global_params, n_push, log_id, is_available = modman.fetch_params(
        URL, ALIAS)
    P("Number Push: ", n_push)
    log_data.append(
        [f'Log Data for Client IPAddres: {IPAddr} Pid: {getpid()}'])
    log_data.append(["Setting model for server"])
    log_data.append(["Number Push: ", n_push])
    # print(reply)
    # print("\n\n........Params->", global_params)
    pie.load_state_dict(modman.convert_list_to_tensor(global_params))

n_steps = n_push
log_path = './client/logs/'
log_file = log_id + 'client_logs.csv'
log_testing = log_id + 'testing_logs.csv'

path1 = modman.increment_path(
    path=log_path+log_file, exist_ok=False, mkdir=True)
path2 = modman.increment_path(
    path=log_path+log_testing, exist_ok=False, mkdir=True)
modman.csv_writer(path=path1, data=[
                  [f'Log Data for Client IPAddres: {IPAddr} Pid: {getpid()}']])
modman.csv_writer(path=path2, data=[
                  [f'Log Data for Client IPAddres: {IPAddr} Pid: {getpid()}']])

##############################################
# Training
##############################################
P('#', 'Train')
P('Start Training...')
log_data.append(['Start Training...'])
stamp = now()
eps = []
ref = []
c_d1 = []  # communication delay 1
tpc = []  # Timeime per epoch
tft = []  # Time for testing
L_T = []  # Learning Time
REW = []  # Rewards list
REW.append(
    [f'\n\nTesting Data for Client IPAddres: {IPAddr} Pid: {getpid()} ..'])
max_reward1 = Queue(maxsize=100)

P('after max_reward queue')
# exp.reset(clear_mem=True, reset_epsilon=True)
# txp.reset(clear_mem=True, reset_epsilon=True)

LOG_CSV = 'epoch,reward,tr,loss\n'

current_observation = env.reset()


lt1 = now()  # setting initial learning time
for epoch in range(0, TRAIN_PARAMS.EPOCHS, TRAIN_PARAMS.MOVES):

    loss = None
    if ((epoch+1) % 1_000 == 0):
        log_dir = './logs/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sav_instance_path = f'{log_dir+str(getpid())+":"+ENV_NAME}_{stpc.strftime("%d_%m_%Y-%H_%M_%S")}'
        with open(sav_instance_path + '.csv', 'w') as f:
            f.write(LOG_CSV)

    stpc = now()  # start time for epoch
    lt1 += (now()-lt1)  # time at epoch start
    # exploration
    # _ = exp.explore(pie, moves=TRAIN_PARAMS.MOVES,
    #                 decay=decayF, episodic=TRAIN_PARAMS.EPISODIC)

    # if exp.memory.count >= TRAIN_PARAMS.MIN_MEM:

    #     for _ in range(TRAIN_PARAMS.LEARN_STEPS):
    # Single Learning Step
    pie.learn(total_timesteps=TRAIN_PARAMS.MOVES)

    sleep(0.01)
    # Send Parameters to Server
    if (epoch+1) % n_steps == 0:
        lt2 = now()
        print("Learning Time: ", lt2-lt1)
        L_T.append(lt2-lt1)
        lt1 = now()  # setting new initial learning time
        t1 = now()  # time stamp at the start time for communication

        # Sending Locally Trained Params
        print("Sending Local Params")
        reply = modman.send_local_update(URL,
                                         modman.convert_tensor_to_list(
                                             pie.state_dict()),
                                         epoch, ALIAS)
        print(reply, "\n Local update sent")
        log_data.append(reply)

        # Wait for Model Lock to get Released
        while modman.get_model_lock(URL, ALIAS):
            print("Waiting for Model Lock Release2.")
            sleep(0.05)

        # Get Updated Model Params from Server
        global_params, n_push, _, is_available = modman.fetch_params(
            URL, ALIAS)
        n_steps = n_push
        # print("Params->", global_params)
        pie.load_state_dict(
            modman.convert_list_to_tensor(global_params))

        t2 = now()  # time stamp at the end time of communication
        print("Communication delay: ", t2-t1)
        c_d1.append(t2-t1)
    etpc = now()  # end time for epoch
    tpc.append(etpc-stpc)
    stft = now()  # Start time for testing
    # P("after explore epoch#:",epoch)

    if epoch == 0 or (epoch+1) % TRAIN_PARAMS.TEST_FREQ == 0:
        res = np.array([test_model(venv, pie)
                       for _ in range(1)]).mean()
        trew = res
        ref.append([trew])
        # print('before queue')
        if (max_reward1.full()):
            max_reward1.get()
        max_reward1.put(trew)
        # print('after queue')
        P('[#]'+str(epoch+1), '\t',
            '[REW]'+str(trew),
            '[TR]'+str(pie.train_count),
            '[CR]', np.mean(max_reward1.queue),
            '[LS]', str(loss))
        LOG_CSV += f'{str(epoch+1)},{str(trew)},{str(pie.train_count)},{str(loss)}\n'
        REW.append(["Rew: ", trew, "Train_count: ", pie.train_count])
        if (max_reward1.full()):
            if (np.mean(max_reward1.queue) >= 200):
                break
    etft = now()  # End time for testing
    tft.append(etft-stft)

log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
sav_instance_path = f'{log_dir+str(getpid())+":END"+ENV_NAME}_{ALIAS}_EPC-{pie.train_count}_{MODEL_HYPARAMS}_{stamp.strftime("%d_%m_%Y-%H_%M_%S")}_FINISHED'
with open(sav_instance_path + '.csv', 'w') as f:
    f.write(LOG_CSV)

P('Finished Training!')
elapse = now() - stamp
P('Time Elapsed:', elapse)
P('Mean Learning Time:', np.mean(L_T))
P('MAX Learning Time:', np.max(L_T))
P('MIN Learning Time:', np.min(L_T))
P('Mean Communication Time:', np.mean(c_d1))
P('MAX Communication Time:', np.max(c_d1))
P('MIN Communication Time:', np.min(c_d1))
P('Total Learning Time:->', np.sum(L_T))
P('Total Communication delay:->', np.sum(c_d1))
P('Mean time for epoch:', np.mean(tpc))
P('MIN time for epoch:', np.min(tpc))
P('MAX time for epoch:', np.max(tpc))
P('Total time for epoch:->', np.sum(tpc))
P('Total time for testing:->', np.sum(tft))

# preparing log data
log_data.append([f'\nTotal number of epochs: {epoch}'])
log_data.append(['\nTime Elapsed:', elapse])
log_data.append(['\nMean Learning Time:', np.mean(L_T)])
log_data.append(['MAX Learning Time:', np.max(L_T)])
log_data.append(['MIN Learning Time:', np.min(L_T)])
log_data.append(['\nMean Communication Time:', np.mean(c_d1)])
log_data.append(['MAX Communication Time:', np.max(c_d1)])
log_data.append(['MIN Communication Time:', np.min(c_d1)])
log_data.append(['\nTotal Learning Time:->', np.sum(L_T)])
log_data.append(['\nTotal Communication delay:->', np.sum(c_d1)])
log_data.append(['\nMean time for epoch:', np.mean(tpc)])
log_data.append(['MIN time for epoch:', np.min(tpc)])
log_data.append(['MAX time for epoch:', np.max(tpc)])
log_data.append(['\nTotal time for epoch:->', np.sum(tpc)])
log_data.append(['\nTotal time for testing:->', np.sum(tft)])
modman.csv_writer(path=path1, data=log_data)
modman.csv_writer(path=path2, data=REW)
