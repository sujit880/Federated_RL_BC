from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn as nn
import time
import gym
from typing import List, Dict, Any, Tuple

import numpy as np
import torch


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, layer_list: List[int], action_dim: int):
        super(ActorCritic, self).__init__()

        self.n_layers = len(layer_list)
        if self.n_layers < 1:
            raise ValueError('need at least 1 layers')

        # critic input layer
        actor_layers = [torch.nn.Linear(
            state_dim, layer_list[0]), torch.nn.ReLU()]

        # hidden layers
        for i in range(self.n_layers-1):
            actor_layers.append(torch.nn.Linear(
                layer_list[i], layer_list[i+1]))
            actor_layers.append(torch.nn.ReLU())

        # actor output layers
        actor_layers.append(torch.nn.Linear(layer_list[-1], action_dim))
        actor_layers.append(torch.nn.Softmax(dim=-1))

        # actor
        self.actor = nn.Sequential(*actor_layers)

        # critic input layer
        critic_layers = [torch.nn.Linear(
            state_dim, layer_list[0]), torch.nn.ReLU()]

        # hidden layers
        for i in range(self.n_layers-1):
            critic_layers.append(torch.nn.Linear(
                layer_list[i], layer_list[i+1]))
            critic_layers.append(torch.nn.ReLU())

        # output layer
        critic_layers.append(torch.nn.Linear(layer_list[-1], 1))

        # critic
        self.critic = nn.Sequential(*critic_layers)

    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def get_weights(self) -> Dict[str, Any]:
        '''
        Method to get the weights of the Neural Network
        '''
        return self.state_dict()


class PPO:
    """
            This is the PPO class we will use as our model in main.py
    """

    def __init__(self, env, state_dim, layer_list, action_dim, optimizer=torch.optim.Adam, loss_fn=torch.nn.MSELoss, eps_clip=0.2, k_epochs=10, update_timestep=10, learning_rate=0.001,  gamma=0.99,  device='cpu'):
        """
                Initializes the PPO model, including hyperparameters.

                Parameters:
                    state_dim       Observation Space Shape
                    layer_list      List of layer sizes for eg. LL = [32, 16, 8]
                    action_dim      Action Space(should be discrete)
                    optimizer       torch.optim(eg - torch.optim.Adam)
                    loss_fn         torch.nn. < loss > (eg - torch.nn.MSELoss)
                    learning_rate   Learning Rate for DQN Optimizer()
                    discount        discount factor
                    device          can be 'cuda' or 'cpu'
        """
        self.env = env

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.loss_fn = loss_fn()
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        self.update_timestep = update_timestep

        self.buffer = RolloutBuffer()

        self.base_policy = ActorCritic(
            state_dim, layer_list, action_dim).to(self.device)

        self.policy = ActorCritic(
            state_dim, layer_list, action_dim).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.0001},
            {'params': self.policy.critic.parameters(), 'lr': 0.0001}
        ])

        self.learn_rew = 0

        self.policy_old = ActorCritic(
            state_dim, layer_list, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.clear()

    def clear(self):
        '''
        Method to clear the A2C agent
        '''
        self._clear_policy_net()
        self.train_count = 0
        self.update_count = 0

    def _clear_policy_net(self):
        self.policy.load_state_dict(self.base_policy.state_dict())
        self.policy.eval()

    def _select_action(self, observation, test=False):
        observation = torch.from_numpy(observation)

        with torch.no_grad():
            action, action_log_prob, state_val = self.policy.act(
                observation)

        if not test:
            self.buffer.states.append(observation)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_log_prob)
            self.buffer.state_values.append(state_val)

        return action.item()

    def predict(self, observation: np.array) -> int:
        '''
        Predict the concrete action based on the obervation provided to the actor network
        '''
        return self._select_action(observation, test=True)

    def learn(self, observation):
        """
                Train the actor and critic networks. Here is where the main PPO algorithm resides.

                Parameters:
                        total_timesteps - the total number of timesteps to train for

                Return:
                        None
        """

        # select action with policy
        action = self._select_action(observation)
        observation, reward, done, _ = self.env.step(action)

        # print('LEARN STEP DONE', done)

        # saving reward and is_terminals
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

        self.train_count += 1
        self.learn_rew += reward

        loss = 999

        # update PPO agent
        if self.train_count % self.update_timestep == 0:
            loss = self._update()

        if done:
            # print('TR REW', self.learn_rew, done)
            self.learn_rew = 0
            observation = self.env.reset()

        return observation, loss

    def _update(self):
        """
                Train the actor and critic networks. Here is where the main PPO algorithm resides.

                Parameters:
                        total_timesteps - the total number of timesteps to train for

                Return:
                        None
        """

        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(
            self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.loss_fn(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        # self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return 0  # loss.item()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        '''
        Segregate and load parameters from the merged dict into the actor and critic models
        '''

        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def state_dict(self) -> Dict[str, Any]:
        '''
        Return the state dict of both actor and critic networks as a single dict
        '''

        return self.policy.get_weights().copy()


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
