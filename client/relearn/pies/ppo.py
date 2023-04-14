from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn as nn
import time
import gym
from typing import List, Dict, Any, Tuple

import numpy as np
import torch


class ActorNet(torch.nn.Module):
    '''
    Actor Deep Neural Network class implementation
    '''

    def __init__(self, state_dim: int, layer_list: List[int], action_dim: int, learning_rate=0.001, optimizer=torch.optim.Adam):
        '''
        Constructor method for Actor Critic Deep Neural Network

        Parameters:
        state_dim : int, defines the state dimension of the RL environment
        layer_list : list(int), defines the list of layers for the DNN, e.g., [8, 32, 16, 2]
        action_dim : int, defines the action dimension of the RL agent
        '''

        super(ActorNet, self).__init__()

        self.n_layers = len(layer_list)
        if self.n_layers < 1:
            raise ValueError('need at least 1 layers')

        # input layer
        layers = [torch.nn.Linear(state_dim, layer_list[0]), torch.nn.ReLU()]

        # hidden layers
        for i in range(self.n_layers-1):
            layers.append(torch.nn.Linear(layer_list[i], layer_list[i+1]))
            layers.append(torch.nn.ReLU())

        # output layer
        layers.append(torch.nn.Linear(layer_list[-1], action_dim))
        layers.append(torch.nn.Softmax())

        self.network = torch.nn.Sequential(*layers)
        self.lr = learning_rate
        # self.optimizer = optimizer(
        #     self.network.parameters(), lr=self.lr)  # opt = optim.Adam

    def get_weights(self) -> Dict[str, Any]:
        '''
        Method to get the weights of the Neural Network
        '''
        return self.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Method for forward pass of the Neural Network

        Parameters:

        x: torch.Tensor, the input tensor for forward pass of the Neural Network

        Returns:
        torch.Tensor, the output tensor of the Neural Network
        '''
        return self.network(x)


class CriticNet(torch.nn.Module):
    '''
    Critic Deep Neural Network class implementation
    '''

    def __init__(self, state_dim: int, layer_list: List[int], learning_rate=0.001, optimizer=torch.optim.Adam):
        '''
        Constructor method for Actor Critic Deep Neural Network

        Parameters:
        state_dim : int, defines the state dimension of the RL environment
        layer_list : list(int), defines the list of layers for the DNN, e.g., [8, 32, 16, 2]
        action_dim : int, defines the action dimension of the RL agent
        '''

        super(CriticNet, self).__init__()

        self.n_layers = len(layer_list)
        if self.n_layers < 1:
            raise ValueError('need at least 1 layers')

        # input layer
        layers = [torch.nn.Linear(state_dim, layer_list[0]), torch.nn.ReLU()]

        # hidden layers
        for i in range(self.n_layers-1):
            layers.append(torch.nn.Linear(layer_list[i], layer_list[i+1]))
            layers.append(torch.nn.ReLU())

        # output layer
        layers.append(torch.nn.Linear(layer_list[-1], 1))

        self.network = torch.nn.Sequential(*layers)
        self.lr = learning_rate
        # self.optimizer = optimizer(
        #     self.network.parameters(), lr=self.lr)  # opt = optim.Adam

    def get_weights(self) -> Dict[str, Any]:
        '''
        Method to get the weights of the Neural Network
        '''
        return self.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Method for forward pass of the Neural Network

        Parameters:

        x: torch.Tensor, the input tensor for forward pass of the Neural Network

        Returns:
        torch.Tensor, the output tensor of the Neural Network
        '''
        return self.network(x)


class PPO:
    """
            This is the PPO class we will use as our model in main.py
    """

    def __init__(self, env, state_dim, layer_list, action_dim, optimizer=torch.optim.Adam, loss_fn=torch.nn.MSELoss, learning_rate=0.001, discount=0.7, gamma=0.99,  device='cpu', seed=None):
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

        self.learning_rate = 0.005
        self.discount = discount
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rand = np.random.default_rng(seed)
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.actor_base_model = ActorNet(
            state_dim, layer_list, action_dim, self.learning_rate, self.optimizer).to(self.device)
        self.actor_net = ActorNet(
            state_dim, layer_list, action_dim, self.learning_rate, self.optimizer).to(self.device)
        self.critic_base_model = CriticNet(
            state_dim, layer_list, self.learning_rate, self.optimizer).to(self.device)
        self.critic_net = CriticNet(
            state_dim, layer_list, self.learning_rate, self.optimizer).to(self.device)
        self.clear()

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.env = env
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.memory = RolloutMemory(
            self.env, self.actor_net, self.gamma, self.cov_mat)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(
            self.actor_net.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(
            self.critic_net.parameters(), lr=self.learning_rate)

    def clear(self):
        '''
        Method to clear the A2C agent
        '''
        self._clear_actor_net()
        self._clear_critic_net()
        self.train_count = 0
        self.update_count = 0

    def _clear_actor_net(self):
        self.actor_net.load_state_dict(self.actor_base_model.state_dict())
        self.actor_net.eval()

    def _clear_critic_net(self):
        self.critic_net.load_state_dict(self.critic_base_model.state_dict())
        self.critic_net.eval()

    def predict(self, observation: np.array) -> int:
        '''
        Predict the concrete action based on the obervation provided to the actor network
        '''
        # with self.actor_net.eval():
        # also can be torch.argmax, for maximum probability
        action = self.actor_net(torch.from_numpy(observation))
        return torch.argmax(action).detach().item()

    def learn(self, batch_size: int):
        """
                Train the actor and critic networks. Here is where the main PPO algorithm resides.

                Parameters:
                        total_timesteps - the total number of timesteps to train for

                Return:
                        None
        """

        # Autobots, roll out (just kidding, we're collecting our batch simulations here)
        batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.memory.rollout(
            batch_size)                     # ALG STEP 3

        # # Calculate how many timesteps we collected this batch
        # t_so_far += np.sum(batch_lens)

        # # Increment the number of iterations
        # i_so_far += 1

        # # Logging timesteps so far and iterations so far
        # self.logger['t_so_far'] = t_so_far
        # self.logger['i_so_far'] = i_so_far

        # Calculate advantage at k-th iteration
        V, _ = self.evaluate(batch_obs, batch_acts)
        # ALG STEP 5
        A_k = batch_rtgs - V.detach()

        # print('ADVANTEGE', A_k)

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases the variance of
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it.
        # A_k = torch.nn.functional.normalize(A_k)

        # print('NORMALIZE', A_k1, A_k)

        # This is the loop where we update our network for some n epochs
        # ALG STEP 6 & 7
        for _ in range(self.n_updates_per_iteration):
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation:
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            # print('EXPLOD', ratios, A_k, surr1, surr2)

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.update_count += 1

        self.train_count += 1

        # print(self.state_dict())

        # print(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def evaluate(self, batch_obs, batch_acts):
        """
                Estimate the values of each observation, and the log probs of
                each action in the most recent batch with the most recent
                iteration of the actor network. Should be called from learn.
                Parameters:
                        batch_obs - the observations from the most recently collected batch as a tensor.
                                                Shape: (number of timesteps in batch, dimension of observation)
                        batch_acts - the actions from the most recently collected batch as a tensor.
                                                Shape: (number of timesteps in batch, dimension of action)
                Return:
                        V - the predicted values of batch_obs
                        log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic_net(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()

        # print(batch_obs)

        mean = self.actor_net(batch_obs)
        # print('MEAN', mean)
        dist = Categorical(mean)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        '''
        Segregate and load parameters from the merged dict into the actor and critic models
        '''

        actor, critic = {}, {}

        for key, value in state_dict.items():
            if 'actor.' in key:
                actor[key.replace('actor.', '')] = value
            elif 'critic.' in key:
                critic[key.replace('critic.', '')] = value

        self.actor_net.load_state_dict(actor)
        self.critic_net.load_state_dict(critic)

        self.actor_net.eval()
        self.critic_net.eval()

    def state_dict(self) -> Dict[str, Any]:
        '''
        Return the state dict of both actor and critic networks as a single dict
        '''

        # get the actor dict
        actor = self.actor_net.get_weights().copy()

        # for all keys in the actor net, append 'actor.' in it
        actor = {f'actor.{key}': value for key, value in actor.items()}

        # get the critic dict
        critic = self.critic_net.get_weights().copy()

        # for all keys in the critic net, append 'critic.' in it
        critic = {f'critic.{key}': value for key, value in critic.items()}

        # merge the actor dict with the critic dict
        merged = actor
        merged.update(critic)

        return merged


class RolloutMemory:
    '''
    Class for rollout memory 
    '''

    def __init__(self, env, actor, gamma, cov_mat):
        self.env = env
        self.actor = actor
        self.gamma = gamma
        self.cov_mat = cov_mat
        self.current_observation, self.done = self.env.reset(), False

    def rollout(self, batch_size):
        """
                This is where we collect the batch of data
                from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
                of data each time we iterate the actor/critic networks.
                Parameters:
                        None
                Return:
                        batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                        batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                        batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                        batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < batch_size:
            t += 1  # Increment timesteps ran this batch so far

            # Track observations in this batch
            batch_obs.append(self.current_observation)

            # Calculate action and make a step in the env.
            action, log_prob = self.get_action(self.current_observation)
            self.current_observation, reward, self.done, _ = self.env.step(
                action)

            # Track recent reward, action, and action log probability
            batch_rews.append(reward)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)

            # If the environment tells us the episode is terminated, break
            if self.done:
                self.current_observation, self.done = self.env.reset(), False

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs,

    def get_action(self, observation):
        """
                Queries an action from the actor network, should be called from rollout.
                Parameters:
                        observation - the observation at the current timestep
                Return:
                        action - the action to take, as a numpy array
                        log_prob - the log probability of the selected action in the distribution
        """

        # print('OBS GET ACN', observation)

        # Query the actor network for a mean action
        mean = self.actor(torch.from_numpy(observation))

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM

        # print('ACTION GET', mean)

        dist = Categorical(mean)

        # Sample an action from the distribution
        action = dist.sample()

        # discrete_action = np.random.choice(
        #     self.env.action_space.n, 1, p=action.detach().numpy())[0]

        # print("ACTION", action.item())

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.item(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        """
                Compute the Reward-To-Go of each timestep in a batch given the rewards.
                Parameters:
                        batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
                Return:
                        batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """

        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        discounted_reward = 0  # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(batch_rews):
            discounted_reward = rew + discounted_reward * self.gamma
            batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs
