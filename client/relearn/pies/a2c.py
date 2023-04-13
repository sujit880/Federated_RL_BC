from typing import List, Dict, Any, Tuple

import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from copy import deepcopy


class ActorNet(torch.nn.Module):
    '''
    Actor Deep Neural Network class implementation
    '''

    def __init__(self, state_dim: int, layer_list: List[int], action_dim: int, learning_rate=0.001, optimizer=torch.optim.Adam, loss_fn=torch.nn.MSELoss):
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
        self.optimizer = optimizer(
            self.network.parameters(), lr=self.lr)  # opt = optim.Adam
        self.loss_fn = loss_fn()  # cost=nn.MSELoss()

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

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> float:
        '''
        Method to train the Neural Network

        Parameters:

        x: torch.Tensor, the input tensor for forward pass of the Neural Network
        '''
        target = y.detach().clone().float()

        # Forward Pass
        pred = self.forward(x)

        # Loss calculation
        loss = self.loss_fn(pred, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class CriticNet(torch.nn.Module):
    '''
    Critic Deep Neural Network class implementation
    '''

    def __init__(self, state_dim: int, layer_list: List[int], learning_rate=0.001, optimizer=torch.optim.Adam, loss_fn=torch.nn.MSELoss):
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
        self.optimizer = optimizer(
            self.network.parameters(), lr=self.lr)  # opt = optim.Adam
        self.loss_fn = loss_fn()  # cost=nn.MSELoss()

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

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        '''
        Method to train the Neural Network

        Parameters:

        x: torch.Tensor, the input tensor for forward pass of the Neural Network
        '''

        target = y.detach().clone().float()

        # Forward Pass
        pred = self.forward(x)

        # Loss calculation
        loss = self.loss_fn(pred, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class A2C:

    """
        Implements A2C based Policy

        state_dim       Observation Space Shape
        layer_list      List of layer sizes for eg. LL=[32,16,8]
        action_dim      Action Space (should be discrete)
        opt             torch.optim     (eg - torch.optim.Adam)
        cost            torch.nn.<loss> (eg - torch.nn.MSELoss)
        lr              Learning Rate for DQN Optimizer ()
        discount        discount factor
        device          can be 'cuda' or 'cpu'
    """

    def __init__(self, state_dim, layer_list, action_dim, optimizer=torch.optim.Adam, loss_fn=torch.nn.MSELoss, learning_rate=0.001, discount=0.7, gamma=0.99,  device='cpu', seed=None):
        '''
        Constructor Method for A2C RL algorithm

        state_dim       Observation Space Shape
        layer_list      List of layer sizes for eg. LL = [32, 16, 8]
        action_dim      Action Space(should be discrete)
        optimizer       torch.optim(eg - torch.optim.Adam)
        loss_fn         torch.nn. < loss > (eg - torch.nn.MSELoss)
        learning_rate   Learning Rate for DQN Optimizer()
        discount        discount factor
        device          can be 'cuda' or 'cpu'
        '''
        self.learning_rate = learning_rate
        self.discount = discount
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rand = np.random.default_rng(seed)
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.actor_base_model = ActorNet(
            state_dim, layer_list, action_dim, self.learning_rate, self.optimizer, self.loss_fn).to(self.device)
        self.actor_net = ActorNet(
            state_dim, layer_list, action_dim, self.learning_rate, self.optimizer, self.loss_fn).to(self.device)
        self.critic_base_model = CriticNet(
            state_dim, layer_list, self.learning_rate, self.optimizer, self.loss_fn).to(self.device)
        self.critic_net = CriticNet(
            state_dim, layer_list, self.learning_rate, self.optimizer, self.loss_fn).to(self.device)
        self.clear()

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

    def learn(self, env, observation) -> Tuple[Any, Tuple[float, float]]:
        '''
        Train the model of a single step
        '''

        # Reshape Observations => model dims are (batch, env.observation_space.n)
        observation_reshaped = torch.from_numpy(observation.reshape(
            [1, observation.shape[0]]))
        action_probs = self.actor_net(observation_reshaped).flatten()

        # Note we're sampling from the prob distribution instead of using argmax
        action = np.random.choice(
            env.action_space.n, 1, p=action_probs.detach().numpy())[0]
        encoded_action = self._one_hot_encode_action(
            action, env.action_space.n)

        # Reshape Next Observations => model dims are (batch, env.observation_space.n)
        next_observation, reward, done, _ = env.step(action)
        next_observation_reshaped = torch.from_numpy(next_observation.reshape(
            [1, next_observation.shape[0]]))

        value_curr = np.asscalar(
            np.array(self.critic_net(observation_reshaped).detach().numpy()))
        value_next = np.asscalar(
            np.array(self.critic_net(next_observation_reshaped).detach().numpy()))

        # Fit on the current observation
        td_target = reward + (1 - done) * self.discount * value_next
        advantage = td_target - value_curr
        print(np.around(action_probs.detach().numpy(), 2), np.around(value_next -
              value_curr, 3), 'Advantage:', np.around(advantage, 2))

        advantage_reshaped = np.vstack([advantage])
        td_target = np.vstack([td_target])
        critic_loss = self.critic_net.fit(
            observation_reshaped, torch.from_numpy(td_target))

        gradient = torch.from_numpy(encoded_action) - action_probs
        gradient_with_advantage = .0001 * gradient * \
            torch.from_numpy(advantage_reshaped) + action_probs
        actor_loss = self.actor_net.fit(
            observation_reshaped, gradient_with_advantage)

        self.train_count += 1

        # if environment is done, reset and return the new observation, instead of returning next_observation, return it
        if done:
            return env.reset(), tuple([actor_loss, critic_loss])

        return next_observation, tuple([actor_loss, critic_loss])

    # def render(self, mode=0, p=print):
    #     return None

    # def save(self, path):
    #     th.save(self.actor_critic_net, path)
    #     return

    # def load(self, path):
    #     self.base_model = th.load(path)
    #     self._clear_acnet()
    #     return

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        '''
        Segregate and load parameters from the merged dict into the actor and critic models
        '''

        actor, critic = {}, {}

        for key, value in state_dict.items():
            print(key)
            if 'actor.' in key:
                actor[key.replace('actor.', '')] = value
            elif 'critic.' in key:
                critic[key.replace('critic.', '')] = value

        print(critic)

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

        print(actor)

        # merge the actor dict with the critic dict
        merged = actor
        merged.update(critic)

        return merged

    def _one_hot_encode_action(self, action, n_actions):
        encoded = np.zeros(n_actions)
        encoded[action] = 1
        return encoded
