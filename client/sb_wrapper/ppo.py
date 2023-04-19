from stable_baselines3 import PPO as sb_ppo


class PPO():
    def __init__(self, env, verbose=0):
        self.model = sb_ppo('MlpPolicy', env, verbose=verbose, n_steps=100)
        self.train_count = 0

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
        self.train_count += total_timesteps

    def predict(self, observation):
        action, _states = self.model.predict(observation)
        return action

    def state_dict(self):
        return self.model.policy.state_dict()

    def load_state_dict(self, state_dict):
        self.model.policy.load_state_dict(state_dict)
