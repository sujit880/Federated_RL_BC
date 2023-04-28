from stable_baselines3 import A2C as sb_a2c


class A2C():
    def __init__(self, env, verbose=0):
        self.model = sb_a2c('MlpPolicy', env, verbose=verbose, n_steps=100)
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
