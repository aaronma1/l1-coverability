

from stable_baselines3 import SAC
import gymnasium as gym


env = gym.make("MountainCarContinuous-v0", render_mode= "rgb_array")

env.reset()

model = SAC('MlpPolicy', env)
model._setup_model()
model.learn(100)