import gym
from stable_baselines3 import DQN, A2C, PPO

# Create environment
env = gym.make('LunarLander-v2')
# Load the trained agent
model = PPO.load("PPO_lunar.zip", env=env)

# Enjoy trained agent
while True:
    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            break
