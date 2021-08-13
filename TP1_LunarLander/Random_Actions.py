import gym
import random

# Create environment
env = gym.make('LunarLander-v2')

while True:
    env.reset()
    for _ in range(500):
        action = random.randint(0, 3)

        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            break
