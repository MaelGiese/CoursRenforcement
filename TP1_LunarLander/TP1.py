import gym

from stable_baselines3 import DQN, A2C, PPO

# Create environment
env = gym.make('LunarLander-v2')
if True:
    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(6000000))
    # Save the agent
    model.save("DQN_lunar")
    del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("DQN_lunar.zip", env=env)


# Enjoy trained agent
while True:
    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            break

