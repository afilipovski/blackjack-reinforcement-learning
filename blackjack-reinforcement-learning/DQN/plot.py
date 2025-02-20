import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from gymnasium.spaces import Discrete
import gymnasium as gym

class BlackjackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Discrete(32 * 11 * 2)

    def observation(self, obs):
        player_sum, dealer_card, usable_ace = obs
        return (player_sum * 11 * 2) + (dealer_card * 2) + usable_ace

env = BlackjackObsWrapper(gym.make("Blackjack-v1"))
name = input("Enter the name of the agent: ")
model = DQN.load(name)

num_episodes = 50000
cumulative_rewards = []
total_reward = 0

for i in range(num_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward

    total_reward += episode_reward
    cumulative_rewards.append(total_reward)

plt.figure(figsize=(10, 5))
plt.plot(cumulative_rewards, label=f"DQN Cumulative Reward - caps at {total_reward}")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Episodes")
plt.legend()
plt.grid()
plt.show()