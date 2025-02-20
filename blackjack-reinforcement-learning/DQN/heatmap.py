import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
name = input("Enter the name of the trained agent file: ")
model = DQN.load(name)

# Dimensions: (Player's total 4-21, Dealer's visible card 1-10, Usable Ace True/False)
action_counts = np.zeros((22, 11, 2))

num_episodes = 100000

for _ in range(num_episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        
        # Decode state from observation space encoding
        player_sum = (obs // (11 * 2))
        dealer_card = (obs % (11 * 2)) // 2
        usable_ace = (obs % 2)

        # Store action choice (0 = Stick, 1 = Hit)
        action_counts[player_sum, dealer_card, usable_ace] += action

        obs, _, done, _, _ = env.step(action)

# Normalize actions to probability (0 = Stick, 1 = Hit)
action_probs = action_counts / np.maximum(1, action_counts.sum(axis=-1, keepdims=True))

def plot_heatmap(action_probs, usable_ace, title, filename):
    plt.figure(figsize=(10, 6))
    sns.heatmap(action_probs[:, :, usable_ace], annot=True, cmap="coolwarm", 
                     xticklabels=range(1, 11), yticklabels=range(4, 22), cbar_kws={'label': "Probability of Hitting (1 = Hit, 0 = Stick)"})
    plt.xlabel("Dealer's Visible Card")
    plt.ylabel("Player's Total")
    plt.title(title)
    plt.savefig(filename)
    plt.show()
    print(f"Saved heatmap as {filename}")

plot_heatmap(action_probs, usable_ace=0, title="DQN Action Preferences Without Usable Ace", filename="heatmap_no_usable_ace.png")
plot_heatmap(action_probs, usable_ace=1, title="DQN Action Preferences With Usable Ace", filename="heatmap_usable_ace.png")

print("Heatmaps generated successfully.")
