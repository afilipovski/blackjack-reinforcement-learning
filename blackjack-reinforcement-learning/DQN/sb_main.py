import gymnasium as gym
from gymnasium.spaces import Discrete
from stable_baselines3 import DQN

class BlackjackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Discrete(32 * 11 * 2)

    def observation(self, obs):
        player_sum, dealer_card, usable_ace = obs
        return (player_sum * 11 * 2) + (dealer_card * 2) + usable_ace

env = BlackjackObsWrapper(gym.make("Blackjack-v1"))

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0064935100500415335,
    buffer_size=240000,
    learning_starts=1000,
    batch_size=128,
    gamma=0.55,
    train_freq=13,
    target_update_interval=3000,
    exploration_fraction=0.0004291352471643776,
    exploration_final_eps=0.01,
    exploration_initial_eps=1,
    verbose=1,
)

if input("Train? (Y/N): ") == "Y":
    model.learn(total_timesteps=1_000_00)
    name = input("Name the agent: ")
    model.save(name)

if input("Test? (Y/N): ") == "Y":
    results = []
    name = input("Enter the name of the agent: ")
    model = DQN.load(name)
    
    for i in range(50000):
        obs, _ = env.reset()
        done = False
        reward_sum = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            reward_sum += reward
            print(f"Action: {action}, Reward: {reward}")
        results.append(reward_sum)
        
    print(f"Win rate: {len([r for r in results if r>=1])/len(results):.2%}")
    print(f"Draw rate: {len([r for r in results if r==0])/len(results):.2%}")
    print(f"Loss rate: {len([r for r in results if r==-1])/len(results):.2%}")