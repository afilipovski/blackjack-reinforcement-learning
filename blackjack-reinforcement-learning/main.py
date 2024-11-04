import gym

env = gym.make("Blackjack-v1", render_mode="human")

done = False
observation, info = env.reset()

pass