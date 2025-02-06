import gymnasium as gym
import numpy as np
import os
import warnings
from deep_q_learning import DQN
from utils import (
    ensure_directories,
    setup_csv_logger,
    log_episode,
    save_weights,
    build_model,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    ensure_directories()

    # Environment setup
    env = gym.make("Blackjack-v1")
    env.reset()

    # Parameters
    STATE_SPACE_SHAPE = 3
    NUM_ACTIONS = 2
    NUM_EPISODES = 50
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01

    # User inputs
    is_training = str.lower(input("Training or testing? (1/2): ")) == "1"
    if is_training:
        should_save_weights = str.lower(input("Save weights? (y/n): ")) == "y"

    # Model initialization
    model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS)
    target_model = build_model(STATE_SPACE_SHAPE, NUM_ACTIONS)
    agent = DQN(
        STATE_SPACE_SHAPE,
        NUM_ACTIONS,
        model,
        target_model,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
    )

    if is_training:
        csv_logger, file_handle = setup_csv_logger("training")
        try:
            epsilon = 1.0
            total_rewards = []

            for episode in range(NUM_EPISODES):
                state, _ = env.reset()
                state = np.array(state)
                done = False
                steps = 0
                episode_reward = 0

                player_sum, dealer_card, usable_ace = state
                is_natural = player_sum == 21
                dealer_sum = dealer_card  # Initial card

                while not done:
                    action = agent.get_action(state, epsilon)
                    next_state, reward, done, _, info = env.step(action)
                    next_state = np.array(next_state)

                    if done:
                        dealer_sum = info.get("dealer_sum", dealer_sum)

                    agent.update_memory(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    steps += 1

                if len(agent.memory) > BATCH_SIZE:
                    agent.train()

                if episode % 100 == 0:
                    agent.update_target_model()

                epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
                total_rewards.append(episode_reward)

                log_episode(
                    csv_logger,
                    episode,
                    NUM_EPISODES,
                    steps,
                    state,
                    dealer_sum,
                    is_natural,
                    epsilon,
                    len(agent.memory),
                )

            if should_save_weights:
                save_weights(agent, NUM_EPISODES)

        finally:
            file_handle.close()

    else:
        csv_logger, file_handle = setup_csv_logger("testing")
        try:
            agent.load("blackjack", NUM_EPISODES)
            print("Loaded weights")

            test_episodes = 1000
            test_rewards = []

            for episode in range(test_episodes):
                state, _ = env.reset()
                env.render()
                state = np.array(state)
                done = False
                episode_reward = 0
                steps = 0

                player_sum, dealer_card, usable_ace = state
                is_natural = player_sum == 21
                dealer_sum = dealer_card

                while not done:
                    action = agent.get_action(state, 0)
                    next_state, reward, done, _, info = env.step(action)
                    state = np.array(next_state)
                    episode_reward += reward
                    steps += 1

                    if done:
                        dealer_sum = info.get("dealer_sum", dealer_sum)

                    state = np.array(next_state)
                    episode_reward += reward
                    steps += 1

                test_rewards.append(episode_reward)

                log_episode(
                    csv_logger,
                    episode,
                    test_episodes,
                    steps,
                    state,
                    dealer_sum,
                    is_natural,
                    0,
                    len(agent.memory),
                )

        finally:
            file_handle.close()

    env.close()
