import gymnasium as gym
import numpy as np
import time
import os
import csv
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from deep_q_learning import DQN
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def setup_csv_logger(mode="training"):
    """Setup CSV logger and create necessary directories"""
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = f"logs/{mode}_{timestamp}.csv"

    headers = [
        "Episode",
        "Steps",
        "Reward",
        "Epsilon",
        "Memory_Size",
        "Win",
        "Timestamp",
    ]

    file_handle = open(csv_file, "w", newline="")
    csv_writer = csv.DictWriter(file_handle, fieldnames=headers)
    csv_writer.writeheader()
    return csv_writer, file_handle


def log_episode(
    csv_writer, episode, total_episodes, steps, reward, epsilon, memory_size
):
    """Helper function to log episode data to CSV"""
    csv_writer.writerow(
        {
            "Episode": episode + 1,
            "Steps": steps,
            "Reward": reward,
            "Epsilon": epsilon,
            "Memory_Size": memory_size,
            "Win": 1 if reward > 0 else 0,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    print(f"Episode {episode + 1}/{total_episodes} - Reward: {reward:.2f}")


def save_weights(agent, episode, is_final=False):
    """Helper function to save model weights"""
    try:
        filename = "final" if is_final else f"episode_{episode}"
        weight_path = os.path.join("weights", f"blackjack_{filename}.h5")
        agent.save(weight_path, episode)
        print(f"Saved weights to {weight_path}")
    except Exception as e:
        print(f"Error saving weights: {e}")


def build_model(state_space_shape, num_actions):
    """Build and compile the DQN model"""
    model = Sequential(
        [
            Input(shape=state_space_shape),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(num_actions, activation="linear"),
        ]
    )
    model.compile(Adam(learning_rate=0.001), mean_squared_error)
    return model


if __name__ == "__main__":
    # Environment setup
    env = gym.make("Blackjack-v1", render_mode="human")
    env.reset()

    # Parameters
    STATE_SPACE_SHAPE = 3
    NUM_ACTIONS = 2
    NUM_EPISODES = 25  # Temp while in dev
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    VISUALIZATION_DELAY = 0.5

    # User inputs
    is_training = str.lower(input("Training or testing? (1/2): ")) == "1"
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
            start_time = time.time()

            for episode in range(NUM_EPISODES):
                state, _ = env.reset()
                env.render()
                state = np.array(state)
                done = False
                episode_reward = 0
                steps = 0

                while not done:
                    action = agent.get_action(state, epsilon)
                    next_state, reward, done, _, _ = env.step(action)
                    env.render()
                    time.sleep(VISUALIZATION_DELAY)
                    next_state = np.array(next_state)

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
                    episode_reward,
                    epsilon,
                    len(agent.memory),
                )

            if should_save_weights:
                save_weights(agent, NUM_EPISODES, is_final=True)

        finally:
            file_handle.close()

    else:
        csv_logger, file_handle = setup_csv_logger("testing")
        try:
            weight_files = sorted(
                [f for f in os.listdir("weights") if f.endswith(".h5")]
            )
            if weight_files:
                latest_weights = os.path.join("weights", weight_files[-1])
                agent.load(latest_weights, NUM_EPISODES)
                print(f"Loaded weights from {latest_weights}")

            test_episodes = 1000
            test_rewards = []
            start_time = time.time()

            for episode in range(test_episodes):
                state, _ = env.reset()
                env.render()
                state = np.array(state)
                done = False
                episode_reward = 0
                steps = 0

                while not done:
                    action = agent.get_action(state, 0)
                    next_state, reward, done, _, _ = env.step(action)
                    env.render()
                    time.sleep(VISUALIZATION_DELAY)
                    state = np.array(next_state)
                    episode_reward += reward
                    steps += 1

                test_rewards.append(episode_reward)

                log_episode(
                    csv_logger,
                    episode,
                    test_episodes,
                    steps,
                    episode_reward,
                    0,
                    len(agent.memory),
                )

        finally:
            file_handle.close()

    env.close()
