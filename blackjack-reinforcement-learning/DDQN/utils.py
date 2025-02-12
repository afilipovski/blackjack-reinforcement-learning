import os
import csv
from pathlib import Path
import time
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam


def ensure_directories():
    """Create necessary directories"""
    os.makedirs("logs", exist_ok=True)


def setup_csv_logger(mode="training"):
    """Setup CSV logger with blackjack-specific metrics"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = f"logs/{mode}/{timestamp}.csv"

    Path(f"logs/{mode}").mkdir(parents=True, exist_ok=True)

    headers = [
        "Episode",
        "Steps",
        "Reward",
        "Epsilon",
        "Memory_Size",
        "Outcome",
        "Player_Sum",
        "Dealer_Sum",
        "Dealer_Show_Card",
        "Game_State",
        "Usable_Ace",
        "Is_Natural",
        "Timestamp",
    ]

    file_handle = open(csv_file, "w", newline="")
    csv_writer = csv.DictWriter(file_handle, fieldnames=headers)
    csv_writer.writeheader()
    return csv_writer, file_handle


def get_game_state(player_sum, dealer_sum, is_natural):
    """Determine game state based on sums"""
    if player_sum > 21:
        return "bust"
    if is_natural and player_sum == 21:
        return "natural"
    if dealer_sum > 21:
        return "dealer_bust"
    if player_sum == dealer_sum:
        return "draw"
    return "stick"


def get_outcome(player_sum, dealer_sum, is_natural):
    """Determine game outcome"""
    state = get_game_state(player_sum, dealer_sum, is_natural)

    if state == "bust":
        return "lose", -1
    if state == "natural":
        return "natural", 1.5
    if state == "dealer_bust":
        return "win", 1
    if state == "draw":
        return "draw", 0

    # Compare sums if neither busted
    if player_sum > dealer_sum:
        return "win", 1
    return "lose", -1


def log_episode(
    csv_writer,
    episode,
    total_episodes,
    steps,
    state,
    dealer_sum,
    is_natural,
    epsilon=0,
    memory_size=0,
):
    """Log episode with complete game state"""
    player_sum, dealer_card, usable_ace = state
    outcome, reward = get_outcome(player_sum, dealer_sum, is_natural)
    game_state = get_game_state(player_sum, dealer_sum, is_natural)

    csv_writer.writerow(
        {
            "Episode": episode + 1,
            "Steps": steps,
            "Reward": reward,
            "Epsilon": epsilon,
            "Memory_Size": memory_size,
            "Outcome": outcome,
            "Player_Sum": player_sum,
            "Dealer_Sum": dealer_sum,
            "Dealer_Show_Card": dealer_card,
            "Game_State": game_state,
            "Usable_Ace": usable_ace,
            "Is_Natural": is_natural,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    print(
        f"Episode {episode + 1}/{total_episodes} - "
        f"Player: {player_sum} {'(Natural) ' if is_natural else ''}"
        f"Dealer: {dealer_sum} - "
        f"Outcome: {outcome} ({reward:+.1f})"
    )


def save_weights(agent, episode):
    """Helper function to save model weights"""
    try:
        agent.save("blackjack", f"{episode}")
        print(f"Saved weights at episode {episode}")
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
