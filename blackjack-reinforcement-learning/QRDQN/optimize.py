import optuna
import gymnasium as gym
from gymnasium.spaces import Discrete
from sb3_contrib import QRDQN
from stable_baselines3.common.evaluation import evaluate_policy

class BlackjackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Discrete(32 * 11 * 2)

    def observation(self, obs):
        player_sum, dealer_card, usable_ace = obs
        return (player_sum * 11 * 2) + (dealer_card * 2) + usable_ace

def objective(trial):
    env = BlackjackObsWrapper(gym.make("Blackjack-v1"))

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    buffer_size = trial.suggest_int("buffer_size", 10_000, 500_000, step=10_000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.1, 0.99, step=0.05)
    train_freq = trial.suggest_int("train_freq", 1, 20)
    target_update_interval = trial.suggest_int("target_update_interval", 500, 5000, step=500)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.0001, 0.1, log=True)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1, step=0.01)

    model = QRDQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=0,
    )

    model.learn(total_timesteps=5000) 
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100) 

    return mean_reward 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print("Best parameters:", study.best_params)
pass