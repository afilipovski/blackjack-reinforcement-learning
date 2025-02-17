import csv
import os
import matplotlib.pyplot as plt


def sum_rewards(file_path):
    with open(file_path, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        return sum(float(row["Reward"]) for row in csv_reader)


def sum_rewards_in_folder(folder_path):
    rewards = []
    file_names = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            reward_sum = sum_rewards(file_path)
            rewards.append(reward_sum)
            file_names.append(file_name)
            print(f"Processed {file_name}: Current Total = {reward_sum}")

    return file_names, rewards


def plot_rewards(file_names, rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(file_names, rewards, marker="o", linestyle="-")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("CSV Files")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards by File")
    plt.grid()
    plt.show()


folder_path = "logs/testing"
a, b = sum_rewards_in_folder(folder_path)
print("Final Total Reward:", a)
plot_rewards(a, b)
