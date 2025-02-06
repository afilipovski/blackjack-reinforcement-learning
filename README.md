# Blackjack Environment

[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-blue)](https://gymnasium.farama.org/environments/toy_text/blackjack/)

## Overview

Blackjack is a card game where players aim to beat the dealer by getting cards that sum closer to 21 without exceeding it. This implementation is based on Example 5.1 in [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton and Barto.

## Environment Details

### Action Space

- Type: `Discrete(2)`
- Actions:
  - `0`: Stick (stop taking cards)
  - `1`: Hit (take another card)

### Observation Space

- Type: `Tuple(Discrete(32), Discrete(11), Discrete(2))`
- Components:
  1. Player's current sum
  2. Dealer's showing card (1-10)
  3. Usable ace indicator (0 or 1)

## Game Rules

### Card Values

- Face cards (Jack, Queen, King): 10 points
- Aces: 11 points (usable) or 1 point
- Number cards (2-9): Face value

### Gameplay

1. Initial deal: Dealer gets one face-up and one face-down card; player gets two face-up cards
2. Player's turn: Choose to hit or stick until either sticking or busting
3. Dealer's turn: Reveals face-down card and must hit until sum is 17 or greater

### Rewards

- Win: +1
- Lose: -1
- Draw: 0
- Natural Blackjack: +1.5 (if natural=True) or +1 (if natural=False)

### Episode Termination

Episodes end when:

- Player busts (sum > 21)
- Player sticks
- Note: Aces are counted as 11 unless doing so would cause a bust

## Usage

```python
import gymnasium as gym
env = gym.make("Blackjack-v1")
```
