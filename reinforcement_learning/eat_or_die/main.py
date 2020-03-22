import numpy as np
import random

# So the scenario is we have an agent called Walter, he's a blob that's completely incapable of moving from his
# location, having a conveyor belt in front of him with food that passes by (maybe it's a sushi restaurant), he also
# has a reservoir of water with a bite valve on a tube hanging in front of his face like one you'd take hiking.
# his actions at any given moment are to:
# sleep, eat, drink, or do nothing, each on a scale of 0 -> 10 where when he reaches either end of the spectrum he
# kaputs

RANGES = [(0, 10), (0, 10), (0, 10)]
DECAY_RATES = [1, 1, 1]
RECOVERY_RATES = [3, 3, 3]

ACTIONS = ['eat', 'drink', 'sleep']
LIFE_TIME = 25000

EPSILON = 0.001
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.98  # how much we value future reward over current reward


def step(state, action, state_size):
    new_state = [state[0] - DECAY_RATES[0], state[1] - DECAY_RATES[1], state[2] - DECAY_RATES[2]]

    if random.uniform(0, 1) < EPSILON:
        action = random.randint(0, len(ACTIONS) - 1)

    new_state[action] = min(new_state[action] + RECOVERY_RATES[action], RANGES[action][1])

    dead = any(ele <= 0 for ele in new_state)

    if dead:
        return tuple(new_state), -1, dead, action
    else:
        return tuple(new_state), get_reward(new_state, state_size), dead, action


def get_reward(state, state_size):
    return ((state[0] / state_size[0]) +
            (state[1] / state_size[1]) +
            (state[2] / state_size[2])) / 3


def q_learning(num_episodes):
    state_size = ([(RANGES[0][1] - RANGES[0][0]) + 1] +
                  [(RANGES[1][1] - RANGES[1][0]) + 1] +
                  [(RANGES[2][1] - RANGES[2][0]) + 1])
    action_size = [len(ACTIONS)]

    # Initialize q-table values to 0
    # q_table = np.zeros(state_size + action_size)
    q_table = np.random.uniform(low=-2, high=0, size=(state_size + action_size))
    last_winning_episode = 0
    global EPSILON

    last_thousand_eps = 0
    for episode in range(num_episodes):
        current_state = (7, 7, 7)
        day = 0
        dead = False

        while day < LIFE_TIME and not dead:
            action = np.argmax(q_table[current_state])
            new_state, reward, dead, action = step(current_state, action, state_size)

            if not dead:
                max_future_q = np.max(q_table[new_state])
                current_q = q_table[current_state + (action, )]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
                q_table[current_state + (action, )] = new_q

                if day == LIFE_TIME:
                    q_table[current_state + (action, )] = 0
            else:
                q_table[current_state + (action, )] = -1
            current_state = new_state
            day += 1
        last_thousand_eps += day
        if not dead:
            print(f"It's been {episode - last_winning_episode - 1} episodes since survival!")
            last_winning_episode = episode
            EPSILON *= 0.9
        else:
            print(f"Survived: {day} days")


if __name__ == "__main__":
    q_learning(200000)
