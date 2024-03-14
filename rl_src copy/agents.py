import numpy as np

class NaiveAgent:
    def __init__(self, num_actions=5, epsilon = 0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        # Initialize Q-table with zeros
        self.q_table = np.zeros(num_actions)

    def act(self, state):
        # Epsilon-greedy policy: choose the best action with probability 1 - epsilon,
        # or a random action with probability epsilon
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.q_table)

        return action

    def update_policy(self, state, action, reward, next_state, done):
        # Calculate the temporal difference (TD) target
        if done:
            td_target = reward
        else:
            td_target = reward + 0.95 * np.max(self.q_table)

        # Update the Q-value for the action pair
        td_error = td_target - self.q_table[action]
        self.q_table[action] += 0.1 * td_error


class RandomAgent:
    def __init__(self,num_actions=5):
        self.num_actions = num_actions 

       
    def act(self, obs):
        action = np.random.randint(0, self.num_actions)
        return action

    def update_policy(self, state, action, reward, next_state, done):
        pass


class QLearningAgent:
    def __init__(self, data_to_init = {}, num_actions=5, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.q_table = data_to_init

    def init_q_table(self, encoded_obs):
        # Initialize Q-table with zeros
        self.q_table[encoded_obs] = np.zeros(self.num_actions)

    def act(self, obs):
        encoded_obs = self.encode_state(obs)
        # Epsilon-greedy policy: choose the best action with probability 1 - epsilon,
        # or a random action with probability epsilon
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            if self.q_table.get(encoded_obs) is None:
                self.init_q_table(encoded_obs)
            action = np.argmax(self.q_table[encoded_obs])

        return action

    def update_policy(self, obs, action, reward, next_obs, done):
        encoded_obs = self.encode_state(obs)
        encoded_next_obs = self.encode_state(next_obs)

        if self.q_table.get(encoded_next_obs) is None:
            self.init_q_table(encoded_next_obs)
        if self.q_table.get(encoded_obs) is None:
            self.init_q_table(encoded_obs)
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(self.q_table[encoded_next_obs])

        # Update the Q-value for the state-action pair
        td_error = td_target - self.q_table[encoded_obs][action]
        self.q_table[encoded_obs][action] += self.learning_rate * td_error

    def encode_state(self, observation):
        #simple obs : closest enemy, closest bomb
        encoded_state = np.array([])
        # Local grid
        encoded_state = np.append(encoded_state, observation['local_grid'])
        # Closest enemy distance
        encoded_state = np.append(encoded_state, np.round(observation['closest_enemy_distance']))

        # Closest bomb distance
        encoded_state = np.append(encoded_state, observation['closest_bomb_distance'])
        # Bomb timers
        # encoded_state = np.append(encoded_state, observation['bomb_timers'])
        # Explosion timers
        # encoded_state = np.append(encoded_state, observation['explosion_timers'])
        #Bombs available
        encoded_state = np.append(encoded_state, observation['bombs_available'])

        return tuple(encoded_state)