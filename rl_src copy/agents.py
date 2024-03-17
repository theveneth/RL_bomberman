import numpy as np

class NaiveAgent:
    def __init__(self, num_actions=6, epsilon = 0.1):
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
    def __init__(self,num_actions=6):
        self.num_actions = num_actions 

       
    def act(self, obs):
        action = np.random.randint(0, self.num_actions)
        action = 5
        return action

    def update_policy(self, state, action, reward, next_state, done):
        pass


class QLearningAgent:
    def __init__(self, data_to_init = {}, num_actions=6, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.q_table = data_to_init

    def init_q_table(self, encoded_obs):
        # Initialize Q-table with zeros
        self.q_table[encoded_obs] = np.random.rand(self.num_actions)

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
        maze = observation['full_grid']
        encoded_maze = np.zeros((len(maze), len(maze[0])))
        for y, row in enumerate(maze):
            for x, tile in enumerate(row):
                if tile == '#':
                    encoded_maze[y][x] = 1
                else:
                    encoded_maze[y][x] = 0

        encoded_maze[observation['agent_zone'][1]][observation['agent_zone'][0]] = 2

        for bomb in observation['bomb_zones']:
            if observation['agent_zone'][1] == bomb[1] and  observation['agent_zone'][0] == bomb[0]:
                encoded_maze[bomb[1]][bomb[0]] = 6
            else : encoded_maze[bomb[1]][bomb[0]] = 4

        for explosion in observation['explosion_zones']:
            for tile in explosion:
                if observation['agent_zone'][1] == tile[1] and  observation['agent_zone'][0] == tile[0]:
                    encoded_maze[tile[1]][tile[0]] = 7
                else : encoded_maze[tile[1]][tile[0]] = 5
                
        
        for enemy in observation['enemy_zones']:
            encoded_maze[enemy[1]][enemy[0]] = 3

    
        return tuple(encoded_maze.flatten())
    
import numpy as np
class MonteCarloAgent:
    def __init__(self, data_to_init={}, num_actions=5, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        #self.decay = 0.99963
        self.q_table = data_to_init
  
        self.returns = {}  # To store returns for each state-action pair
        self.visits = {}  # To store number of visits for each state-action pair

    def init_q_table(self, encoded_obs):
        self.q_table[encoded_obs] = np.random.rand(self.num_actions)

    def act(self, obs):
        encoded_obs = self.encode_state(obs)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            if self.q_table.get(encoded_obs) is None:
                self.init_q_table(encoded_obs)
            action = np.argmax(self.q_table[encoded_obs])
        
        #self.epsilon *= self.decay

        return action

    def update_policy(self, obs, action, reward, next_obs, done):
        encoded_obs = self.encode_state(obs)
        encoded_next_obs = self.encode_state(next_obs)

        if self.q_table.get(encoded_obs) is None:
            self.init_q_table(encoded_obs)

        self.visits[(encoded_obs, action)] = self.visits.get((encoded_obs, action), 0) + 1

        # If the episode is done, update the return for the state-action pair
        if done:
            self.returns[(encoded_obs, action)] = self.returns.get((encoded_obs, action), 0) + reward
            # Monte Carlo update
            self.q_table[encoded_obs][action] = self.returns[(encoded_obs, action)] / self.visits[(encoded_obs, action)]
        # If the episode is not done, continue to accumulate the return
        else:
            self.returns[(encoded_obs, action)] = self.returns.get((encoded_obs, action), 0) + reward

    def encode_state(self, observation):
        maze = observation['full_grid']
        encoded_maze = np.zeros((len(maze), len(maze[0])))
        for y, row in enumerate(maze):
            for x, tile in enumerate(row):
                if tile == '#':
                    encoded_maze[y][x] = 1
                else:
                    encoded_maze[y][x] = 0

        encoded_maze[observation['agent_zone'][1]][observation['agent_zone'][0]] = 2

        for bomb in observation['bomb_zones']:
            if observation['agent_zone'][1] == bomb[1] and  observation['agent_zone'][0] == bomb[0]:
                encoded_maze[bomb[1]][bomb[0]] = 6
            else : encoded_maze[bomb[1]][bomb[0]] = 4

        for explosion in observation['explosion_zones']:
            for tile in explosion:
                if observation['agent_zone'][1] == tile[1] and  observation['agent_zone'][0] == tile[0]:
                    encoded_maze[tile[1]][tile[0]] = 7
                else : encoded_maze[tile[1]][tile[0]] = 5
                
        
        for enemy in observation['enemy_zones']:
            encoded_maze[enemy[1]][enemy[0]] = 3

    
        return tuple(encoded_maze.flatten())



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NNAgent: 
    def __init__(self, model_to_init=None, num_actions=6, learning_rate=0.1, discount_factor=0.95, epsilon=0.05):
        input_shape = 96
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.q_values = None
        if model_to_init is not None:
            self.model = model_to_init
        else :
            self.model = self.build_model(input_shape)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self, input_shape):
        model = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        return model

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            state = torch.tensor(self.encode_state(obs), dtype=torch.float32)
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
        #self.epsilon *= self.decay
        return action

    def update_policy(self, obs, action, reward, next_obs, done):
        state = torch.tensor(self.encode_state(obs), dtype=torch.float32)
        
        next_state = torch.tensor(self.encode_state(next_obs), dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        if done:
            target_q_value = reward
        else:
            target_q_value = reward + self.discount_factor * torch.max(next_q_values).item()

        q_values[action] = target_q_value

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, self.model(state))
        loss.backward()
        self.optimizer.step()

    def encode_state(self, observation):
        maze = observation['full_grid']
        encoded_maze = np.zeros((len(maze), len(maze[0])))
        for y, row in enumerate(maze):
            for x, tile in enumerate(row):
                if tile == '#':
                    encoded_maze[y][x] = 1
                else:
                    encoded_maze[y][x] = 0

        encoded_maze[observation['agent_zone'][1]][observation['agent_zone'][0]] = 2

        for bomb in observation['bomb_zones']:
            if observation['agent_zone'][1] == bomb[1] and  observation['agent_zone'][0] == bomb[0]:
                encoded_maze[bomb[1]][bomb[0]] = 6
            else : encoded_maze[bomb[1]][bomb[0]] = 4

        for explosion in observation['explosion_zones']:
            for tile in explosion:
                if observation['agent_zone'][1] == tile[1] and  observation['agent_zone'][0] == tile[0]:
                    encoded_maze[tile[1]][tile[0]] = 7
                else : encoded_maze[tile[1]][tile[0]] = 5
                
        
        for enemy in observation['enemy_zones']:
            encoded_maze[enemy[1]][enemy[0]] = 3

    
        return tuple(encoded_maze.flatten())
