import pickle
from new_world import Bomberman
import numpy as np
from tqdm import tqdm

def train(**args):
    num_episodes = 100000  # Number of episodes to train for
    save_qtable = True  # Whether to save the Q-table after training

    # List to store the rewards of each agent for tracking performance
    agent_rewards = [[] for _ in range(len(args['type_agents']))]
    winners = []

    for episode in tqdm(range(num_episodes), desc="Episode"):
        world = Bomberman(**args)
        winner, rewards, data_to_save = world.run()
        winners.append(winner)

        # Store the rewards of each agent for this episode
        for i, rewards_i in enumerate(rewards):
            agent_rewards[i].append(rewards_i)

        args['init_data_agents'] = data_to_save

        
    # Save the Q-table of the QLearningAgent after training
    print(f"Winners: {winners}")

    if save_qtable :
        for data in data_to_save :
            if data is not None :
                with open(f"./pickles/qtable_{num_episodes}_episodes.pkl", "wb") as f: #for qlearning only for now
                    pickle.dump(data, f)

if __name__ == "__main__":
    args = {
        'display' : False,
        'maze_size': (2, 2),
        'nb_bombs': [1, 0], #No bomb for the random agent 
        'type_agents': ['qlearning', 'random'],
        'bombing_range': 3,
        'diag_bombing_range': 2,
        'bomb_time': 3000,
        'explosion_time': 1000,
        'agent_move_time': 300,
        'init_data_agents' : [{},None] #q learning agent data, none for the random agent
    }
    train(**args)

    print('Training finished')

    


#Y'A TJR UN ECRAN NOIR QUAND ON LANCE LE TRAINING, C'EST NORMAL ET DU A PYGAME.