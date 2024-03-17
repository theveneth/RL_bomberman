import pickle
from new_world import Bomberman
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(**args):
    num_episodes = 10000  # Number of episodes to train for
    save_qtable = True  # Whether to save the Q-table after training

    # List to store the rewards of each agent for tracking performance
    agent_rewards = [[] for _ in range(len(args['type_agents']))]
    winners = []

    for episode in tqdm(range(num_episodes), desc="Episode"):
        world = Bomberman(**args)
        winner, rewards, data_to_save = world.run()
        winners.append(winner)
        if len(winners)>10 and winners[-10:] == ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]:
            print('10 consecutive draws, stopping training')
            break

        # Store the rewards of each agent for this episode
        for i, rewards_i in enumerate(rewards):
            agent_rewards[i].append(sum(rewards_i))

        args['init_data_agents'] = data_to_save

    #plot the rewards
    for i, rewards in enumerate(agent_rewards):
        plt.plot(rewards, label=args['type_agents'][i])
        #add legend 
        
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episode for each agent")
    plt.legend()
    plt.show()

    # Save the Q-table of the QLearningAgent after training
    print(f"Winners: {winners}")

    if save_qtable :
        for data in data_to_save :
            if data is not None :
                with open(f"./pickles/qtable_Q_{num_episodes}_episodes.pkl", "wb") as f: #for qlearning only for now
                    pickle.dump(data, f)
                    print('saved in : ', f'qtable_Q_{num_episodes}_episodes.pkl')

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

    