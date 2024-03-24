import pickle
from new_world import Bomberman
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from agents import QLearningAgent, RandomAgent, DQNAgent, DQNAgent2

                    
def train(**args):
    num_episodes = 600# Number of episodes to train for
    save_qtable = True  # Whether to save the Q-table after training
    data_to_save = None
    to_check = []

    # List to store the rewards of each agent for tracking performance
    agent_rewards = [[] for _ in range(len(args['type_agents']))]
    last_agent_rewards = [[] for _ in range(len(args['type_agents']))]
    winners = []

    AGENTS = [DQNAgent(), RandomAgent()]
    args['AGENTS'] = AGENTS
    wrs = []
    episodes = []
    for episode in tqdm(range(num_episodes), desc="Episode"):
        world = Bomberman(**args)
        #to_check.append(len(world.AGENTS[0].q_table.keys()))

        winner, rewards, data_to_save = world.run()
        winners.append(winner)
        if len(winners)>20 and winners[-15:] == ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]:
            print('15 wins in a row')
        
        if episode%20==19 and episode >=99:
            perc1 = winners[-100:].count("blue")
            perc2 = winners[-100:].count("green")
            print("\nwr: ",perc1,"%")
            print("\nlr: ",perc2,"%")
            wrs.append(perc1)
            episodes.append(episode+1)

        # Store the rewards of each agent for this episode
        for i, rewards_i in enumerate(rewards):
            agent_rewards[i].append(np.sum(rewards_i))
            last_agent_rewards[i].append(rewards_i[-1])

        #MAJ AGENTS
        args['AGENTS'] = world.AGENTS
        #print(list(world.AGENTS[0].policy_net.parameters()))
        
    #plot the rewards
    for i, rewards in enumerate(agent_rewards):
        if i==0:
            rewards_series = pd.Series(rewards)
            moving_avg = rewards_series.rolling(window=50).mean()
            plt.plot(moving_avg, label=args['type_agents'][i])
            plt.show()
            plt.plot(last_agent_rewards[i],label = args['type_agents'][i])
            plt.show()
        #add legend 
    print("wrs: ",wrs)
    plt.plot(episodes, wrs)
    plt.show()    

    # Save the Q-table of the QLearningAgent after training
    print(f"Winners: {winners[-20:]}")

    if save_qtable :
        for data in data_to_save :
            if data is not None :
                path = f"qtable_DQN_{num_episodes}_episodes.pkl"
                with open(f"./pickles/{path}", "wb") as f: #for qlearning only for now
                    pickle.dump(data, f)
                    print('saved in : ', f'qtable_DQN_{path}_episodes.pkl')

    print(to_check)
if __name__ == "__main__":
    args = {
        'display' : False,
        'maze_size': (2, 2),
        'nb_bombs': [1, 0], #No bomb for the random agent 
        'type_agents': ['dqn', 'random'],
        'bombing_range': 3,
        'diag_bombing_range': 2,
        'bomb_time': 3000,
        'explosion_time': 1000,
        'agent_move_time': 300,
    }
    train(**args)

    print('Training finished')


