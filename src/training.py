import pickle
from new_world import Bomberman
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from agents import RandomAgent, DQNAgent, QLearningAgent, MonteCarloAgent

                    
def train(**args):

    num_episodes = 5000 # Number of episodes to train for
    save_qtable = True  # Whether to save the Q-table after training
    data_to_save = None
    to_check = []
    display_plots = False

    # List to store the rewards of each agent for tracking performance
    agent_rewards = [[] for _ in range(len(args['type_agents']))]
    last_agent_rewards = [[] for _ in range(len(args['type_agents']))]
    winners = []

    first_agent = args['type_agents'][0]
    second_agent = args['type_agents'][1]
    print('Starting the training of the agent: ', first_agent, ' over ', num_episodes, ' episodes', ' against a ', second_agent, ' agent.' )
    path_pickle = f'./pickles/qtable_{first_agent}_{num_episodes}_episodes.pkl'
    
    wrs = []
    episodes = []
    for episode in tqdm(range(num_episodes), desc="Episode"):
        world = Bomberman(**args)
        #to_check.append(len(world.AGENTS[0].q_table.keys()))

        winner, rewards, data_to_save = world.run()
        winners.append(winner)
        if display_plots :
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


    #plot the rewards
    if display_plots :
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
                with open(path_pickle, "wb") as f: #for qlearning only for now
                    pickle.dump(data, f)
                    print('saved in : ', path_pickle)

    print(to_check)

if __name__ == "__main__":

    args = {
        'display' : False,
        'maze_size': (2, 2),
        'nb_bombs': [1, 0], #No bomb for the random agent 
        'type_agents': ['qlearning', 'random'],
        'AGENTS': [QLearningAgent(), RandomAgent()],
    }
    train(**args)

    print('Training finished')


