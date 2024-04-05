from new_world import Bomberman
import pickle
from agents import QLearningAgent, RandomAgent, DQNAgent


#open pickle
pickle_path = 'pickles/qtable_qlearning_5000_episodes.pkl'

#check if pickle exists
try:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

        #agents_ = [DQNAgent(init_data_agents = data), RandomAgent()]
        agents_ = [QLearningAgent(data_to_init = data), RandomAgent()]


        args = {
                'display' : True,
                'maze_size': (2, 2),
                'nb_bombs': [1, 0], #No bomb for the random agent 
                'type_agents': ['qlearning', 'random'],
                'AGENTS' : agents_
            }
except:
    print('Pickle not found')

def main(**args):
    world = Bomberman(**args)
    _, rewards, _ = world.run()
    print('FINAL REWARD : ', rewards[0][-1])


if __name__ == "__main__":
    main(**args)