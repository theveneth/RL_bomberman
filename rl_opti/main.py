from new_world import Bomberman
import pickle
from agents import QLearningAgent, RandomAgent, DQNAgent


#open pickle
with open('pickles/qtable_qlearning_100000_episodes.pkl', 'rb') as f:
    data = pickle.load(f)

#agents_ = [DQNAgent(init_data_agents = data), RandomAgent()]
agents_ = [QLearningAgent(data_to_init = data), RandomAgent()]


args = {
        'display' : True,
        'maze_size': (2, 2),
        'nb_bombs': [0, 0], #No bomb for the random agent 
        'type_agents': ['qlearning', 'random'],
        'bombing_range': 3,
        'diag_bombing_range': 2,
        'bomb_time': 3000,
        'explosion_time': 1000,
        'agent_move_time': 300,
        'AGENTS' : agents_
    }


def main(**args):
    world = Bomberman(**args)
    _, rewards, _ = world.run()
    print('FINAL REWARD : ', rewards[0][-1])
if __name__ == "__main__":
    main(**args)