from new_world import Bomberman
import pickle
#open pickle
with open('pickles/qtable_mcq_1000000_episodes.pkl', 'rb') as f:
    data = pickle.load(f)

args = {
        'display' : True,
        'maze_size': (2, 2),
        'nb_bombs': [1, 0], #No bomb for the random agent 
        'type_agents': ['pastawaremontecarlo', 'random'],
        'bombing_range': 3,
        'diag_bombing_range': 2,
        'bomb_time': 3000,
        'explosion_time': 1000,
        'agent_move_time': 300,
        'init_data_agents' : [data,None] #q learning agent data, none for the random agent
    }


def main(**args):
    world = Bomberman(**args)
    world.run()

if __name__ == "__main__":
    main(**args)