from new_world import Bomberman
args ={'nb_agents': 4, 
       'maze_size' : (3,5), 
       'bombs_available' : 1, 
       'type_agent' : 'naive', 
       'bombing_range' : 3, 
       'diag_bombing_range' : 2, 
       'bomb_time' : 3000, 
       'explosion_time' : 1000, 
       'agent_move_time' : 300}


def main(**args):
    world = Bomberman(**args)
    world.run()

if __name__ == "__main__":
    main(**args)