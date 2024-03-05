import pygame
import os
from utils import load_image
import random

class Agent:
    def __init__(self, nb_agents, maze_real_size, bombs_available = 1,type_agent = "naive"):
        # Initialize maze attributes and methods here
        assert type_agent in ["naive"]
        assert nb_agents < 5 and nb_agents > 0

        colors = ["_blue", "_green", "_pink", "_yellow"]
        agents_id = ["blue", "green", "pink", "yellow"]

        data_agents = [{'agent_x': 1, 'agent_y': 1, 'agent_image': load_image(os.path.join("images", "robot" + colors[i] + ".png")), 'alive' : True, 'bombs_available' : bombs_available, 'type_agent' : type_agent, 'agent_id' : agents_id[i]} for i in range(nb_agents)]

        #Init position of agents
        if nb_agents == 1:
            data_agents[0]['agent_x'] = 1
            data_agents[0]['agent_y'] = 1

        if nb_agents == 2:
            data_agents[0]['agent_x'] = 1
            data_agents[0]['agent_y'] = 1
            data_agents[1]['agent_x'] = maze_real_size[0] - 2
            data_agents[1]['agent_y'] = maze_real_size[1] - 2

        if nb_agents == 3:
            data_agents[0]['agent_x'] = 1
            data_agents[0]['agent_y'] = 1
            data_agents[1]['agent_x'] = maze_real_size[0] - 2
            data_agents[1]['agent_y'] = maze_real_size[1] - 2
            data_agents[2]['agent_x'] = 1
            data_agents[2]['agent_y'] = maze_real_size[1] - 3 #There's a wall in this position at -2

            #DEBUG : MAKE THEM APPEAR AT THE TOP LEFT
            # data_agents[0]['agent_x'] = 1
            # data_agents[0]['agent_y'] = 1
            # data_agents[1]['agent_x'] = 2
            # data_agents[1]['agent_y'] = 1
            # data_agents[2]['agent_x'] = 1
            # data_agents[2]['agent_y'] = 2

        if nb_agents == 4:
            data_agents[0]['agent_x'] = 1
            data_agents[0]['agent_y'] = 1
            data_agents[1]['agent_x'] = maze_real_size[0] - 2
            data_agents[1]['agent_y'] = maze_real_size[1] - 2
            data_agents[2]['agent_x'] = 1
            data_agents[2]['agent_y'] = maze_real_size[1] - 3 #There's a wall in this position at -2
            data_agents[3]['agent_x'] = maze_real_size[0] - 2
            data_agents[3]['agent_y'] = 1

        #add width, height
        for agent in data_agents:
            agent['agent_width'], agent['agent_height'] = agent['agent_image'].get_width(), agent['agent_image'].get_height()


        self.data_agents = data_agents

    
    def update_all_agents(self, unusable_zones):
        for agent in self.data_agents:
            if agent['alive']:
                if agent['type_agent'] == "naive":
                    self.update_agent_naive(agent, unusable_zones)
                    


    def update_agent_naive(self, agent, unusable_zones):
        agent_x, agent_y = agent['agent_x'], agent['agent_y']
        #add all agent position in unsable zones, except the current agent. In this strategy, the first agent to move has the priority.
        unusable_zones_adapted = unusable_zones + [(a['agent_x'], a['agent_y']) for a in self.data_agents if a['alive'] and a['agent_id'] != agent['agent_id']]

        possible_moves = [(agent_x + 1, agent_y), (agent_x - 1, agent_y), (agent_x, agent_y + 1), (agent_x, agent_y - 1), (agent_x, agent_y)]
        random.shuffle(possible_moves)
        for move in possible_moves:
            if move not in unusable_zones_adapted:
                #UPDATE POSITION
                agent['agent_x'], agent['agent_y'] = move
                #not sure about this (pb memory ?)
                break
        
    def get_dropped_bombs(self):
        bombs_to_drop = []
        for agent in self.data_agents:
            if agent['alive']:
                if agent['type_agent'] == "naive":
                    bombs_to_drop = bombs_to_drop + self.drop_bomb_naive(agent)
        return bombs_to_drop
    

    def drop_bomb_naive(self, agent):
        if agent['bombs_available'] > 0:
            if random.random() < 0.1: #10% of chance to drop a bomb
                agent['bombs_available'] -= 1
                return [{'agent_id' : agent['agent_id'], 'x' : agent['agent_x'], 'y' : agent['agent_y']}]
            
        return []
    

    def get_killed_agents(self, killing_zone):
        killed_agents = []
        for agent in self.data_agents:
            if agent['alive']:
                if (agent['agent_x'], agent['agent_y']) in killing_zone:
                    killed_agents.append(agent['agent_id'])
                    agent['alive'] = False
        if killed_agents != []:
            print(self.data_agents)
        return killed_agents
    
    def update_bombs_available(self, agent_id):
        for agent in self.data_agents:
            if agent['agent_id'] == agent_id:
                agent['bombs_available'] += 1
        
    
    