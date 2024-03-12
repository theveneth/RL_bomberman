import pygame
import os
import random
import time
from agents import NaiveAgent
from utils import load_image
import numpy as np

class Bomberman():

    def __init__(self, nb_agents, maze_size = (3,5), bombs_available = 1, type_agent = "naive", bombing_range = 3, diag_bombing_range = 2, bomb_time = 3000, explosion_time = 1000, agent_move_time = 300):

        self.bombs_available = bombs_available
        self.bombing_range = bombing_range
        self.diag_bombing_range = diag_bombing_range
        self.bomb_time = bomb_time
        self.explosion_time = explosion_time
        self.agent_move_time = agent_move_time
        self.explosion_time = explosion_time
        self.type_agent = type_agent
        


        #init pygame
        pygame.init()
        pygame.font.init()

        #Settings
        self.maze_size = maze_size
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(os.path.join('images','emulogic.ttf'), 36)
        self.agent_move_timer = pygame.time.get_ticks()  # Timer to control agent movement

        #######
        #######INIT STATE
        #######STATE WILL BE A DICT OF DICT CONTAINING ALL THE DATA OF THE GAME
        #######

        #AGENT STATE DATA
        assert type_agent in ["naive"]
        assert nb_agents < 5 and nb_agents > 0

        colors = ["_blue", "_green", "_pink", "_yellow"]
        agents_id = ["blue", "green", "pink", "yellow"]
        
        data_agents = [{'agent_x': 1, 'agent_y': 1, 'alive' : True, 'bombs_available' : bombs_available, 'type_agent' : type_agent, 'agent_id' : agents_id[i]} for i in range(nb_agents)]
        real_maze_size = (17,17)
        data_agents = self.init_pos_agents(data_agents, real_maze_size)

        #INIT STATE BOMB DATA
        #pour l'instant, ils auront une bombe chacun.
        #data_bombs = [{'agent_id': i, 'bomb_time': None, 'bomb_x': None, 'bomb_y': None} for i in range(nb_agents)]

        #INIT STATE MAZE DATA
        size_x, size_y = maze_size
        maze_layout = [
            ''.join(['#'] + ['    #']*size_x + ['#']),
            ''.join(['#'] + ['  #  ']*size_x + ['#']),
            ''.join(['#'] + ['#    ']*size_x + ['#']),
        ] * size_y
        self.data_maze = [''.join(['#']* len(maze_layout[0]))] + maze_layout + [''.join(['#']* len(maze_layout[0]))]
        self.walls = self.get_brick_zones()
        #INIT EXPLOSION DATA
        #data_explosions = [{'agent_id' : None,'explosion_zone': [], 'explosion_time': None}]
        
        ####
        self.STATE = {'data_agents' : data_agents, 'data_bombs' : [], 'data_maze' : self.data_maze, 'data_explosions' : []}
        ####


        ###########################
        #INIT IMAGES
        ###########################
        agents_images = [load_image(os.path.join("images",'robot' + colors[i] + '.png')) for i in range(nb_agents)]
        bomb_images = [load_image(os.path.join("images",'bomb' + colors[i] + '.png')) for i in range(nb_agents)]
        
        self.images = {'explosion' : load_image(os.path.join("images",'explosion_2.png')), 'brick' : load_image(os.path.join("images", "brick.png"))}
        self.images['agents'] = agents_images
        self.images['bombs'] = bomb_images
        

        ###########################
        #INIT AGENTS
        ###########################
        #let's take 4 agents for now
        self.AGENTS = [NaiveAgent() for i in range(nb_agents)]
        
        #Start game directly
        self.running = True

    def init_pos_agents(self, data_agents, maze_real_size):
        new_data_agents = data_agents.copy()
        nb_agents = len(new_data_agents)
    
        #Init position of agents
        if nb_agents == 1:
            new_data_agents[0]['agent_x'] = 1
            new_data_agents[0]['agent_y'] = 1

        if nb_agents == 2:
            new_data_agents[0]['agent_x'] = 1
            new_data_agents[0]['agent_y'] = 1
            new_data_agents[1]['agent_x'] = maze_real_size[0] - 2
            new_data_agents[1]['agent_y'] = maze_real_size[1] - 2

        if nb_agents == 3:
            new_data_agents[0]['agent_x'] = 1
            new_data_agents[0]['agent_y'] = 1
            new_data_agents[1]['agent_x'] = maze_real_size[0] - 2
            new_data_agents[1]['agent_y'] = maze_real_size[1] - 2
            new_data_agents[2]['agent_x'] = 1
            new_data_agents[2]['agent_y'] = maze_real_size[1] - 3 #There's a wall in this position at -2

        if nb_agents == 4:
            new_data_agents[0]['agent_x'] = 1
            new_data_agents[0]['agent_y'] = 1
            new_data_agents[1]['agent_x'] = maze_real_size[0] - 2
            new_data_agents[1]['agent_y'] = maze_real_size[1]- 2
            new_data_agents[2]['agent_x'] = 1
            new_data_agents[2]['agent_y'] = maze_real_size[1] - 3 #There's a wall in this position at -2
            new_data_agents[3]['agent_x'] = maze_real_size[0] - 2
            new_data_agents[3]['agent_y'] = 1

        return new_data_agents

    def display_maze(self):
        maze = self.STATE['data_maze']
        brick_width, brick_height = self.images['brick'].get_width(), self.images['brick'].get_height()

        for y, row in enumerate(maze):
            for x, tile in enumerate(row):
                if tile == '#':
                    self.screen.blit(self.images['brick'], (x * brick_width, y * brick_height))
                
    def display_agents(self):
        brick_width, brick_height = self.images['brick'].get_width(), self.images['brick'].get_height()
        
        agents = self.STATE['data_agents']
        for i in range(len(agents)):
            agent = agents[i]
            if agent['alive']:
                agent_image = self.images['agents'][i]
                agent_width, agent_height = agent_image.get_width(), agent_image.get_height()
                agent_x, agent_y = agent['agent_x'], agent['agent_y']
                
                self.screen.blit(agent_image, ((agent_x + 0.5) * brick_width - agent_width / 2, 
                                                (agent_y + 0.5) * brick_height - agent_height / 2))
            
    def display_bombs(self):
        brick_width, brick_height = self.images['brick'].get_width(), self.images['brick'].get_height()
        bombs = self.STATE['data_bombs']
        for i in range(len(bombs)):
            bomb = bombs[i]
            bomb_image = self.images['bombs'][0] #####
            bomb_width, bomb_height = bomb_image.get_width(), bomb_image.get_height()
            bomb_x, bomb_y = bomb['bomb_x'], bomb['bomb_y']

            self.screen.blit(bomb_image, ((bomb_x + 0.5) * brick_width - bomb_width / 2,
                                        (bomb_y + 0.5) * brick_height - bomb_height / 2))
            
    def display_explosions(self):
        brick_width, brick_height = self.images['brick'].get_width(), self.images['brick'].get_height()
        for explosion in self.STATE['data_explosions']:
            explosion_image = self.images['explosion']
            explosion_width, explosion_height = explosion_image.get_width(), explosion_image.get_height()
            for x,y in explosion['explosion_zone']:
                self.screen.blit(explosion_image, ((x + 0.5) * brick_width - explosion_width / 2,
                                        (y + 0.5) * brick_height - explosion_height / 2))
           
    def display_all(self):
        self.display_maze()
        self.display_agents()
        self.display_bombs()
        self.display_explosions()

    ##########BOMBS
        
    def drop_bomb(self, agent_id, x, y):
        new_bomb = {}
        new_bomb['agent_id'] = agent_id

        new_bomb['bomb_time'] = pygame.time.get_ticks()
        new_bomb['bomb_x'] = x
        new_bomb['bomb_y'] = y

        self.STATE["data_bombs"].append(new_bomb)

    def update_bombs(self, maze_layout, bomb_time):
        for bomb in self.STATE["data_bombs"]:
            if bomb['bomb_time'] is not None:
                if pygame.time.get_ticks() - bomb['bomb_time'] > bomb_time:
                    self.make_bomb_explode(bomb, maze_layout)
                    #delete bomb
                    self.STATE["data_bombs"].remove(bomb)

    def get_explosion_zone(self, bomb_x, bomb_y, maze_layout):
        bombing_range = self.bombing_range
        diag_bombing_range = self.diag_bombing_range

        def valid_zone(x, y):
            return 0 <= x < len(maze_layout[0]) and 0 <= y < len(maze_layout) and maze_layout[y][x] != '#'

        zone = [(bomb_x, bomb_y)]

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]

        for dx, dy in directions:
            current_range = diag_bombing_range if dx != 0 and dy != 0 else bombing_range #if diag, use diag range, else usual
            
            for i in range(1, current_range):
                new_x, new_y = bomb_x + dx * i, bomb_y + dy * i
                if valid_zone(new_x, new_y):
                    zone.append((new_x, new_y))
                else:
                    break
        
        return zone
        
    def make_bomb_explode(self, bomb, maze_layout):
        new_explosion = {}
        #put agent_id to reload bomb when explosion is done in update_explosion
        new_explosion['agent_id'] = bomb['agent_id']
        new_explosion['explosion_time'] = pygame.time.get_ticks()

        #get explosion zone
        new_explosion['explosion_zone'] = self.get_explosion_zone(bomb['bomb_x'], bomb['bomb_y'], maze_layout)

        self.STATE["data_explosions"].append(new_explosion)
        
    def update_explosions(self, explosion_time):
        new_bombs_availables = []
        for explosion in self.STATE["data_explosions"]:
            if pygame.time.get_ticks() - explosion['explosion_time'] > explosion_time:

                new_bombs_availables.append(explosion['agent_id'])
                self.STATE["data_explosions"].remove(explosion)
        
        #Reload bombs
        self.update_bombs_after_expl(new_bombs_availables=new_bombs_availables)

    
    def update_bombs_after_expl(self, new_bombs_availables):
        for agent_id in new_bombs_availables:
            #update with new bomb if agent == agent_id
            for i, agent in enumerate(self.STATE["data_agents"]):
                if agent["agent_id"] == agent_id:
                    self.STATE["data_agents"][i]["bombs_available"] += 1


        
    def get_kill_zone(self):
        return [zone for dictionnaire in self.data_explosions for zone in dictionnaire['explosion_zone']]
    
    def get_killed_agents(self,killing_zone):
        killed_agents = []
        for i,agent in enumerate(self.STATE["data_agents"]):
            if agent['alive']:
                if (agent['agent_x'], agent['agent_y']) in killing_zone:
                    killed_agents.append(agent['agent_id'])
                    self.STATE["data_agents"][i]['alive'] = False
        if killed_agents != []:
            print(self.STATE["data_agents"])
        return killed_agents

    def get_brick_zones(self):
            brick_zones = []
            for y, row in enumerate(self.data_maze):
                for x, tile in enumerate(row):
                    if tile == '#':
                        brick_zones.append((x, y))
            return brick_zones

    def is_action_possible(self, agent, action): #TODO
        #si il n'y a pas de wall, mob ou bombe là ou il veut aller 
        #(dans un premier temps on peut considérer les mob/bomb comme transparent,
        # et ajouter les blocages après)
        #s'il veut poser une bombe, en a il déjà posé une ?
        x,y = agent["agent_x"],agent["agent_y"]
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1
        elif action == 4:
            return agent["bombs_available"]>=1
        return (x,y) not in self.walls
        
    def perform_actions(self, actions): #TODO
        time.sleep(2)
        # Store the rewards for each agent
        rewards = [0] * len(self.AGENTS)

        # Store the next state
        next_state = self.STATE.copy()

        # Store whether the game is over
        done = False

        # Update the position of each agent
        for i, agent in enumerate(next_state['data_agents']):
            if agent['alive']:
                if self.is_action_possible(agent, actions[i]):
                    print(agent["agent_id"])
                    if actions[i] == 0:
                        agent['agent_x'] -= 1
                    elif actions[i] == 1:
                        agent['agent_x'] += 1
                    elif actions[i] == 2:
                        agent['agent_y'] -= 1
                    elif actions[i] == 3:
                        agent['agent_y'] += 1
                    elif actions[i] == 4:
                        #TODO : poser une bombe
                        self.drop_bomb(agent_id = agent["agent_id"],x=agent["agent_x"], y = agent["agent_y"])
                        self.STATE["data_agents"][i]["bombs_available"]-=1
                else:
                    rewards[i] = -1

        #update the bombs -> explode & delete bomb if needed
        self.update_bombs(maze_layout=self.data_maze,bomb_time=3000)
                        
        #check if a bomb has exploded and update the state
        self.update_explosions(explosion_time=3000)

        #check if an agent has been killed and update the state, give a reward to the killer
        for explosion in self.STATE["data_explosions"]:
            expl_zone = explosion["explosion_zone"]
            agent_dropper = explosion["agent_id"]
            killed_agents = self.get_killed_agents(killing_zone=expl_zone)
            #Reward to change
            rewards_id = {"blue":0,"green":1,"pink":2,"yellow":3}
            reward_id = rewards_id[agent_dropper]
            if agent_dropper not in killed_agents:
                rewards[reward_id] = 10 * len(killed_agents)
        #check if the game is over
        alive_agents = [1,1,1,1]
        for i, agent in enumerate(next_state['data_agents']):
            if not agent['alive']:
                if alive_agents[i]==1:
                    alive_agents[i]=0
                    rewards[i]-=10
        if np.sum(alive_agents) <=1:
            i = alive_agents.index(1)
            rewards[i]+=50
            done = True

        
        return rewards, next_state, done 


    def run(self):

        while self.running:
            current_time = pygame.time.get_ticks()
            self.screen.fill((0, 0, 0))  # Fill screen with black color

            #For manual control
            for event in pygame.event.get(): #No usage : agent is random.
                if event.type == pygame.QUIT:
                    self.running = False
            
            #display
            self.display_all()
            pygame.display.flip()


            # Store the actions taken by all agents
            actions = []

            for agent in self.AGENTS:
                # Select an action based on the current state and the agent's policy
                action = agent.act(self.STATE)
                actions.append(action)

            # Execute the actions of all agents and update the environment
            rewards, next_state, done = self.perform_actions(actions)

            # Update the policy of each agent based on the transition
            for i, agent in enumerate(self.AGENTS):
                agent.update_policy(self.STATE, actions[i], rewards[i], next_state, done)

            # Set the current state to the new state for the next iteration
            self.STATE = next_state

            #check if the game is over
            if done:
                print(self.STATE["data_agents"])
                self.running = False


        #FREEZE WINDOW AT THE END OF THE GAME
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()  # Close pygame
                    return  # Exit the function, effectively ending the program    


        #pygame.quit()

