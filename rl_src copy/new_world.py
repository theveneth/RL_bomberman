import pygame
import os
import random
import time
from agents import NaiveAgent, QLearningAgent, RandomAgent, MonteCarloAgent, NNAgent, DQNAgent
from utils import load_image
import numpy as np

class Bomberman():

    def __init__(self, display = True, maze_size = (3,5), nb_bombs = [1], type_agents = ["naive"], init_data_agents = [None], bombing_range = 3, diag_bombing_range = 2, bomb_time = 3000, explosion_time = 1000, agent_move_time = 300):
        self.nb_agents = len(type_agents)
        self.nb_bombs = nb_bombs
        self.bombing_range = bombing_range
        self.diag_bombing_range = diag_bombing_range
        self.bomb_time = bomb_time
        self.explosion_time = explosion_time
        self.agent_move_time = agent_move_time
        self.explosion_time = explosion_time
        self.type_agents = type_agents
        self.display = display
        self.all_rewards = [[] for i in range(self.nb_agents)]
        self.t = 0
        #init pygame
        
        pygame.init()
        pygame.font.init()

        #Settings
        self.maze_size = maze_size
        if self.display:
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(os.path.join('images','emulogic.ttf'), 36)
            self.agent_move_timer = pygame.time.get_ticks()  # Timer to control agent movement

        #######
        #######INIT STATE
        #######STATE WILL BE A DICT OF DICT CONTAINING ALL THE DATA OF THE GAME
        #######

        #AGENT STATE DATA
        for type_agent in type_agents:
            assert type_agent in ["naive", "qlearning", "random", "montecarlo","nn", "pastawaremontecarlo", "dqn"]
        assert self.nb_agents < 5 and self.nb_agents > 0

        colors = ["_blue", "_green", "_pink", "_yellow"]
        agents_id = ["blue", "green", "pink", "yellow"]
        
        data_agents = [{'agent_x': 1, 'agent_y': 1, 'alive' : True, 'bombs_available' : self.nb_bombs[i], 'type_agent' : ag, 'agent_id' : agents_id[i]} for i,ag in enumerate(self.type_agents)]
        
        #INIT STATE BOMB DATA
        #pour l'instant, ils auront une bombe chacun.
        #data_bombs = [{'agent_id': i, 'bomb_time': None, 'bomb_x': None, 'bomb_y': None} for i in range(nb_agents)]

        #INIT STATE MAZE DATA
        size_x, size_y = maze_size
        real_maze_size = (5*size_x + 2, 3*size_y + 2)
        data_agents = self.init_pos_agents(data_agents, real_maze_size)


        # maze_layout = [
        #     ''.join(['#'] + ['    #']*size_x + ['#']),
        #     ''.join(['#'] + ['  #  ']*size_x + ['#']),
        #     ''.join(['#'] + ['#    ']*size_x + ['#']),
        # ] * size_y
        maze_layout = [
            ''.join(['#'] + ['     ']*size_x + ['#']),
            ''.join(['#'] + ['     ']*size_x + ['#']),
            ''.join(['#'] + ['     ']*size_x + ['#']),
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
        if self.display:
            agents_images = [load_image(os.path.join("images",'robot' + colors[i] + '.png')) for i in range(self.nb_agents)]
            bomb_images = [load_image(os.path.join("images",'bomb' + colors[i] + '.png')) for i in range(self.nb_agents)]
            
            self.images = {'explosion' : load_image(os.path.join("images",'explosion_2.png')), 'brick' : load_image(os.path.join("images", "brick.png"))}
            self.images['agents'] = agents_images
            self.images['bombs'] = bomb_images
        

        ###########################
        #INIT AGENTS
        ###########################
        #let's take 4 agents for now
        self.AGENTS = []

        for i,ag in enumerate(self.type_agents):
            data_to_init = init_data_agents[i]
            if ag == "naive":
                self.AGENTS.append(NaiveAgent())
            elif ag == "qlearning":
                if display :
                    print('eps set to 0')
                    self.AGENTS.append(QLearningAgent(epsilon = 0, data_to_init = data_to_init))
                else:
                    self.AGENTS.append(QLearningAgent(data_to_init = data_to_init))
            elif ag == "random":
                self.AGENTS.append(RandomAgent())
            elif ag == "montecarlo":
                if display:
                    print('eps set to 0')
                    self.AGENTS.append(MonteCarloAgent(epsilon = 0, data_to_init = data_to_init))
                else:
                    self.AGENTS.append(MonteCarloAgent(data_to_init = data_to_init))
            elif ag == "nn":
                if display:
                    print('eps set to 0')
                    self.AGENTS.append(NNAgent(epsilon = 0, model_to_init = data_to_init))
                else:
                    self.AGENTS.append(NNAgent(model_to_init = data_to_init))
            elif ag == "dqn":
                self.AGENTS.append(DQNAgent(init_data_agents = data_to_init))
                    
            
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
            new_data_agents[1]['agent_x'] = 5
            new_data_agents[1]['agent_y'] = 5

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
        
    def drop_bomb(self, next_state, agent_id, x, y):
        #print('agent_id', agent_id, ' dropped a bomb')
        new_bomb = {}
        new_bomb['agent_id'] = agent_id

        new_bomb['bomb_time'] = 10
        new_bomb['bomb_x'] = x
        new_bomb['bomb_y'] = y

        next_state["data_bombs"].append(new_bomb)

        return next_state

    def update_bombs(self, next_state, maze_layout, bomb_time):
        for bomb in next_state["data_bombs"]:
            bomb['bomb_time'] -= 1
            if bomb['bomb_time'] == 0:
                self.make_bomb_explode(bomb, maze_layout)
                #delete bomb
                next_state["data_bombs"].remove(bomb)

        return next_state

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
        new_explosion['explosion_time'] = 10

        #get explosion zone
        new_explosion['explosion_zone'] = self.get_explosion_zone(bomb['bomb_x'], bomb['bomb_y'], maze_layout)

        self.STATE["data_explosions"].append(new_explosion)
        
    def update_explosions(self, next_state, explosion_time):
        new_bombs_availables = []
        for explosion in next_state["data_explosions"]:
            explosion['explosion_time'] -= 1
            if explosion['explosion_time'] == 0:
                #reload bomb
                new_bombs_availables.append(explosion['agent_id'])
                next_state["data_explosions"].remove(explosion)
        
        #Reload bombs
        next_state = self.update_bombs_after_expl(next_state, new_bombs_availables=new_bombs_availables)
        return next_state
    
    def update_bombs_after_expl(self, next_state, new_bombs_availables):

        for agent_id in new_bombs_availables:
            #update with new bomb if agent == agent_id
            for i, agent in enumerate(next_state["data_agents"]):
                if agent["agent_id"] == agent_id:
                    next_state["data_agents"][i]["bombs_available"] += 1
        return next_state

    def get_kill_zone(self):
        return [zone for dictionnaire in self.data_explosions for zone in dictionnaire['explosion_zone']]
    
    def get_killed_agents(self, next_state, killing_zone):
        killed_agents = []
        for i,agent in enumerate(next_state["data_agents"]):
            if agent['alive']:
                if (agent['agent_x'], agent['agent_y']) in killing_zone:
                    killed_agents.append(agent['agent_id'])
                    next_state["data_agents"][i]['alive'] = False
        #debug
        # if killed_agents != []:
        #     print('killed_agents : ', killed_agents)
        return killed_agents, next_state

    def get_brick_zones(self):
            brick_zones = []
            for y, row in enumerate(self.data_maze):
                for x, tile in enumerate(row):
                    if tile == '#':
                        brick_zones.append((x, y))
            return brick_zones

    def is_action_possible(self, agent, action): 
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
        elif action == 5:
            return True
        return (x,y) not in self.walls
        
    def perform_actions(self, actions, display = True): 
        

        # Store the rewards for each agent
        rewards = [0 for i in range(len(self.AGENTS))]

        # Store the next state
        next_state = self.STATE.copy()

        # Store whether the game is over
        done = False

        # Store whether a bomb has been dropped
        dropped_bomb = [False for i in range(len(self.AGENTS))]
        # Update the position of each agent
        for i, agent in enumerate(next_state['data_agents']):
            if agent['alive']:
                # if i==0:
                #     print('agent', i, 'action', actions[i], 'possible', self.is_action_possible(agent, actions[i]))
                if self.is_action_possible(agent, actions[i]):
                    if actions[i] == 0:
                        agent['agent_x'] -= 1
                        rewards[i] += 10
                    elif actions[i] == 1:
                        agent['agent_x'] += 1
                        rewards[i] += 10
                    elif actions[i] == 2:
                        agent['agent_y'] -= 1
                        rewards[i] += 10
                    elif actions[i] == 3:
                        agent['agent_y'] += 1
                        rewards[i] += 10
                    elif actions[i] == 4:
                        if agent["bombs_available"]>=1:
                            next_state = self.drop_bomb(next_state, agent_id = agent["agent_id"],x=agent["agent_x"], y = agent["agent_y"])
                            next_state["data_agents"][i]["bombs_available"]-=1
                            #calculate the distance from the bomb to the enemy
                            for j,other_agent in enumerate(next_state['data_agents']):
                                if other_agent['agent_id'] != agent['agent_id'] and other_agent['alive']:
                                    dist = abs(other_agent['agent_x'] - agent['agent_x']) + abs(other_agent['agent_y'] - agent['agent_y'])
                                    rewards[i] += max((5-dist),0)*1000
                            
                            dropped_bomb[i] = True
                                
                    elif actions[i] == 5:
                        rewards[i] += -10
                        pass

                else :
                    rewards[i] += -150
                
         
        # get Closest enemy distance and give rewards to those who are going for the enemy
        for j, ag in enumerate(next_state['data_agents']):
            obs = self.get_observation(j, state = next_state)
            past_obs = self.get_observation(j, state = self.STATE)
            
            closest_enemy_distance = past_obs['closest_enemy_distance']
            new_closest_enemy_distance = obs['closest_enemy_distance']
            
            if new_closest_enemy_distance>3 and closest_enemy_distance>3:
                rewards[j] += (closest_enemy_distance-new_closest_enemy_distance)*100

            new_closest_bomb_distance = obs['closest_bomb_distance']
            closest_bomb_distance = past_obs['closest_bomb_distance']
            if closest_bomb_distance<5 and dropped_bomb[j]==False:
                rewards[j] += (new_closest_bomb_distance-closest_bomb_distance)*100
                #s'éloigner
            

            new_closest_explosion_distance = obs['closest_explosion_distance']
            closest_explosion_distance = past_obs['closest_explosion_distance']
            if closest_explosion_distance<5:
                rewards[j] += (new_closest_explosion_distance-closest_explosion_distance)*100



            
        #update the bombs -> explode & delete bomb if needed
        next_state = self.update_bombs(next_state = next_state, maze_layout=self.data_maze,bomb_time=self.bomb_time)
                        
        #check if a bomb has exploded and update the state
        next_state = self.update_explosions(next_state = next_state, explosion_time=self.explosion_time)

        #check if an agent has been killed and update the state, give a reward to the killer
        for explosion in next_state["data_explosions"].copy():
            expl_zone = explosion["explosion_zone"]
            agent_dropper = explosion["agent_id"]
            killed_agents, next_state = self.get_killed_agents(next_state, killing_zone=expl_zone)
            #Reward to change
            rewards_id = {"blue":0,"green":1,"pink":2,"yellow":3}
            reward_id = rewards_id[agent_dropper]
            if agent_dropper not in killed_agents:
                rewards[reward_id] += 10000 * len(killed_agents)

        #check if the game is over
        alive_agents = [agent['alive'] for agent in next_state['data_agents']]

        #give reward to the last agent alive
        if np.sum(alive_agents)<=1:
            # if np.sum(alive_agents)==1:
            #     i = alive_agents.index(1)
            #     rewards[i]+=10000
            done = True

        #attributing bad if dead 
        for i, agent in enumerate(next_state['data_agents']):
            if not agent['alive']:
                rewards[i]-=10000
            else: 
                rewards[i]+=10
           
        self.t += 1
        #print(self.t)
        if self.t >= 200:
            for i in range(len(rewards)):
                rewards[i] -= 1000
                
        if self.t >=400:
            rewards[0] -= 10000
            next_state['data_agents'][0]['alive'] = False
            done = True

        #print(rewards)
        return rewards, next_state, done 

    def get_observation(self, i, state):

        agent = state['data_agents'][i]

        #AGENT POSITION AND SELF AWARNESS
        agent_x, agent_y = agent['agent_x'], agent['agent_y']
        grid_size = 3  #local grid size (e.g., 5x5, 7x7, etc.)
        half_grid_size = grid_size // 2
        # Local grid
        local_grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        for dx in range(-half_grid_size, half_grid_size + 1):
            for dy in range(-half_grid_size, half_grid_size + 1):
                x, y = agent_x + dx, agent_y + dy
                if 0 <= x < len(state['data_maze'][0]) and 0 <= y < len(state['data_maze']):
                    if state['data_maze'][y][x] == '#':
                        local_grid[dx + half_grid_size][dy + half_grid_size] = '#'
                
        # Closest enemy distance
        closest_enemy_distance = 20
        for other_agent in state['data_agents']:
            if other_agent['agent_id'] != agent['agent_id'] and other_agent['alive']:
                dist = abs(other_agent['agent_x'] - agent_x) + abs(other_agent['agent_y'] - agent_y)
                closest_enemy_distance = min(closest_enemy_distance, dist)

        # Closest bomb distance
        closest_bomb_distance = 40
        for bomb in state['data_bombs']:
            dist = abs(bomb['bomb_x'] - agent_x) + abs(bomb['bomb_y'] - agent_y)
            closest_bomb_distance = min(closest_bomb_distance, dist)

        # Bomb and explosion timers within the local grid
        bomb_timers = []
        explosion_timers = []
        for bomb in state['data_bombs']:
            if abs(bomb['bomb_x'] - agent_x) <= half_grid_size and abs(bomb['bomb_y'] - agent_y) <= half_grid_size:
                bomb_timers.append(bomb['bomb_time'])
        
        #to change : put the distance to the closest explosion
        for explosion in state['data_explosions']:
            for ex, ey in explosion['explosion_zone']:
                if abs(ex - agent_x) <= half_grid_size and abs(ey - agent_y) <= half_grid_size:
                    explosion_timers.append(explosion['explosion_time'])
        #closest 
        closest_explosion_distance = 40
        for explosion in state['data_explosions']:
            explosion_x = explosion['explosion_zone'][0][0]
            explosion_y = explosion['explosion_zone'][0][1]
            dist = abs(explosion_x - agent_x) + abs(explosion_y - agent_y)
            closest_explosion_distance = min(closest_explosion_distance, dist)
        



        # observation = {
        #     'local_grid': local_grid,
        #     'closest_enemy_distance': closest_enemy_distance,
        #     'closest_bomb_distance': closest_bomb_distance,
        #     'bomb_timers': bomb_timers,
        #     'explosion_timers': explosion_timers,
        #     'bombs_available': agent['bombs_available'],
        #     'closest_explosion_distance': closest_explosion_distance
        # }
    
        # observation = {
        #     'agent_pos': (agent_x, agent_y),
        #     'enemy_pos': [(other_agent['agent_x'], other_agent['agent_y']) for other_agent in state['data_agents'] if other_agent['agent_id'] != agent['agent_id'] and other_agent['alive'] ],
        #     'wall_local_pos': [(agent_x + dx, agent_y + dy) for dx in range(-half_grid_size, half_grid_size + 1) for dy in range(-half_grid_size, half_grid_size + 1) if local_grid[dx + half_grid_size][dy + half_grid_size] == '#'],
        #     #if no bom, set default value
        #     'bomb_pos' : [(bomb['bomb_x'], bomb['bomb_y']) for bomb in state['data_bombs'] ],
        #     'explosion_pos' : [(ex, ey) for explosion in state['data_explosions'] for ex, ey in explosion['explosion_zone']],

        #     'local_grid': local_grid,
        #     'closest_enemy_distance': closest_enemy_distance,
        #     'closest_bomb_distance': closest_bomb_distance,
        #     'bomb_timers': bomb_timers,
        #     'explosion_timers': explosion_timers,
        #     'bombs_available': agent['bombs_available'],
        #     'closest_explosion_distance': closest_explosion_distance
        # }
            
        #get full grid, explosion zones, bomb zones, and agent zones
        agent_zone = (agent_x, agent_y)
        observation = {
            'full_grid': state['data_maze'],
            'explosion_zones': [explosion['explosion_zone'] for explosion in state['data_explosions']],
            'bomb_zones': [(bomb['bomb_x'], bomb['bomb_y']) for bomb in state['data_bombs']],
            'enemy_zones': [(enemy['agent_x'], enemy['agent_y']) for enemy in state['data_agents'] if enemy['agent_id'] != agent['agent_id'] and enemy['alive'] ], 
            'agent_zone' : agent_zone,  
            'closest_enemy_distance': closest_enemy_distance,
            'closest_bomb_distance': closest_bomb_distance,
            'bomb_timers': bomb_timers,
            'explosion_timers': explosion_timers,
            'bombs_available': agent['bombs_available'],
            'closest_explosion_distance': closest_explosion_distance
        }


        return observation

    def get_winner(self):
        alive_agents = [agent['alive'] for agent in self.STATE['data_agents']]
        if sum(alive_agents) == 1:
            return self.STATE['data_agents'][alive_agents.index(True)]['agent_id']
        else:
            return None
           
    def run(self):

        while self.running:
            
            
            #display
            if self.display:
                current_time = pygame.time.get_ticks()
                self.screen.fill((0, 0, 0))  # Fill screen with black color
                self.display_all()
                pygame.display.flip()
                time.sleep(0.1)


            # Store the actions taken by all agents
            actions = []

            for i in range(len(self.AGENTS)):
                agent = self.AGENTS[i]
                #Given the complexity of the state space, we  want to simplify it so that the agent can learn effectively

                obs_agent = self.get_observation(i, state = self.STATE)
                #print(obs_agent)
                # Select an action based on the current state and the agent's policy
                action = agent.act(obs_agent)
                actions.append(action)

            # Execute the actions of all agents and update the environment
            rewards, next_state, done = self.perform_actions(actions, self.display)

            # Update the policy of each agent based on the transition
            for i, agent in enumerate(self.AGENTS):
                self.all_rewards[i].append(rewards[i])

                obs_agent = self.get_observation(i, state = self.STATE)
                next_obs_agent = self.get_observation(i, state = next_state)

                agent.update_policy(obs_agent, actions[i], rewards[i], next_obs_agent, done)

            # Set the current state to the new state for the next iteration
            self.STATE = next_state.copy()

            #check if the game is over
            if done:
                #print(rewards)
                self.winner = self.get_winner()
                self.running = False
        

                   
        data_to_save = []
        for agent in self.AGENTS:
            if isinstance(agent, (QLearningAgent,MonteCarloAgent)): #only qlearning for now
                data_to_save.append(agent.q_table)
            elif isinstance(agent, (NNAgent)):
                data_to_save.append(agent.model)
            elif isinstance(agent, (DQNAgent)):
                data_to_save.append(agent.policy_net)
            else: 
                data_to_save.append(None)
            
        return self.winner, self.all_rewards, data_to_save

