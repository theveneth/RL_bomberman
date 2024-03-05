import pygame
import os
import random
import time
from maze import Maze
from bombs import Bombs
from agent import Agent
from utils import load_image

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

        #Init classes
        size_x, size_y = maze_size

        self.bombs_class = Bombs(bombing_range=self.bombing_range, diag_bombing_range=self.diag_bombing_range)
        self.maze_class = Maze(size_x = size_x, size_y = size_y) # real size : 5*size_x + 2, 3*size_y + 2
        self.agent_class = Agent(nb_agents, (size_x*self.maze_size[1] + 2, size_y*self.maze_size[0]+2), bombs_available = self.bombs_available ,type_agent = self.type_agent)
        
        #Get data from classes
        self.data_bombs = self.bombs_class.data_bombs #No bomb yet
        self.data_explosions = self.bombs_class.data_explosions #No explosion yet
        self.data_agents = self.agent_class.data_agents #Contains agent position, statut, image

        #Start game directly
        self.running = True

    def display_maze(self):
        for y, row in enumerate(self.maze_class.maze_layout):
            for x, tile in enumerate(row):
                if tile == '#':
                    self.screen.blit(self.maze_class.brick_image, (x * self.maze_class.brick_width, y * self.maze_class.brick_height))
    def display_agents(self):
        for agent in self.data_agents:
            if agent['alive']:
                agent_image = agent['agent_image']
                agent_width, agent_height = agent_image.get_width(), agent_image.get_height()
                agent_x, agent_y = agent['agent_x'], agent['agent_y']
                
                self.screen.blit(agent_image, ((agent_x + 0.5) * self.maze_class.brick_width - agent_width / 2, 
                                                (agent_y + 0.5) * self.maze_class.brick_height - agent_height / 2))
            
    def display_bombs(self):
        for bomb in self.data_bombs:
            bomb_image = bomb['bomb_image']
            bomb_width, bomb_height = bomb_image.get_width(), bomb_image.get_height()
            bomb_x, bomb_y = bomb['bomb_x'], bomb['bomb_y']

            self.screen.blit(bomb_image, ((bomb_x + 0.5) * self.maze_class.brick_width - bomb_width / 2,
                                        (bomb_y + 0.5) * self.maze_class.brick_height - bomb_height / 2))
    
    def display_explosions(self):
        for explosion in self.data_explosions:
            explosion_image = explosion['explosion_image']
            explosion_width, explosion_height = explosion_image.get_width(), explosion_image.get_height()
            for x,y in explosion['explosion_zone']:
                self.screen.blit(explosion_image, ((x + 0.5) * self.maze_class.brick_width - explosion_width / 2,
                                        (y + 0.5) * self.maze_class.brick_height - explosion_height / 2))
        
            

    def get_unusable_zones(self):
        #Get bomb zones
        bomb_zones = [(bomb['bomb_x'], bomb['bomb_y']) for bomb in self.data_bombs]
    
        #Get brick zones
        brick_zones = self.maze_class.get_brick_zones()
        
        #get agent zones
        #Agent zones will be added directly at the update stae : it need the positions of the other agents in live

        return bomb_zones + brick_zones 
    
    def get_alive_agents(self):
        return [agent for agent in self.data_agents if agent['alive']]
    
    
    def display_winner(self, agent_name):
        text_surface = self.font.render("Agent {} Wins!".format(agent_name), True, (255, 255, 255))  # Render the text
        text_rect = text_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))  # Center the text
        self.screen.blit(text_surface, text_rect)  # Draw the text on the screen




    def run(self):

        while self.running:
            current_time = pygame.time.get_ticks()
            self.screen.fill((0, 0, 0))  # Fill screen with black color

            #For manual control
            for event in pygame.event.get(): #No usage : agent is random.
                if event.type == pygame.QUIT:
                    self.running = False
            
            
            #This loop refreshes the maze and the agent

            ######
            #MOVEMENTS
            ######
            
            #Agents can move every 300ms and place a bomb
            if current_time - self.agent_move_timer >= self.agent_move_time:
                dropped_bombs = self.agent_class.get_dropped_bombs()
                for bomb in dropped_bombs:
                    self.bombs_class.drop_bomb(bomb['agent_id'], bomb['x'], bomb['y'])
                    self.data_bombs = self.bombs_class.data_bombs

                self.agent_class.update_all_agents(self.get_unusable_zones()) 

                self.data_agents = self.agent_class.data_agents
                self.agent_move_timer = current_time
            
            #Bombs can explode
            self.bombs_class.update_bombs(self.maze_class.maze_layout, self.bomb_time)
            self.data_bombs = self.bombs_class.data_bombs

            #Explosions last a few seconds
            new_bombs_available = self.bombs_class.update_explosions(self.explosion_time)
            for agent_id in new_bombs_available:
                self.agent_class.update_bombs_available(agent_id)
            self.data_agents = self.agent_class.data_agents
            self.data_explosions = self.bombs_class.data_explosions


            #Agents can die
            #get kill zones :
            kill_zone = self.bombs_class.get_kill_zone()
            killed_agents = self.agent_class.get_killed_agents(kill_zone)
            if len(killed_agents) > 0:
                print('KILLED AGENT(S) : ', killed_agents)

            self.data_agents = self.agent_class.data_agents

            
            #####
            #DISPLAY
            #####
            #Of course, the maze doesn't change -> let's display it
            self.display_maze()

            #display agents
            self.display_agents() 
            
            #display bombs
            self.display_bombs() 

            #display explosion
            self.display_explosions() 

            
            #UPDATE SCREEN
            pygame.display.flip()

            #CHECK IF SOMEONE HAS WOWN AND END GAME
            
            alive = self.get_alive_agents()
            if len(alive) == 1:
                winning_agent =alive[0]['agent_id']  # Get the winning agent's name
                print("AGENT", winning_agent, "WINS!")
                self.display_winner(winning_agent)  # Display the winning agent's name
                pygame.display.update()  # Update the screen
                self.running = False 

        #FREEZE WINDOW AT THE END OF THE GAME
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()  # Close pygame
                    return  # Exit the function, effectively ending the program    


        #pygame.quit()

