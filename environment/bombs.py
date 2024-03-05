
import pygame
import os
from utils import load_image

class Bombs:
    def __init__(self, bombing_range = 3, diag_bombing_range = 2):
        self.bombing_range = bombing_range
        self.diag_bombing_range = diag_bombing_range
        # Initialize maze attributes and methods here
        # EXAMPLE data = [{'agent_id': 0, 'bomb_time': None, 'bomb_x': None, 'bomb_y': None, 'kill_zones': [], 'display_explosion': [], 'bomb_image': None} for i in range(nb_agents)]
        self.data_bombs = []


        #EXAMPLE EXPLOSIONS
        #{'explosion_time': 0, 'explosion_zone', 'explosion_image': None, 'explosion_width': 0, 'explosion_height': 0}
        self.data_explosions = []

    def get_bomb_image(self,agent_id):
        bombs_id = {"blue" :"bomb_blue.png", "green" : "bomb_green.png","pink":"bomb_pink.png", "yellow": "bomb_yellow.png"}
        
        return bombs_id[agent_id]
    
    def drop_bomb(self, agent_id, x, y):
        new_bomb = {}
        new_bomb['agent_id'] = agent_id

        new_bomb['bomb_time'] = pygame.time.get_ticks()
        new_bomb['bomb_x'] = x
        new_bomb['bomb_y'] = y
        new_bomb['kill_zones'] = [(x, y)]
        new_bomb['display_explosion'] = []

        new_bomb['bomb_image'] = load_image(os.path.join("images", self.get_bomb_image(agent_id))) 
        new_bomb['explosion_image'] = load_image(os.path.join("images", "explosion_2.png"))

        new_bomb['explosion_width'], new_bomb['explosion_height'] = new_bomb['explosion_image'].get_width(), new_bomb['explosion_image'].get_height()
        new_bomb['bomb_width'], new_bomb['bomb_height'] = new_bomb['bomb_image'].get_width(), new_bomb['bomb_image'].get_height()

        self.data_bombs.append(new_bomb)

    def update_bombs(self, maze_layout, bomb_time):
        for bomb in self.data_bombs:
            if bomb['bomb_time'] is not None:
                if pygame.time.get_ticks() - bomb['bomb_time'] > bomb_time:
                    self.make_bomb_explode(bomb, maze_layout)
                    #delete bomb
                    self.data_bombs.remove(bomb)

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
        new_explosion['agent_id'] = bomb['agent_id']
        new_explosion['explosion_time'] = pygame.time.get_ticks()
        new_explosion['center_x'], new_explosion['center_y'] = bomb['bomb_x'], bomb['bomb_y']

        #get explosion zone
        new_explosion['explosion_zone'] = self.get_explosion_zone(bomb['bomb_x'], bomb['bomb_y'], maze_layout)

        new_explosion['explosion_image'] = bomb['explosion_image']
        new_explosion['explosion_width'], new_explosion['explosion_height'] = bomb['explosion_width'], bomb['explosion_height']

        self.data_explosions.append(new_explosion)
        
    def update_explosions(self, explosion_time):
        new_bombs_availables = []
        for explosion in self.data_explosions:
            if pygame.time.get_ticks() - explosion['explosion_time'] > explosion_time:

                new_bombs_availables.append(explosion['agent_id'])
                self.data_explosions.remove(explosion)

        return new_bombs_availables

        
    def get_kill_zone(self):
        return [zone for dictionnaire in self.data_explosions for zone in dictionnaire['explosion_zone']]

    
