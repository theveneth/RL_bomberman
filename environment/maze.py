
import pygame
import os
from utils import load_image


class Maze:
    def __init__(self, size_x,size_y): #real size is 5*size_x + 2, 3*size_y + 2
        # Initialize maze attributes and methods here
        
        maze_layout = [
            ''.join(['#'] + ['    #']*size_x + ['#']),
            ''.join(['#'] + ['  #  ']*size_x + ['#']),
            ''.join(['#'] + ['#    ']*size_x + ['#']),
        ] * size_y
        self.maze_layout = [''.join(['#']* len(maze_layout[0]))] + maze_layout + [''.join(['#']* len(maze_layout[0]))]

        #Load brick image
        self.brick_image = load_image(os.path.join("images", "brick.png"))
        self.brick_width, self.brick_height = self.brick_image.get_width(), self.brick_image.get_height()


    def get_brick_zones(self):
        brick_zones = []
        for y, row in enumerate(self.maze_layout):
            for x, tile in enumerate(row):
                if tile == '#':
                    brick_zones.append((x, y))
        return brick_zones
    

    # Add methods for creating, rendering, and updating the maze