import pygame
import os
import random


class World():
    def __init__(self):
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.running = True
        self.agent_move_timer = pygame.time.get_ticks()  # Timer to control agent movement
    def load_image(self, image_name):
        image = pygame.image.load(image_name).convert()
        return image
    
    def is_valid_move(self, x, y, maze_layout):
        return maze_layout[y][x] != '#'

    def run(self):
        ### MAZE
        brick_image = self.load_image(os.path.join("images", "brick.png"))
        brick_width, brick_height = brick_image.get_width(), brick_image.get_height()
        counting = 5
        maze_layout = [
            ''.join(['#'] + ['    #']*counting + ['#']),
            ''.join(['#'] + ['  #  ']*counting + ['#']),
            ''.join(['#'] + ['#    ']*counting + ['#']),
        ] * 5
        maze_layout = [''.join(['#']* len(maze_layout[0]))] + maze_layout + [''.join(['#']* len(maze_layout[0]))]

        ### AGENT

        agent_image = self.load_image(os.path.join("images", "robot_blue.png"))
        agent_width, agent_height = agent_image.get_width(), agent_image.get_height()
        agent_x, agent_y = 1, 1 #no brick here
        
        while self.running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.screen.fill((0, 0, 0))  # Fill screen with black color
            
            #BUILD MAZE

            for y, row in enumerate(maze_layout):
                for x, tile in enumerate(row):
                    if tile == '#':
                        self.screen.blit(brick_image, (x * brick_width, y * brick_height))
            
            self.screen.blit(agent_image, ((agent_x + 0.5) * brick_width - agent_width / 2, (agent_y + 0.5) * brick_height - agent_height / 2))
            pygame.display.flip()
            self.clock.tick(60)

            #AGENT
            # Random movement logic
            current_time = pygame.time.get_ticks()
            if current_time - self.agent_move_timer >= 300:  # Move the agent every 1 second
                possible_moves = [(agent_x + 1, agent_y), (agent_x - 1, agent_y), (agent_x, agent_y + 1), (agent_x, agent_y - 1)]
                random.shuffle(possible_moves)
                for move in possible_moves:
                    if self.is_valid_move(move[0], move[1], maze_layout):
                        agent_x, agent_y = move
                        break
                self.agent_move_timer = current_time
            
        pygame.quit()


if __name__ == "__main__":
    world = World()
    world.run()