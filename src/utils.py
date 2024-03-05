import pygame

def load_image(image_name):
    image = pygame.image.load(image_name).convert_alpha()
    return image
