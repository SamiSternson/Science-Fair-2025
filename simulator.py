import pygame
import random
import math


# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
EARTH_GRAVITY = -9.807
FRICTION = 0.95

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Pile Simulation")
clock = pygame.time.Clock()


# Main loop
running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

pygame.quit()
