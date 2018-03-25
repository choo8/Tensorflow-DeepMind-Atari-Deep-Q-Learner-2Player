#!/usr/bin/env python

# This test script runs random agents for both players and dispalys screen contents using pygame

from xitari_python_interface import ALEInterface, ale_fillRgbFromPalette
import sys
import pygame
import numpy as np
from numpy.ctypeslib import as_ctypes
from dqn.game_screen import GameScreen

# Converts the palette values to RGB values


def getRgbFromPalette(ale, surface, rgb_new):
    # Environment parameters
    width = ale.ale_getScreenWidth()
    height = ale.ale_getScreenHeight()

    # Get current observations
    obs = np.zeros(width * height, dtype=np.uint8)
    n_obs = obs.shape[0]
    ale.ale_fillObs(as_ctypes(obs), width * height)

    # Get RGB values of values
    n_rgb = n_obs * 3
    rgb = np.zeros(n_rgb, dtype=np.uint8)
    ale_fillRgbFromPalette(as_ctypes(rgb), as_ctypes(obs), n_rgb, n_obs)

    # Convert uint8 array into uint32 array for pygame visualization
    for i in range(n_obs):
        # Convert current pixel into RGBA format in pygame
        cur_color = pygame.Color(int(rgb[i]), int(rgb[i + n_obs]), int(rgb[i + 2 * n_obs]))
        cur_mapped_int = surface.map_rgb(cur_color)
        rgb_new[i] = cur_mapped_int

    # Reshape and roll axis until it fits imshow dimensions
    return np.rollaxis(rgb.reshape(3, height, width), axis=0, start=3)


if(len(sys.argv) < 2):
    print("Usage ./random_agent.py <ROM_FILE_NAME>")
    sys.exit()

ale = ALEInterface(sys.argv[1].encode('utf-8'))
width = ale.ale_getScreenWidth()
height = ale.ale_getScreenHeight()

# Reset game
ale.ale_resetGame()

(display_width, display_height) = (width * 2, height * 2)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Arcade Learning Environment Random Agent Display")
pygame.display.flip()

game_surface = pygame.Surface((width, height), depth=8)

# Initialize GameScreen object for framepooling
game_screen = GameScreen()

# init clock
clock = pygame.time.Clock()

# Clear screen
screen.fill((0, 0, 0))

while not ale.ale_isGameOver():
    # Both agents perform random actions
    # Agent A : [NOOP, FIRE, RIGHT, LEFT]
    # Agent B : [NOOP, FIRE, RIGHT, LEFT]
    ale.ale_act2(np.random.choice([0, 1, 3, 4]), np.random.choice([20, 21, 23, 24]))

    # Fill buffer of game screen with current frame
    numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
    rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
    del numpy_surface

    # Print frame onto display screen
    screen.blit(pygame.transform.scale2x(game_surface), (0, 0))

    # Update the display screen
    pygame.display.flip()

    # delay to 60fps
    clock.tick(60.)

# Print out result of the episode
print("Result of Episode:")
print("A Reward:", ale.ale_getRewardA())
print("B Reward:", ale.ale_getRewardB())

# Properly close display
pygame.display.quit()
pygame.quit()
