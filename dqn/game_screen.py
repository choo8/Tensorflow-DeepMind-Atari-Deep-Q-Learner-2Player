import pygame
import numpy as np

# A port of the GameScreen class from the Lua implementation in alewrap

# The GameScreen class is designed to efficiently combine a fixed number
# of consecutive images of identical dimensions coming in a sequence

# Several Atari games (e.g. Space Invaders, Ms. Pacman, Alien, etc.) contain
# blinking sprites, or important game objects which are drawn every other frame
# for technical reasons (e.g. asteroids in the eponymous game). Using single
# frames as state, even when sampled at a fixed interval, can miss many such
# elements, with possibly severe consequences during gameplay (e.g. bullets
# blink in most games). Pooling over consecutive frames reduces the risk of
# missing such game elements.

# The GameScreen class allows users to `paint` individual frames on a simulated
# screen and then `grab` the mean/max/etc of the last N painted frames. The
# default configuration will return the mean over the last two consecutive frames.


class GameScreen:
    # Create a game screen with an empty frame buffer.
    def __init__(self):
        self.reset()

    def clear(self):
        if self.frameBuffer is not None:
            self.frameBuffer = np.zeros(self.frameBuffer.shape, dtype=np.uint8)

        self.lastIndex = 0

    def reset(self):
        self.frameBuffer = None
        self.poolBuffer = None
        self.lastIndex = 0
        self.bufferSize = 2

    # Use the frame buffer to capture screen.
    def grab(self):
        assert self.frameBuffer is not None
        return np.amax(self.frameBuffer, axis=0)

    # Adds a frame at the top of the buffer.
    def paint(self, frame):
        assert np.any(frame)
        if self.frameBuffer is None:
            # Set up framebuffer
            dims = (self.bufferSize,) + frame.shape
            self.frameBuffer = np.zeros(dims, dtype=np.uint8)
            self.clear()

        self.frameBuffer[self.lastIndex] = frame
        self.lastIndex = (self.lastIndex + 1) % self.bufferSize
