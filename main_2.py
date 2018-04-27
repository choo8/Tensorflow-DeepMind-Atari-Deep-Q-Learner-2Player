from __future__ import print_function
import random
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
import os

from dqn.agent import Agent
from dqn.agent2 import Agent2

from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config
from xitari_python_interface import ALEInterface, ale_fillRgbFromPalette
import pygame
from numpy.ctypeslib import as_ctypes
import sys
from dqn.game_screen import GameScreen
from dqn.scale import scale_image

from dqn.history import History

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '5/6', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

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


def main(_):

  with tf.Session() as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    else:
      env = GymEnvironment(config)

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    roms = 'roms/Pong2PlayerVS.bin'
    ale = ALEInterface(roms.encode('utf-8'))
    width = ale.ale_getScreenWidth()
    height = ale.ale_getScreenHeight()
    game_screen = GameScreen()
    ale.ale_resetGame()
    (display_width, display_height) = (width * 2, height * 2)

    pygame.init()
    screen_ale = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Arcade Learning Environment Random Agent Display")
    pygame.display.flip()

    game_surface = pygame.Surface((width, height), depth=8)
    clock = pygame.time.Clock()

    # Clear screen
    screen_ale.fill((0, 0, 0))

    agent = Agent(config, env, sess, 'A')
    agent2 = Agent2(config, env, sess, 'B')

    if FLAGS.is_train:
      start_epoch = agent.epoch_op.eval()
      start_step = agent.step_op.eval()
      start_time = time.time()

      # Loop for epochs
      for agent.epoch in range(start_epoch, agent.max_epoch):
        agent2.epoch = agent.epoch

        # Initialize information of gameplay
        num_game, agent.update_count, agent2.update_count, ep_rewardA, ep_rewardB = 0, 0, 0, 0., 0.
        total_rewardA, total_rewardB, agent.total_loss, agent2.total_loss, agent.total_q, agent2.total_q = 0., 0., 0., 0., 0., 0.
        max_avg_ep_rewardA, max_avg_ep_rewardB = 0, 0
        ep_rewardsA, ep_rewardsB, actionsA, actionsB = [], [], [], []

        # Get first frame of gameplay
        numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
        rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
        del numpy_surface        
        game_screen.paint(rgb)
        pooled_screen = game_screen.grab()
        scaled_pooled_screen = scale_image(pooled_screen)

        # Add first frame of gameplay into both agents' replay history
        for _ in range(agent.history_length):
          agent.history.add(scaled_pooled_screen)
          agent2.history.add(scaled_pooled_screen)

        # Loop for training iterations
        for agent.step in tqdm(range(start_step, agent.max_step), ncols=70, initial=start_step):
          agent2.step = agent.step

          # End of burn in period, start to learn from frames
          if agent.step == agent.learn_start:
            num_game, agent.update_count, agent2.update_count, ep_rewardA, ep_rewardB = 0, 0, 0, 0., 0.
            total_rewardA, total_rewardB, agent.total_loss, agent2.total_loss, agent.total_q, agent2.total_q = 0., 0., 0., 0., 0., 0.
            max_avg_ep_rewardA, max_avg_ep_rewardB = 0, 0
            ep_rewardsA, ep_rewardsB, actionsA, actionsB = [], [], [], []
          
          # 1. predict
          action1 = agent.predict(agent.history.get())
          action2 = agent2.predict(agent2.history.get())

          # 2. act
          ale.ale_act2(action1, action2)
          terminal = ale.ale_isGameOver()
          # End of end epoch, finish up training so that game statistics can be collected without training data being messed up
          if agent.step == agent.max_step - 1:
            terminal = True
          rewardA = ale.ale_getRewardA()
          rewardB = ale.ale_getRewardB()
          
          # Fill buffer of game screen with current frame
          numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
          rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
          del numpy_surface        
          game_screen.paint(rgb)
          pooled_screen = game_screen.grab()
          scaled_pooled_screen = scale_image(pooled_screen)
          agent.observe(scaled_pooled_screen, rewardA, action1, terminal)
          agent2.observe(scaled_pooled_screen, rewardB, action2, terminal)

          # Print frame onto display screen
          screen_ale.blit(pygame.transform.scale2x(game_surface), (0, 0))

          # Update the display screen
          pygame.display.flip()

          # Check if current episode ended
          if terminal:
            ale.ale_resetGame()
            terminal = ale.ale_isGameOver()
            rewardA = ale.ale_getRewardA()
            rewardB = ale.ale_getRewardB()
            numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)

            rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
            del numpy_surface        
            game_screen.paint(rgb)
            pooled_screen = game_screen.grab()
            scaled_pooled_screen = scale_image(pooled_screen)

            # End of an episode
            num_game += 1
            ep_rewardsA.append(ep_rewardA)
            ep_rewardsB.append(ep_rewardB)
            ep_rewardA = 0.
            ep_rewardB = 0.
          else:
            ep_rewardA += rewardA
            ep_rewardB += rewardB

          actionsA.append(action1)
          actionsB.append(action2)
          total_rewardA += rewardA
          total_rewardB += rewardB

          # Do a test to get statistics so far
          if agent.step >= agent.learn_start:
            if agent.step % agent.test_step == agent.test_step - 1:
              avg_rewardA = total_rewardA / agent.test_step
              avg_rewardB = total_rewardB / agent2.test_step
              avg_lossA = agent.total_loss / agent.update_count
              avg_lossB = agent2.total_loss / agent2.update_count
              avg_qA = agent.total_q / agent.update_count
              avg_qB = agent2.total_q / agent2.update_count

              try:
                max_ep_rewardA = np.max(ep_rewardsA)
                min_ep_rewardA = np.min(ep_rewardsA)
                avg_ep_rewardA = np.mean(ep_rewardsA)
                max_ep_rewardB = np.max(ep_rewardsB)
                min_ep_rewardB = np.min(ep_rewardsB)
                avg_ep_rewardB = np.mean(ep_rewardsB)
              except:
                max_ep_rewardA, min_ep_rewardA, avg_ep_rewardA, max_ep_rewardB, min_ep_rewardB, avg_ep_rewardB = 0, 0, 0, 0, 0, 0

              print('\nFor Agent A at Epoch %d: avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                  % (agent.epoch, avg_rewardA, avg_lossA, avg_qA, avg_ep_rewardA, max_ep_rewardA, min_ep_rewardA, num_game))
              print('\nFor Agent B at Epoch %d: avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                  % (agent2.epoch, avg_rewardB, avg_lossB, avg_qB, avg_ep_rewardB, max_ep_rewardB, min_ep_rewardB, num_game))

              if max_avg_ep_rewardA * 0.9 <= avg_ep_rewardA:
                agent.step_assign_op.eval({agent.step_input: agent.step + 1})
                agent.save_model(agent.step + 1)

                max_avg_ep_rewardA = max(max_avg_ep_rewardA, avg_ep_rewardA)

              if max_avg_ep_rewardB * 0.9 <= avg_ep_rewardB:
                agent2.step_assign_op.eval({agent2.step_input: agent2.step + 1})
                agent2.save_model(agent2.step + 1)

                max_avg_ep_rewardB = max(max_avg_ep_rewardB, avg_ep_rewardB)

              if agent.step > 180:
                agent.inject_summary({
                    'average.reward': avg_rewardA,
                    'average.loss': avg_lossA,
                    'average.q': avg_qA,
                    'episode.max reward': max_ep_rewardA,
                    'episode.min reward': min_ep_rewardA,
                    'episode.avg reward': avg_ep_rewardA,
                    'episode.num of game': num_game,
                    'episode.rewards': ep_rewardsA,
                    'episode.actions': actionsA,
                    'training.learning_rate': agent.learning_rate_op.eval({agent.learning_rate_step: agent.step}),
                  }, agent.step)

              if agent2.step > 180:
                agent2.inject_summary({
                    'average.reward': avg_rewardB,
                    'average.loss': avg_lossB,
                    'average.q': avg_qB,
                    'episode.max reward': max_ep_rewardB,
                    'episode.min reward': min_ep_rewardB,
                    'episode.avg reward': avg_ep_rewardB,
                    'episode.num of game': num_game,
                    'episode.rewards': ep_rewardsB,
                    'episode.actions': actionsB,
                    'training.learning_rate': agent2.learning_rate_op.eval({agent2.learning_rate_step: agent2.step}),
                  }, agent2.step)

              # Reset statistics
              num_game = 0
              total_rewardA, total_rewardB = 0., 0.
              agent.total_loss, agent2.total_loss = 0., 0.
              agent.total_q, agent2.total_q = 0., 0.
              agent.update_count, agent2.update_count = 0, 0
              ep_rewardA, ep_rewardB = 0., 0.
              ep_rewardsA, ep_rewardsB = [], []
              actionsA, actionsB = [], []

        # Play 10 games at the end of epoch to get game statistics
        total_points, paddle_bounce, wall_bounce, serving_time = [], [], [], []
        for _ in range(10):
          cur_total_points, cur_paddle_bounce, cur_wall_bounce, cur_serving_time = 0, 0, 0, 0

          # Restart game
          ale.ale_resetGame()

          # Get first frame of gameplay
          numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
          rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
          del numpy_surface        
          game_screen.paint(rgb)
          pooled_screen = game_screen.grab()
          scaled_pooled_screen = scale_image(pooled_screen)

          # Create history for testing purposes
          test_history = History(config)

          # Fill first 4 images with initial screen
          for _ in range(agent.history_length):
            test_history.add(scaled_pooled_screen)

          while not ale.ale_isGameOver():
            # 1. predict
            action1 = agent.predict(agent.history.get())
            action2 = agent2.predict(agent2.history.get())

            # 2. act
            ale.ale_act2(action1, action2)
            terminal = ale.ale_isGameOver()
            rewardA = ale.ale_getRewardA()
            rewardB = ale.ale_getRewardB()

            # Record game statistics of current episode
            cur_total_points = ale.ale_getPoints()
            cur_paddle_bounce = ale.ale_getSideBouncing()
            if ale.ale_getWallBouncing():
              cur_wall_bounce += 1
            if ale.ale_getServing():
              cur_serving_time += 1

            # Fill buffer of game screen with current frame
            numpy_surface = np.frombuffer(game_surface.get_buffer(), dtype=np.uint8)
            rgb = getRgbFromPalette(ale, game_surface, numpy_surface)
            del numpy_surface        
            game_screen.paint(rgb)
            pooled_screen = game_screen.grab()
            scaled_pooled_screen = scale_image(pooled_screen)
            agent.observe(scaled_pooled_screen, rewardA, action1, terminal)
            agent2.observe(scaled_pooled_screen, rewardB, action2, terminal)

            # Print frame onto display screen
            screen_ale.blit(pygame.transform.scale2x(game_surface), (0, 0))

            # Update the display screen
            pygame.display.flip()

          # Append current episode's statistics into list
          total_points.append(cur_total_points)
          paddle_bounce.append(cur_paddle_bounce / cur_total_points)
          if cur_paddle_bounce == 0:
            wall_bounce.append(cur_wall_bounce / (cur_paddle_bounce + 1))
          else:
            wall_bounce.append(cur_wall_bounce / cur_paddle_bounce)
          serving_time.append(cur_serving_time / cur_total_points)

        # Save results of test after current epoch
        cur_paddle_op = agent.paddle_op.eval()
        cur_paddle_op[agent.epoch] = sum(paddle_bounce) / len(paddle_bounce)
        agent.paddle_assign_op.eval({agent.paddle_input: cur_paddle_op})

        cur_wall_op = agent.wall_op.eval()
        cur_wall_op[agent.epoch] = sum(wall_bounce) / len(wall_bounce)
        agent.wall_assign_op.eval({agent.wall_input: cur_wall_op})

        cur_serving_op = agent.serving_op.eval()
        cur_serving_op[agent.epoch] = sum(serving_time) / len(serving_time)
        agent.serving_assign_op.eval({agent.serving_input: cur_serving_op})

        agent.save_model(agent.step + 1)
    else:
      agent.play()
      agent2.play()

if __name__ == '__main__':
  tf.app.run()
