from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from convnet import create_network

class Agent:

    def __init__(self, args):
        self.state_dim = args.state_dim
        self.actions = args.actions
        self.num_actions = len(actions)
        self.verbose = args.verbose
        self.best = args.best

        # epsilon annealing
        self.eps_start = args.ep_start or 1
        self.ep = args.ep
        self.ep_end = args.ep_end or self.ep
        self.ep_endt = args.ep_endt or 1000000

        # learning rate annealing
        self.lr_start = args.lr or 0.01
        self.lr = self.lr_start
        self.lr_end = args.lr_end or self.lr
        self.lr_endt = args.lr_endt or 10000000
        self.wc = args.wc or 0
        self.minibatch_size = args.minibatch_size or 1
        self.valid_size = args.valid_size or 500

        self.discount       = args.discount or 0.99
        self.update_freq    = args.update_freq or 1
        # Number of points to replay per learning step
        self.n_replay       = args.n_replay or 1
        # Number of steps after which learning starts
        self.learn_start    = args.learn_start or 0
        # Size of the transition table
        self.replay_memory  = args.replay_memory or 1000000
        self.hist_len       = args.hist_len or 1
        self.rescale_r      = args.rescale_r
        self.max_reward     = args.max_reward
        self.min_reward     = args.min_reward
        self.clip_delta     = args.clip_delta
        self.target_q       = args.target_q
        self.bestq = 0

        self.num_cols = args.num_cols or 1  # Number of color channels in input
        self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
        self.preproc        = args.preproc
        self.histType       = args.histType or "linear"
        self.histSpacing    = args.histSpacing or 1
        self.nonTermProb    = args.nonTermProb or 1
        self.bufferSize = args.bufferSize or 512

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        # Create Network
        create_network()


    def perceive(self, screen, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < self.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        q_t_plus_1 = self.target_q.eval({self.s_t: s_t_plus_1})

        terminal = np.array(terminal) + 0

        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)

        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward


        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
        self.target_q_t: target_q_t,
        self.action: action,
        self.s_t: s_t,
        self.learning_rate_step: self.step,
        })

        self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1
