import itertools
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import random
import collections

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box


class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        img = np.squeeze(img)
        return img


def make_env():
    env_spec = gym.spec('ppaquette/DoomBasic-v0')
    env_spec.id = 'DoomBasic-v0'
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                        width=80, height=80, grayscale=True)
    return e


env = make_env()

NOOP, SHOOT, RIGHT, LEFT = 0, 1, 2, 3
VALID_ACTIONS = [0, 1, 2, 3]
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
USE_REWARD_PREDICTION = True
USE_VALUE_REPLAY = True
USE_PIXEL_CONTROL = True
EXP_HIST_SIZE = 2000
T_MAX = 5
GAMMA_PC = 0.9
PIXEL_CHANGE_LAMBDA = 0.01


class ExperienceFrame(object):
    def __init__(self, state, action, reward, next_state, done, pixel_change):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.pixel_change = pixel_change


class Experience(object):
    def __init__(self, history_size):
        self._history_size = history_size
        self._frames = collections.deque(maxlen=history_size)
        self._zero_reward_indices = collections.deque(maxlen=history_size)
        self._non_zero_reward_indices = collections.deque(maxlen=history_size)

    def add_frame(self, frame):
        self._frames.append(frame)
        if frame.reward == 0:
            self._zero_reward_indices.append(frame)
        else:
            self._non_zero_reward_indices.append(frame)

    def is_full(self):
        return len(self._frames) >= self._history_size

    def ready_rp(self):
        return len(self._zero_reward_indices) >= 10 and len(self._non_zero_reward_indices) >= 10

    def sample_sequence(self, sequence_size):
        # -5 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        start_pos = np.random.randint(0, max(1, len(self._frames) - sequence_size + 1))

        sampled_frames = []

        for i in range(sequence_size):
            frame = self._frames[start_pos + i]
            sampled_frames.append(frame)
            if frame.done:
                break

        return sampled_frames

    def sample_rp_sequence(self, sequence_size=1):
        from_zero = True
        if np.random.randint(2) == 1 or len(self._zero_reward_indices) == 0:
            from_zero = False
        if len(self._non_zero_reward_indices) == 0:
            from_zero = True

        if from_zero:
            start_pos = np.random.randint(0, len(self._zero_reward_indices) - sequence_size + 1)
        if not from_zero:
            start_pos = np.random.randint(0, len(self._non_zero_reward_indices) - sequence_size + 1)

        sampled_frames = []

        for i in range(sequence_size):
            if from_zero:
                frame = self._zero_reward_indices[start_pos + i]
            if not from_zero:
                frame = self._non_zero_reward_indices[start_pos + i]

            sampled_frames.append(frame)
            if frame.done:
                break

        return sampled_frames


class Estimator():
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

        self.states = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='X')
        self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pi")
        self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_v")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.states)[0]

        with tf.variable_scope("shared", reuse=False):
            conv2 = self.build_shared_network(self.states)

        # Fully connected layer
        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.contrib.layers.flatten(conv2),
            num_outputs=256,
            scope="fc1")

        ### Policy
        self.logits_pi = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
        self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

        self.predictions_pi = {
            "logits": self.logits_pi,
            "probs": self.probs_pi
        }

        # We add entropy to the loss to encourage exploration
        self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")
        self.entropy_mean_pi = tf.reduce_mean(self.entropy_pi, name="entropy_mean")

        # Get the predictions for the chosen actions only
        gather_indices_pi = tf.range(batch_size) * tf.shape(self.probs_pi)[1] + self.actions
        self.picked_action_probs_pi = tf.gather(tf.reshape(self.probs_pi, [-1]), gather_indices_pi)

        self.losses_pi = - (tf.log(self.picked_action_probs_pi) * self.targets_pi + 0.01 * self.entropy_pi)
        self.loss_pi = tf.reduce_sum(self.losses_pi, name="loss_pi")

        ### Value
        self.logits_v = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=1,
            activation_fn=None,
            scope='logits_value')
        self.logits_v = tf.squeeze(self.logits_v, squeeze_dims=[1])

        self.predictions_v = {
            "logits": self.logits_v
        }

        self.losses_v = tf.squared_difference(self.logits_v, self.targets_v)
        self.loss_v = tf.reduce_sum(self.losses_v, name="loss_v")

        # Combine loss
        self.loss = self.loss_pi + self.loss_v

        self.pc_vr_lambda = tf.placeholder(dtype=tf.float32, shape=())
        if USE_REWARD_PREDICTION:
            ### Reward prediction
            self._build_rp_network()
            self.loss = self.loss + self.rp_loss

        if USE_VALUE_REPLAY:
            ### Value Replay
            self._build_vr_network()
            self.loss = self.loss + self.vr_loss

        if USE_PIXEL_CONTROL:
            ### Pixel Control
            self._build_pc_network()
            self.loss = self.loss + self.pc_loss

        # Train
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)

    def build_shared_network(self, X):
        conv1 = tf.contrib.layers.conv2d(
            X, 16, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 32, 4, 2, activation_fn=tf.nn.relu)

        return conv2

    def _build_rp_network(self):
        self.rp_states = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32)
        self.rp_lambda = tf.placeholder(dtype=tf.float32, shape=())

        with tf.variable_scope("shared", reuse=True):
            conv2 = self.build_shared_network(self.rp_states)

        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.contrib.layers.flatten(conv2),
            num_outputs=128,
            scope="rp_fc1")

        self.rp_logits = tf.contrib.layers.fully_connected(fc1, 3, activation_fn=None)
        self.rp_c = tf.nn.softmax(self.rp_logits) + 1e-8

        self.rp_c_targets = tf.placeholder(shape=[None, 3], dtype=tf.float32)

        self.rp_loss = -tf.reduce_sum(self.rp_c_targets * tf.log(self.rp_c))
        self.rp_loss = self.rp_lambda * self.rp_loss

    def _build_vr_network(self):
        self.vr_states = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32)
        self.vr_value_targets = tf.placeholder(shape=[None], dtype=tf.float32)

        with tf.variable_scope("shared", reuse=True):
            conv2 = self.build_shared_network(self.vr_states)

        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.contrib.layers.flatten(conv2),
            num_outputs=256,
            scope="fc1",
            reuse=True)

        self.vr_value = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=1,
            activation_fn=None,
            scope='logits_value',
            reuse=True)

        self.vr_value = tf.squeeze(self.vr_value, squeeze_dims=[1])

        self.vr_losses = tf.squared_difference(self.vr_value, self.vr_value_targets)
        self.vr_loss = tf.reduce_sum(self.vr_losses)
        self.vr_loss = self.pc_vr_lambda * self.vr_loss

    def _build_pc_network(self):
        self.pc_input = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32)

        with tf.variable_scope("shared", reuse=True):
            conv2 = self.build_shared_network(self.pc_input)

        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.contrib.layers.flatten(conv2),
            num_outputs=256,
            scope="fc1",
            reuse=True)

        h_pc_fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=9 * 9 * 32)

        h_pc_fc1_reshaped = tf.reshape(h_pc_fc1, [-1, 9, 9, 32])

        # 1 - value, 4 - action
        h_pc_deconv_v = tf.contrib.layers.conv2d_transpose(h_pc_fc1_reshaped, 1, 4, 2, activation_fn=tf.nn.relu,
                                                           padding='VALID')
        h_pc_deconv_a = tf.contrib.layers.conv2d_transpose(h_pc_fc1_reshaped, 4, 4, 2, activation_fn=tf.nn.relu,
                                                           padding='VALID')

        h_pc_deconv_a_mean = tf.reduce_mean(h_pc_deconv_a, axis=3, keep_dims=True)

        # {Pixel change Q (output)
        self.pc_q = h_pc_deconv_v + h_pc_deconv_a - h_pc_deconv_a_mean
        # (-1, 20, 20, action_size)

        # Max Q
        self.pc_q_max = tf.reduce_max(self.pc_q, reduction_indices=3, keep_dims=False)
        # (-1, 20, 20)

        self._pc_loss_prepare()

    def _pc_loss_prepare(self):
        # [pixel change]
        self.pc_a = tf.placeholder("float", [None, self.num_outputs])
        pc_a_reshaped = tf.reshape(self.pc_a, [-1, 1, 1, self.num_outputs])

        # Extract Q for taken action
        pc_qa_ = tf.multiply(self.pc_q, pc_a_reshaped)
        pc_qa = tf.reduce_sum(pc_qa_, reduction_indices=3, keep_dims=False)
        # (-1, 20, 20)

        # TD target for Q
        self.pc_r = tf.placeholder("float", [None, 20, 20])

        pc_loss = PIXEL_CHANGE_LAMBDA * tf.nn.l2_loss(self.pc_r - pc_qa)
        self.pc_loss = self.pc_vr_lambda * pc_loss


class Agent():
    def __init__(self, name, env, model_net, discount_factor, t_max, summary_writer):
        self.name = name
        self.env = env
        self.model_net = model_net
        self.discount_factor = discount_factor
        self.t_max = t_max
        self.episode_counter = 1
        self.experience = Experience(EXP_HIST_SIZE)
        self.summary_writer = summary_writer

        self.ep_reward_pl = tf.placeholder(shape=(), dtype=tf.float32)
        self.ep_length_pl = tf.placeholder(shape=(), dtype=tf.float32)

        self.summaries_op = tf.summary.merge([
            tf.summary.scalar("episode_reward", self.ep_reward_pl),
            tf.summary.scalar("episode_length", self.ep_length_pl)
        ])

        self._reset()

    def run(self, sess):
        transitions = self.run_n_steps(self.t_max, sess)
        self.update(transitions, sess)

    def _policy_net_predict(self, state, sess):
        feed_dict = {self.model_net.states: [state]}
        preds = sess.run(self.model_net.predictions_pi, feed_dict)
        return preds["probs"][0]

    def _value_net_predict(self, state, sess):
        feed_dict = {self.model_net.states: [state]}
        preds = sess.run(self.model_net.predictions_v, feed_dict)
        return preds["logits"][0]

    def _run_vr_value(self, state, sess):
        vr_v_out = sess.run(self.model_net.vr_value, {self.model_net.vr_states: [state]})
        return vr_v_out[0]

    def run_pc_q_max(self, state, sess):
        q_max_out = sess.run(self.model_net.pc_q_max, {self.model_net.pc_input: [state]})
        return q_max_out[0]

    def _subsample(self, a, average_width):
        s = a.shape
        sh = s[0] // average_width, average_width, s[1] // average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)

    def _calc_pixel_change(self, state, next_state):
        d = np.absolute(state - next_state)
        # (80,80,3)
        m = np.mean(d, 2)
        c = self._subsample(m, 4)
        return c

    def _reset(self):
        self.total_reward = 0
        self.episode_length = 0
        self.action_counter = []

        state = self.env.reset()
        self.state = np.stack([state] * 4, axis=2)

    def run_n_steps(self, t_max, sess):
        transitions = []
        for _ in range(t_max):
            action_probs = self._policy_net_predict(self.state, sess)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = self.env.step(action)
            reward /= 100.
            next_state = np.append(self.state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            pixel_change = self._calc_pixel_change(self.state, next_state)

            self.total_reward += reward
            self.episode_length += 1
            self.action_counter.append(action)

            frame = ExperienceFrame(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done,
                pixel_change=pixel_change)
            self.experience.add_frame(frame)

            transitions.append(Transition(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done))

            if done:
                print("agent {}, episode {}, total_reward {}, episode_length {}, action distr {}".format(
                    self.name, self.episode_counter, int(self.total_reward * 100), self.episode_length,
                    np.bincount(self.action_counter)))

                summaries = sess.run(self.summaries_op, {self.ep_reward_pl: int(self.total_reward * 100),
                                                         self.ep_length_pl: self.episode_length})
                self.summary_writer.add_summary(summaries, self.episode_counter)

                self._reset()
                self.episode_counter += 1
                break
            else:
                self.state = next_state
        return transitions

    def _process_rp(self, sess):
        # [Reward prediction]
        # default sample size 1
        transitions = self.experience.sample_rp_sequence()

        states = []
        reward_classes = []

        for transition in transitions[::-1]:
            reward_class = np.zeros(3)

        if transition.reward == 0:
            reward_class[0] = 1.0  # zero
        elif transition.reward > 0:
            reward_class[1] = 1.0  # positive
        else:
            reward_class[2] = 1.0  # negative

        states.append(transition.state)
        reward_classes.append(reward_class)

        return states, reward_classes

    def _process_vr(self, sess):
        # [Value replay]
        # Sample 20+1 frame (+1 for last next state)
        transitions = self.experience.sample_sequence(T_MAX)

        reward = 0.0
        if not transitions[-1].done:
            reward = self._run_vr_value(transitions[-1].next_state, sess)

        # Accumulate minibatch exmaples
        states = []
        value_targets = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
        # Accumulate updates
        states.append(transition.state)
        value_targets.append(reward)

        return states, value_targets

    def _process_pc(self, sess):
        transitions = self.experience.sample_sequence(T_MAX)

        pc_R = np.zeros([20, 20], dtype=np.float32)
        if not transitions[-1].done:
            pc_R = self.run_pc_q_max(transitions[-1].next_state, sess)

        batch_pc_si = []
        batch_pc_a = []
        batch_pc_R = []

        for frame in transitions[::-1]:
            pc_R = frame.pixel_change + GAMMA_PC * pc_R
            a = np.zeros([self.model_net.num_outputs])
            a[frame.action] = 1.0

            batch_pc_si.append(frame.state)
            batch_pc_a.append(a)
            batch_pc_R.append(pc_R)

        return batch_pc_si, batch_pc_a, batch_pc_R

    def update(self, transitions, sess):
        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        if not transitions[-1].done:
            reward = self._value_net_predict(transitions[-1].next_state, sess)

        # Accumulate minibatch exmaples
        states = []
        policy_targets = []
        value_targets = []
        actions = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
            policy_target = (reward - self._value_net_predict(transition.state, sess))
            # Accumulate updates
            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)

        feed_dict = {
            self.model_net.states: np.array(states),
            self.model_net.targets_pi: policy_targets,
            self.model_net.targets_v: value_targets,
            self.model_net.actions: actions,
        }

        pc_vr_lambda = 0
        if self.experience.is_full(): pc_vr_lambda = 1

        if USE_REWARD_PREDICTION:
            rp_lambda = 0
            if self.experience.ready_rp(): rp_lambda = 1

            rp_states, rp_c_targets = self._process_rp(sess)
            rp_feed_dict = {
                self.model_net.rp_states: rp_states,
                self.model_net.rp_c_targets: rp_c_targets,
                self.model_net.rp_lambda: rp_lambda
            }
            feed_dict.update(rp_feed_dict)

        if USE_VALUE_REPLAY:
            vr_states, vr_value_targets = self._process_vr(sess)
            vr_feed_dict = {
                self.model_net.vr_states: vr_states,
                self.model_net.vr_value_targets: vr_value_targets,
                self.model_net.pc_vr_lambda: pc_vr_lambda
            }
            feed_dict.update(vr_feed_dict)

        #TODO
        if USE_PIXEL_CONTROL:
            batch_pc_si, batch_pc_a, batch_pc_R = self._process_pc(sess)

            pc_feed_dict = {
                self.model_net.pc_input: batch_pc_si,
                self.model_net.pc_a: batch_pc_a,
                self.model_net.pc_r: batch_pc_R,
                self.model_net.pc_vr_lambda: pc_vr_lambda
            }
            feed_dict.update(pc_feed_dict)

        # Train the global estimators using local gradients
        mnet_loss, _ = sess.run([
            self.model_net.loss,
            self.model_net.train_op
        ], feed_dict)

        return mnet_loss


name_of_run = '3tasks-vr-rp-pc0.01'
summary_dir = 'logs/' + name_of_run
if not os.path.exists(summary_dir): os.makedirs(summary_dir)

summary_writer = tf.summary.FileWriter(summary_dir)

model = Estimator(len(VALID_ACTIONS))
agent = Agent(name=name_of_run, env=env, model_net=model, discount_factor=0.99, t_max=T_MAX,
              summary_writer=summary_writer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:
        agent.run(sess)