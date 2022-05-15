# coding=utf-8
""" Agent zoo. Each agent is a Neural Network carrier """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

nest = tf.contrib.framework.nest

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy baseline q_aux')


class FCNetAgentContinuous(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(FCNetAgentContinuous, self).__init__(name='fc_net_agent_continuous')

        self._num_actions = num_actions

        # 常在DRL中使用，适用于一个timeslot运行一次RNN的场景,用来和环境进行交互 常用num_units=256
        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    # 没有文本游戏，所以不行哟embedding层
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)

        with tf.variable_scope('convnet'):
            conv_out = frame

        conv_out = tf.layers.dense(frame, 200, tf.nn.relu6, kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(reward, -1)
        # _last_action = tf.expand_dims(last_action, 0)

        return tf.concat([conv_out, clipped_reward, last_action],
                         axis=1)

    def _head(self, core_output):
        from gym_settings import action_bound
        with tf.variable_scope('policy'):
            mu = tf.layers.dense(core_output, self._num_actions, tf.nn.tanh,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())  # TODO:输出
            sigma = tf.layers.dense(core_output, self._num_actions, tf.nn.softplus,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            normal_dist = tfp.distributions.Normal(loc=mu * action_bound, scale=sigma + 1e-5)

        baseline = tf.squeeze(
            tf.layers.dense(core_output, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='baseline'),
            axis=-1)

        # Sample an action from the policy.
        new_action = normal_dist.sample(1)
        new_action = tf.squeeze(new_action, axis=0, name='new_action')  # 删除所有维度1

        policy = {'prob': normal_dist.prob(new_action), 'log_prob': normal_dist.log_prob(new_action),
                  'entropy': normal_dist.entropy()}
        # policy=normal_dist.prob(new_action)

        return AgentOutput(new_action, policy, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                  (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state


class FCNetAgent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(FCNetAgent, self).__init__(name='fc_net_agent')

        self._num_actions = num_actions

        # 常在DRL中使用，适用于一个timeslot运行一次RNN的场景,用来和环境进行交互 num_units=256
        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    # 没有文本游戏，所以不行哟embedding层
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)

        with tf.variable_scope('convnet'):
            conv_out = frame

        conv_out = tf.layers.dense(frame, 200, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0., .1))

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(reward, -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        _last_action = one_hot_last_action

        return tf.concat([conv_out, clipped_reward, _last_action],
                         axis=1)

    def _head(self, core_output):
        policy = snt.Linear(self._num_actions, name='policy')(
            core_output)
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy, num_samples=1,
                                    output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')  # 删除所有维度1

        return AgentOutput(new_action, policy, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                  (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state


class SimpleConvNetAgent(snt.AbstractModule):
    """Agent with Simple CNN."""

    def __init__(self, num_actions):
        super(SimpleConvNetAgent, self).__init__(name='simple_convnet_agent')

        self._num_actions = num_actions

    def initial_state(self, batch_size):
        return tf.constant(0, shape=[1, 1])

    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        frame = tf.to_float(frame)
        frame /= 255

        with tf.variable_scope('convnet'):
            conv_out = frame
            conv_out = snt.Conv2D(32, 8, stride=4)(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(64, 4, stride=2)(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(64, 3, stride=1)(conv_out)
            conv_out = tf.nn.relu(conv_out)

        conv_out = snt.BatchFlatten()(conv_out)
        conv_out = snt.Linear(512)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        return tf.concat(
            [conv_out, clipped_reward, one_hot_last_action],
            axis=1)

    def _head(self, core_output):
        policy = snt.Linear(self._num_actions, name='policy')(
            core_output)
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy, num_samples=1,
                                    output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')

        return AgentOutput(new_action, policy, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                  (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        return snt.BatchApply(self._head)(torso_outputs), core_state


class ResNetLSTMAgent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions, num_players=1):
        super(ResNetLSTMAgent, self).__init__(name='resnet_lstm_agent')

        self._num_actions = num_actions
        self._num_players = num_players

        # 常在DRL中使用，适用于一个timeslot运行一次RNN的场景,用来和环境进行交互 num_units=256
        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size, **kwargs):
        return self._core.zero_state(batch_size, tf.float32)

    # 没有文本游戏，所以不行哟embedding层
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)
        # 不是图片
        # frame /= 255

        with tf.variable_scope('convnet'):
            conv_out = frame
            for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
                # Downscale.
                conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                conv_out = tf.nn.pool(
                    conv_out,
                    window_shape=[3, 3],
                    pooling_type='MAX',
                    padding='SAME',
                    strides=[2, 2])

                # Residual block(s).
                for j in range(num_blocks):
                    with tf.variable_scope('residual_%d_%d' % (i, j)):
                        block_input = conv_out
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                        conv_out += block_input

        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)

        if self._num_players > 1:
            one_hot_last_action = tf.reshape(one_hot_last_action, [one_hot_last_action.shape[0], -1])

        return tf.concat([conv_out, clipped_reward, one_hot_last_action],
                         axis=1)

    def _head(self, core_output):
        policy = snt.Linear(self._num_actions, name='policy_0')(
            core_output)
        new_action = tf.multinomial(policy, num_samples=1,
                                    output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')

        pc_fc1 = snt.Linear(9 * 9 * 32, name='pc_fc1')(core_output)
        pc_fc1 = tf.nn.relu(pc_fc1)
        pc_fc1_reshaped = tf.reshape(pc_fc1, [-1, 9, 9, 32])

        # Q_aux
        pc_deconv_v = tf.layers.conv2d_transpose(pc_fc1_reshaped, filters=1, kernel_size=4, strides=2,
                                                 activation=tf.nn.relu, padding='VALID', name='deconv_v_0')
        pc_deconv_a = tf.layers.conv2d_transpose(pc_fc1_reshaped, filters=self._num_actions, kernel_size=4, strides=2,
                                                 activation=tf.nn.relu, padding='VALID', name='deconv_a_0')
        pc_deconv_a_mean = tf.reduce_mean(pc_deconv_a, axis=3, keep_dims=True)
        pc_q = pc_deconv_v + (pc_deconv_a - pc_deconv_a_mean)

        for i in range(self._num_players - 1):
            # policy output
            policy_i = snt.Linear(self._num_actions, name='policy_%d' % (i + 1))(
                core_output)
            new_action_i = tf.multinomial(policy_i, num_samples=1,
                                          output_dtype=tf.int32)
            new_action_i = tf.squeeze(new_action_i, 1, name='new_action')
            if i == 0:
                new_action = tf.stack([new_action, new_action_i], axis=1)
            else:
                new_action_i = tf.expand_dims(new_action_i, 1)
                new_action = tf.concat([new_action, new_action_i], axis=1)

            policy = tf.concat([policy, policy_i], axis=1)

            # pixel control Q_aux output
            pc_deconv_v_i = tf.layers.conv2d_transpose(pc_fc1_reshaped, filters=1, kernel_size=4, strides=2,
                                                       activation=tf.nn.relu, padding='VALID',
                                                       name='deconv_v_%d' % (i + 1))
            pc_deconv_a_i = tf.layers.conv2d_transpose(pc_fc1_reshaped, filters=self._num_actions, kernel_size=4,
                                                       strides=2,
                                                       activation=tf.nn.relu, padding='VALID',
                                                       name='deconv_a_%d' % (i + 1))
            pc_deconv_a_mean_i = tf.reduce_mean(pc_deconv_a_i, axis=3, keep_dims=True)
            pc_q_i = pc_deconv_v_i + (pc_deconv_a_i - pc_deconv_a_mean_i)
            pc_q=tf.concat([pc_q,pc_q_i],axis=1)

        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        return AgentOutput(new_action, policy, baseline,pc_q)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                  (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state


def agent_factory(agent_name):
    supported_agent = {
        'FCNetAgent'.lower(): FCNetAgent,
        'SimpleConvNetAgent'.lower(): SimpleConvNetAgent,
        'ResNetLSTMAgent'.lower(): ResNetLSTMAgent,
        'FCNetAgentContinuous'.lower(): FCNetAgentContinuous,
    }
    return supported_agent[agent_name.lower()]
