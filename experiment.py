# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import sys
import os
import shutil
import random

import environments
import numpy as np
import py_process
import tensorflow as tf
import pandas as pd
import vtrace
import utils
from gym_settings import LEVEL_MAPPING, env_type
from agent import agent_factory
from six.moves import range

try:
    import dynamic_batching
except tf.errors.NotFoundError:
    tf.logging.warning('Running without dynamic batching.')

nest = tf.contrib.framework.nest

flags = tf.app.flags

flags.DEFINE_string('gpu_id', '0', 'choose ID of gpu to use')
flags.DEFINE_string('logdir', './log', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Training.
flags.DEFINE_integer('total_environment_frames', int(7.5*1e7),
                     'Total environment frames to train for.')
flags.DEFINE_integer("max_episode_len", int(400), "maximum episode length, if -1, no limited")
flags.DEFINE_integer('num_actors', 80, 'Number of actors.')  # 100
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 20, 'Unroll length in agent steps.')  # 20
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')  # 动作重复帧数: 4
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_bool('is_visible', False, 'whether we use render() to visualize our games')

# TODO:contribution
flags.DEFINE_string('agent_name', 'ResNetLSTMAgent', 'agent name. is or not Continuous')
flags.DEFINE_integer('num_players', 2,
                     'Number of players in a game. Note that this is a centralized multi-agent framework')
flags.DEFINE_float('pc_cost', 0.1, 'Pixel control loss cost, if ==0, no use pc')
flags.DEFINE_integer('pc_cell_size', 4, 'size of the cells used to derive the pixel based pseudo-rewards')
flags.DEFINE_float('pc_scale', 1., 'scale factor for pixel control')

# TODO：PBT
flags.DEFINE_bool('use_pbt', False, 'whether use population based training (PBT)')
flags.DEFINE_integer('population_size', int(1), 'num of population process')
flags.DEFINE_integer('population_index', int(0), 'population index of population based training')
flags.DEFINE_integer('pbt_period', int(5000), 'period of explore-and-exploit')
flags.DEFINE_string('store_dir', '/data-store', 'shared data-store directory.')
flags.DEFINE_bool('game_end', False, 'Due to complex PBT, show whether game ends')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.1, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', 0.5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', 0.95, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'adaptive_normalisation', ['abs_one', 'soft_asymmetric', 'adaptive_normalisation'],
                  'Reward clipping.')
flags.DEFINE_float('gradients_clipping', 40.0,
                   'Gradients clipping. Negative number means not clipping. ')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0.0, 'RMSProp momentum.')
flags.DEFINE_float('epsilon', 0.01, 'RMSProp epsilon.')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

# Structure to be sent from actors to learner.   创建一个StepOutPut类型，而且带有后面的属性
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')
ActorOutputNoState = collections.namedtuple(
    'ActorOutputNoState', 'level_name env_outputs agent_outputs')


def is_single_machine():
    return FLAGS.task == -1


def build_actor(agent, env, level_name, action_set):
    """Builds the actor loop."""
    # Initial values.
    initial_env_output, initial_env_state = env.initial()
    initial_agent_state = agent.initial_state(1, )  # LSTM 零初始化
    if env_type == 'D':
        if FLAGS.num_players == 1:
            initial_action = tf.zeros([1], dtype=tf.int32)
        else:
            initial_action = tf.zeros([1, FLAGS.num_players], dtype=tf.int32)
    else:
        initial_action = tf.zeros([1, action_set], dtype=tf.float32)
    dummy_agent_output, _ = agent(
        (initial_action,
         nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
        initial_agent_state)
    initial_agent_output = nest.map_structure(
        lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

    # All state that needs to persist across training iterations. This includes
    # the last environment output, agent state and last agent output. These
    # variables should never go on the parameter servers.
    def create_state(t):
        # Creates a unique variable scope to ensure the variable name is unique.
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    persistent_state = nest.map_structure(
        create_state, (initial_env_state, initial_env_output, initial_agent_state,
                       initial_agent_output))

    def step(input_, unused_i):
        """Steps through the agent and the environment."""
        env_state, env_output, agent_state, agent_output = input_

        # Run agent.
        action = agent_output[0]
        batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                env_output)
        agent_output, agent_state = agent((action, batched_env_output), agent_state)

        # Convert action index to the native action.
        if env_type == 'D':
            action = agent_output[0][0]
            raw_action = tf.gather([i for i in range(action_set)], action)
        else:
            # TODO
            raw_action = agent_output[0][0]

        env_output, env_state = env.step(raw_action, env_state)  # 与环境做交互step

        return env_state, env_output, agent_state, agent_output

    # Run the unroll. `read_value()` is needed to make sure later usage will
    # return the first values and not a new snapshot of the variables.  LSTM
    first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
    _, first_env_output, first_agent_state, first_agent_output = first_values

    # Use scan to apply `step` multiple times, therefore unrolling the agent
    # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
    # the output of each call of `step` as input of the subsequent call of `step`.
    # The unroll sequence is initialized with the agent and environment states
    # and outputs as stored at the end of the previous unroll.
    # `output` stores lists of all states and outputs stacked along the entire
    # unroll. Note that the initial states and outputs (fed through `initializer`)
    # are not in `output` and will need to be added manually later.
    '''
      设函数为f，
      x = [u(0),u(1),...,u(n)]
      y = tf.scan(f,x,initializer=v(0))
      此时f的参数类型必须是(v(0),x)，f的输出必须和v(0)保持一致，整个计算过程如下：
      v(1)=f(v(0),u(0))
      v(2)=f(v(1),u(1))
      ....
      v(n+1)=f(v(n),u(n))
      y=v(n+1)
    '''
    output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)

    '''
    env_outputs:done(100,),info(100,),observation(length,72,96,3),reward(100)
    agent_outputs:action(length,1),baseline(length,1),policy_logits(length,1,9)
    '''
    _, env_outputs, _, agent_outputs = output

    # Update persistent state with the last output from the loop.
    assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                    persistent_state, output)

    # The control dependency ensures that the final agent and environment states
    # and outputs are stored in `persistent_state` (to initialize next unroll).
    with tf.control_dependencies(nest.flatten(assign_ops)):
        # Remove the batch dimension from the agent state/output.
        first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
        first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
        agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

        # Concatenate first output and the unroll along the time dimension.
        full_agent_outputs, full_env_outputs = nest.map_structure(
            lambda first, rest: tf.concat([[first], rest], 0),
            (first_agent_output, first_env_output), (agent_outputs, env_outputs))

        output = ActorOutput(
            level_name=level_name, agent_state=first_agent_state,
            env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)

        # No backpropagation should be done here.
        return nest.map_structure(tf.stop_gradient, output)


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    if env_type == 'D':
        return .5 * tf.reduce_mean(tf.square(advantages))
    else:
        return .5 * tf.reduce_mean(tf.square(advantages))


def compute_entropy_loss(policy_input):
    if env_type == 'D':
        policy = tf.nn.softmax(policy_input)
        log_policy = tf.nn.log_softmax(policy_input)
        entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
        return -tf.reduce_mean(entropy_per_timestep)
    else:
        return -tf.reduce_mean(policy_input['entropy'])


def compute_policy_gradient_loss(policy_input, actions, advantages):
    if env_type == 'D':
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=policy_input)
        advantages = tf.stop_gradient(advantages)
        policy_gradient_loss_per_timestep = cross_entropy * advantages
        return tf.reduce_mean(policy_gradient_loss_per_timestep)
    else:
        log_prob = policy_input['log_prob']
        advantages = tf.expand_dims(tf.stop_gradient(advantages), -1)
        return -tf.reduce_mean(log_prob * advantages)


def build_learner(agent, agent_state, env_outputs, agent_outputs, global_step):
    """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

  Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
  """
    # main policy pi
    learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs,
                                      agent_state)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # follow the main policy pi as usual
    action_values = learner_outputs[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.

    agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
    rewards, infos, done, _ = nest.map_structure(
        lambda t: t[1:], env_outputs)
    learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

    ret_rms = None
    if FLAGS.reward_clipping == 'abs_one':
        clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    elif FLAGS.reward_clipping == 'soft_asymmetric':
        squeezed = tf.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.
    elif FLAGS.reward_clipping == 'adaptive_normalisation':  # TODO：PopArt!
        clipped_rewards = rewards
        ret_rms = utils.RunningMeanStd()

    denormalized_values = utils.denormalize(learner_outputs.baseline, ret_rms)
    denormalized_bootstrap_value = utils.denormalize(bootstrap_value, ret_rms)
    agent_name = agent._original_name
    discounts = tf.to_float(~done) * FLAGS.discounting  # not done

    # Compute V-trace returns and weights.
    # Note, this is put on the CPU because it's faster than on GPU. It can be
    # improved further with XLA-compilation or with a custom TensorFlow operation.
    with tf.device('/cpu'):
        if env_type == 'D':
            if FLAGS.num_players == 1:
                vtrace_returns = vtrace.from_logits(
                    behaviour_policy_logits=agent_outputs.policy,
                    target_policy_logits=learner_outputs.policy,
                    actions=agent_outputs.action,
                    discounts=discounts,
                    rewards=clipped_rewards,
                    values=denormalized_values,
                    bootstrap_value=denormalized_bootstrap_value,
                    ret_rms=ret_rms,
                    agent_name=agent_name,
                    flags=FLAGS)
            else:
                vtrace_returns = []
                for i in range(FLAGS.num_players):
                    vtrace_returns_i = vtrace.from_logits(
                        behaviour_policy_logits=agent_outputs.policy[:, :, i * 25:(i + 1) * 25],
                        target_policy_logits=learner_outputs.policy[:, :, i * 25:(i + 1) * 25],
                        actions=agent_outputs.action[:, :, i],
                        discounts=discounts,
                        rewards=clipped_rewards,
                        values=denormalized_values,
                        bootstrap_value=denormalized_bootstrap_value,
                        ret_rms=ret_rms,
                        agent_name=agent_name,
                        flags=FLAGS)
                    vtrace_returns.append(vtrace_returns_i)

        else:
            vtrace_returns = vtrace.from_distribution(
                behaviour_policy_distribution=agent_outputs.policy['prob'],  # (length,batch,9)
                target_policy_distribution=learner_outputs.policy['prob'],  # (length,batch,9)
                actions=agent_outputs.action,  # (length,batch)
                discounts=discounts,  # (length,batch) * 0.99
                rewards=clipped_rewards,  # (length,batch)
                values=learner_outputs.baseline,  # (length,batch)
                bootstrap_value=bootstrap_value)  # (batch,) the last baseline value

    # Compute loss as a weighted sum of the baseline loss, the policy gradient
    # loss and an entropy regularization term.
    if FLAGS.num_players == 1:
        with tf.variable_scope('policy_gradient_loss'):
            policy_gradient_loss = compute_policy_gradient_loss(
                learner_outputs.policy, agent_outputs.action,
                vtrace_returns.pg_advantages)  # 使用vs+1

        with tf.variable_scope('baseline_loss'):
            baseline_loss = compute_baseline_loss(  # value 加权,使用vs
                vtrace_returns.vs - learner_outputs.baseline)

        with tf.variable_scope('entropy_loss'):
            entropy_loss = compute_entropy_loss(  # entropy 加权
                learner_outputs.policy)

        with tf.variable_scope('pixel_control_loss'):
            pc_loss, _ = utils.compute_pixel_control_loss(
                env_outputs.observation, agent_outputs.action, action_values, FLAGS.pc_cell_size, discounts,
                FLAGS.pc_scale)
    else:
        with tf.variable_scope('policy_gradient_loss'):
            policy_gradient_loss = 0
            for i in range(FLAGS.num_players):
                policy_gradient_loss += (compute_policy_gradient_loss(
                    learner_outputs.policy[:, :, i * 25:(i + 1) * 25], agent_outputs.action[:, :, i],
                    vtrace_returns[i].pg_advantages) / FLAGS.num_players)  # 使用vs+1

        with tf.variable_scope('baseline_loss'):
            baseline_loss = 0
            for i in range(FLAGS.num_players):
                baseline_loss += (compute_baseline_loss(  # value 加权,使用vs
                    vtrace_returns[i].vs - learner_outputs.baseline) / FLAGS.num_players)

        with tf.variable_scope('entropy_loss'):
            entropy_loss = 0
            for i in range(FLAGS.num_players):
                entropy_loss += (compute_entropy_loss(  # entropy 加权
                    learner_outputs.policy[:, :, i * 25:(i + 1) * 25]) / FLAGS.num_players)

        with tf.variable_scope('pixel_control_loss'):
            pc_loss = 0
            for i in range(FLAGS.num_players):
                pc_loss_i, _ = utils.compute_pixel_control_loss(
                    env_outputs.observation, agent_outputs.action[:, :, i],
                    action_values[:, :, i * 20:(i + 1) * 20, :, :], FLAGS.pc_cell_size, discounts,
                    FLAGS.pc_scale)
                pc_loss += (pc_loss_i / FLAGS.num_players)

    with tf.variable_scope('total_loss'):
        if FLAGS.pc_cost == 0:
            total_loss = policy_gradient_loss + FLAGS.baseline_cost * baseline_loss + FLAGS.entropy_cost * entropy_loss
        else:
            total_loss = policy_gradient_loss + FLAGS.baseline_cost * baseline_loss \
                         + FLAGS.entropy_cost * entropy_loss + FLAGS.pc_cost * pc_loss

    debug_print_op = tf.py_func(utils.debug_loss,
                                [tf.reduce_mean(rewards), policy_gradient_loss, baseline_loss, entropy_loss, pc_loss,
                                 total_loss],
                                [tf.bool])

    with tf.control_dependencies(debug_print_op):
        # Optimization
        num_env_frames = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                                  FLAGS.total_environment_frames, 0)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                              FLAGS.momentum, FLAGS.epsilon)

        if FLAGS.gradients_clipping > 0.0:
            grads_and_vars = optimizer.compute_gradients(total_loss)
            grads, vars = zip(*grads_and_vars)
            cgrads, _ = tf.clip_by_global_norm(grads, FLAGS.gradients_clipping)
            grads_and_vars = zip(cgrads, vars)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        else:
            train_op = optimizer.minimize(total_loss)

        # Merge updating the network and environment frames into a single tensor.
        with tf.control_dependencies([train_op]):
            num_env_frames_and_train = num_env_frames.assign_add(
                FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)

        # Adding a few summaries.
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('policy_gradient_loss', policy_gradient_loss)
        tf.summary.scalar('baseline_loss', baseline_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('pixel_control_loss', pc_loss)
        tf.summary.histogram('action', agent_outputs.action)

        return done, infos, num_env_frames_and_train


def create_environment(level_name, seed, env_id=-1, is_visible=False):
    """Creates an environment wrapped in a `FlowEnvironment`."""
    if is_visible:
        is_visual = True
    else:
        is_visual = False

    p = py_process.PyProcess(environments.PyProcessGym, level_name, ep_len=FLAGS.max_episode_len, id=env_id,
                             flags=FLAGS, is_visual=is_visual)
    return environments.FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""

    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get('collections', None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope('', custom_getter=getter) as vs:
        yield vs


def train(action_set, level_names):
    """Train."""
    if is_single_machine():
        local_job_device = ''
        shared_job_device = ''
        is_actor_fn = lambda i: True
        is_learner = True
        global_variable_device = '/gpu'
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
        shared_job_device = '/job:learner/task:0'
        is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
        is_learner = FLAGS.job_name == 'learner'

        # Placing the variable on CPU, makes it cheaper to send it to all the
        # actors. Continual copying the variables from the GPU is slow.
        global_variable_device = shared_job_device + '/cpu'
        cluster = tf.train.ClusterSpec({
            'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
            'learner': ['localhost:8000']
        })
        server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                                 task_index=FLAGS.task)
        filters = [shared_job_device, local_job_device]

    # Only used to find the actor output structure.
    Agent = agent_factory(FLAGS.agent_name)
    with tf.Graph().as_default():
        agent = Agent(action_set, FLAGS.num_players)
        env = create_environment(level_name=level_names[0], seed=1)
        structure = build_actor(agent, env, level_names[0], action_set)
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    with tf.Graph().as_default(), \
         tf.device(local_job_device + '/cpu'), \
         pin_global_variables(global_variable_device):
        tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
            agent = Agent(action_set, FLAGS.num_players)

            if is_single_machine() and 'dynamic_batching' in sys.modules:
                # For single machine training, we use dynamic batching for improved GPU
                # utilization. The semantics of single machine training are slightly
                # different from the distributed setting because within a single unroll
                # of an environment, the actions may be computed using different weights
                # if an update happens within the unroll.
                old_build = agent._build

                @dynamic_batching.batch_fn
                def build(*args):
                    with tf.device('/gpu'):
                        return old_build(*args)

                tf.logging.info('Using dynamic batching.')
                agent._build = build

        # Build actors and ops to enqueue their output.
        enqueue_ops = []
        for i in range(FLAGS.num_actors):
            if is_actor_fn(i):
                level_name = level_names[i % len(level_names)]
                tf.logging.info('Creating actor %d with level %s', i, level_name)
                env = create_environment(level_name=level_name, env_id=int(i / len(level_names)),
                                         seed=i + 1)  # env_id:识别本体?分身?
                actor_output = build_actor(agent, env, level_name, action_set)
                with tf.device(shared_job_device):
                    enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

        # If running in a single machine setup, run actors with QueueRunners
        # (separate threads).
        if is_learner and enqueue_ops:
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

        # Build learner.
        if is_learner:
            # Create global step, which is the number of environment frames processed.
            global_step = tf.get_variable(
                'num_environment_frames',
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            # Create batch (time major) and recreate structure.
            dequeued = queue.dequeue_many(FLAGS.batch_size)
            dequeued = nest.pack_sequence_as(structure, dequeued)

            def make_time_major(s):
                return nest.map_structure(
                    lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)

            dequeued = dequeued._replace(
                env_outputs=make_time_major(dequeued.env_outputs),
                agent_outputs=make_time_major(dequeued.agent_outputs))

            with tf.device('/gpu'):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.
                flattened_output = nest.flatten(dequeued)
                area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in flattened_output],
                    [t.shape for t in flattened_output])
                stage_op = area.put(flattened_output)

                data_from_actors = nest.pack_sequence_as(structure, area.get())

                # Unroll agent on sequence, create losses and update ops.
                if hasattr(data_from_actors, 'agent_state'):
                    agent_state = data_from_actors.agent_state
                else:
                    agent_state = agent.initial_state(1, )

                # Unroll agent on sequence, create losses and update ops.
                output = build_learner(agent, agent_state=agent_state,
                                       env_outputs=data_from_actors.env_outputs,
                                       agent_outputs=data_from_actors.agent_outputs,
                                       global_step=global_step)

        # Create MonitoredSession (to run the graph, checkpoint and log).
        tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
        saver = tf.train.Saver()

        with tf.train.MonitoredTrainingSession(
                server.target,
                is_chief=is_learner,
                checkpoint_dir=FLAGS.logdir + "/" + str(FLAGS.population_index) + '/agent',
                save_checkpoint_secs=600,
                save_summaries_secs=30,  # 30
                log_step_count_steps=50000,  # 50000
                config=config,
                hooks=[py_process.PyProcessHook()]) as session:

            if is_learner:
                # Logging.
                level_returns = {level_name: [] for level_name in level_names}
                summary_writer = tf.summary.FileWriterCache.get(
                    FLAGS.logdir + "/" + str(FLAGS.population_index) + '/agent')

                # Prepare data for first run.
                session.run_step_fn(
                    lambda step_context: step_context.session.run(stage_op))

                # Execute learning and track performance.
                num_env_frames_v = 0
                num_episode_rewards = 0
                performance_list_per_population = collections.deque(maxlen=50)

                while num_env_frames_v < FLAGS.total_environment_frames:
                    FLAGS.game_end = True
                    level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
                        (data_from_actors.level_name,) + output + (stage_op,))
                    level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)

                    for level_name, episode_return, episode_step, episode_raw_return, episode_raw_step in zip(
                            level_names_v[done_v],  # 一个回合的奖励
                            infos_v.episode_return[done_v],
                            infos_v.episode_step[done_v],
                            infos_v.episode_raw_return[done_v],
                            infos_v.episode_raw_step[done_v]
                    ):
                        episode_frames = episode_step * FLAGS.num_action_repeats

                        num_episode_rewards += 1
                        level_name = level_name.decode()
                        tf.logging.info(
                            'Pop %d// Total Frames: %d--Level: %s Episode return: %f, Episode raw return: %f',
                            FLAGS.population_index, num_env_frames_v, level_name, episode_return, episode_raw_return)

                        summary = tf.summary.Summary()
                        summary.value.add(tag=level_name + '/episode_return',
                                          simple_value=episode_return)
                        summary.value.add(tag=level_name + '/episode_frames',
                                          simple_value=episode_frames)
                        summary.value.add(tag=level_name + '/episode_raw_return',
                                          simple_value=episode_raw_return)
                        summary.value.add(tag=level_name + '/episode_raw_frames',
                                          simple_value=episode_raw_step)
                        summary_writer.add_summary(summary, num_env_frames_v)

                        level_returns[level_name].append(episode_return)

                    if min(map(len, level_returns.values())) >= 1:
                        summary_writer.add_summary(summary, num_env_frames_v)
                        performance_pair=list(level_returns.values())

                        p_list=[]
                        for p in performance_pair:
                            p_list.append(np.mean(p))

                        performance_list_per_population.append(np.mean(p_list))

                        # Clear level scores.
                        level_returns = {level_name: [] for level_name in level_names}

                    # print(num_episode_rewards, num_env_frames_v)  # debug

                    # TODO:PBT
                    if FLAGS.use_pbt is True and num_episode_rewards > FLAGS.pbt_period:
                        FLAGS.game_end = False
                        np.save(FLAGS.logdir + str(FLAGS.store_dir) + "/per-" + str(FLAGS.population_index) + ".npy",
                                np.array(performance_list_per_population))
                        break

            else:
                # Execute actors (they just need to enqueue their output).
                while True:
                    session.run(enqueue_ops)


def test(action_set, level_names, best_population_index=FLAGS.population_index):
    """Test."""
    Agent = agent_factory(FLAGS.agent_name)
    level_returns = {level_name: [] for level_name in level_names}
    with tf.Graph().as_default():
        agent = Agent(action_set, FLAGS.num_players)
        outputs = {}
        for i, level_name in enumerate(level_names):
            env = create_environment(level_name=level_name, env_id=int(i / len(level_names)), seed=1,
                                     is_visible=FLAGS.is_visible)
            outputs[level_name] = build_actor(agent, env, level_name, action_set)

        with tf.train.SingularMonitoredSession(
                checkpoint_dir=FLAGS.logdir + "/" + str(best_population_index) + '/agent',
                hooks=[py_process.PyProcessHook()]) as session:
            for level_name in level_names:
                tf.logging.info('Testing level: %s', level_name)
                while True:
                    done_v, infos_v = session.run((
                        outputs[level_name].env_outputs.done,
                        outputs[level_name].env_outputs.info
                    ))
                    returns = level_returns[level_name]
                    returns.extend(infos_v.episode_return[1:][done_v[1:]])

                    if len(returns) >= FLAGS.test_num_episodes:
                        tf.logging.info('Mean episode return: %f', np.mean(returns))
                        break


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if os.path.exists(FLAGS.logdir + "/" + str(FLAGS.population_index)) is False:
        os.makedirs(FLAGS.logdir + "/" + str(FLAGS.population_index), exist_ok=True)
        os.makedirs(FLAGS.logdir + str(FLAGS.store_dir), exist_ok=True)
        # TODO: PBT parameters
        pbt_parameters_dict = None
        if FLAGS.use_pbt is True:
            pbt_parameters_dict = {"learning_rate": np.random.uniform(1e-5, 5e-3, 1)[0],
                                   "entropy_cost": np.random.uniform(0.01, 0.5, 1)[0]}
            FLAGS.learning_rate = pbt_parameters_dict["learning_rate"]
            FLAGS.entropy_cost = pbt_parameters_dict["entropy_cost"]

            f = open(FLAGS.logdir + "/" + str(FLAGS.population_index) + '/pbt_parameters.txt', 'w')
            f.write(str(pbt_parameters_dict))
            f.close()

        with open(FLAGS.logdir + "/" + str(FLAGS.population_index) + '/all_hyper_parameters.txt', 'a+') as file:
            print(FLAGS, file=file)
            print("\n-----env------\n", LEVEL_MAPPING, file=file)
            if FLAGS.use_pbt is True:
                print("\n-----population based training\n", pbt_parameters_dict, file=file)

    if env_type == "Unknown":
        print("Unknown Environment")
        exit(0)
    else:
        print("Env type: %s" % env_type)

    if FLAGS.mode == 'train':
        level_names = list(LEVEL_MAPPING.keys())
        action_set = environments.get_action_set(level_name=level_names[0])
        while FLAGS.game_end is False:
            train(action_set, level_names)
            if FLAGS.game_end is False:
                # TODO:PBT
                full_store_dir = FLAGS.logdir + str(FLAGS.store_dir)
                if len(os.listdir(full_store_dir)) == FLAGS.population_size:
                    compare_list = [19970520] * len(os.listdir(full_store_dir))
                    for filename in os.listdir(full_store_dir):
                        current_index = int(filename[4])
                        per = np.load(full_store_dir + "/" + filename)
                        per_mean = np.mean(per)
                        compare_list[current_index] = per_mean
                    max_index = compare_list.index(max(compare_list))
                    min_index = compare_list.index(min(compare_list))

                    if FLAGS.population_index == min_index:
                        # TODO: exploit
                        print("Compare_list:", compare_list)
                        print("Pop %d// Exploit from pop %d" % (FLAGS.population_index, max_index))
                        shutil.rmtree(FLAGS.logdir + "/" + str(FLAGS.population_index) + "/agent")
                        if FLAGS.population_size > 1:  # only considered for one population testing part
                            shutil.copytree(FLAGS.logdir + "/" + str(max_index) + "/agent",
                                            FLAGS.logdir + "/" + str(FLAGS.population_index) + "/agent")
                            shutil.copyfile(FLAGS.logdir + "/" + str(max_index) + '/pbt_parameters.txt',
                                            FLAGS.logdir + "/" + str(FLAGS.population_index) + '/pbt_parameters.txt')

                    # TODO: explore (based on the special setting of IMPALA)
                    f = open(FLAGS.logdir + "/" + str(FLAGS.population_index) + '/pbt_parameters.txt', 'r')
                    str_file = f.read()
                    loaded_parameters_dict = eval(str_file)
                    f.close()
                    tmp_key_list = list(loaded_parameters_dict.keys())
                    tmp_value_list = list(loaded_parameters_dict.values())
                    for key, value in zip(tmp_key_list, tmp_value_list):
                        modified_proportion = random.sample([1, 1, 1, 1, 1.2, 1 / 1.2], 1)[
                            0]  # permute probability is 33%
                        permuted_value = value * modified_proportion
                        setattr(FLAGS, key, permuted_value)
                        loaded_parameters_dict[key] = permuted_value
                    with open(FLAGS.logdir + "/" + str(FLAGS.population_index) + '/all_hyper_parameters.txt',
                              'a+') as file:
                        print("\n-----population based training\n", loaded_parameters_dict, file=file)

                continue
            else:
                break
        print("\nTRAIN finished")

    else:
        level_names = list(LEVEL_MAPPING.values())
        action_set = environments.get_action_set(level_name=level_names[0])
        full_store_dir = FLAGS.logdir + str(FLAGS.store_dir)

        if len(os.listdir(full_store_dir)) == FLAGS.population_size:
            # TODO：test with the results of the PBT
            compare_list = [19970520] * len(os.listdir(full_store_dir))
            for filename in os.listdir(full_store_dir):
                current_index = int(filename[4])
                per = np.load(full_store_dir + "/" + filename)
                per_mean = np.mean(per)
                compare_list[current_index] = per_mean
            max_index = compare_list.index(max(compare_list))
            print("The best population index is: ", max_index)

            test(action_set, level_names, best_population_index=max_index)
        else:
            test(action_set, level_names)

        env_row = 0
        df = pd.DataFrame(columns=["test_environment",
                                   "data_collection_ratio", "dcr_min", "dcr_max",
                                   "geographical_fairness", "gf_min", "gf_max",
                                   "energy consumption", "ec_min", "ec_max",
                                   "efficiency", "eff_min", "eff_max", ])
        all_dcr_list=[]
        all_gf_list=[]
        all_ec_list=[]
        all_eff_list=[]
        for pfilename in os.listdir(FLAGS.logdir + "/test_details"):
            if ".txt" in pfilename:
                with open(FLAGS.logdir + "/test_details/" + pfilename, "r") as file:
                    context = file.read()
                    debug_dict = eval(context)

                env_name = pfilename[:-4]
                dcr_list = debug_dict['data_collection']
                gf_list = debug_dict['normal_fairness']
                ec_list = debug_dict['use_energy']
                eff_list = debug_dict['efficiency']

                all_dcr_list.append(dcr_list)
                all_gf_list.append(gf_list)
                all_ec_list.append(ec_list)
                all_eff_list.append(eff_list)

                df.loc[env_row] = [
                    env_name,
                    np.mean(dcr_list), np.min(dcr_list), np.max(dcr_list),
                    np.mean(gf_list), np.min(gf_list), np.max(gf_list),
                    np.mean(ec_list), np.min(ec_list), np.max(ec_list),
                    np.mean(eff_list), np.min(eff_list), np.max(eff_list),
                ]
                env_row += 1

        df.sort_values("data_collection_ratio", inplace=True)

        mean_all_dcr_list=np.mean(all_dcr_list,axis=0)
        mean_all_gf_list=np.mean(all_gf_list,axis=0)
        mean_all_ec_list=np.mean(all_ec_list,axis=0)
        mean_all_eff_list=np.mean(all_eff_list,axis=0)

        df.loc[env_row] = [
            "Overall performance",
            np.mean(mean_all_dcr_list), np.min(mean_all_dcr_list), np.max(mean_all_dcr_list),
            np.mean(mean_all_gf_list), np.min(mean_all_gf_list), np.max(mean_all_gf_list),
            np.mean(mean_all_ec_list), np.min(mean_all_ec_list), np.max(mean_all_ec_list),
            np.mean(mean_all_eff_list), np.min(mean_all_eff_list), np.max(mean_all_eff_list),
        ]

        df.to_csv(FLAGS.logdir + "/multi-task performance.csv", index=0)
        print("\nTEST finished")


if __name__ == '__main__':
    tf.app.run()
