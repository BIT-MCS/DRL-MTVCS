import functools
import collections
import multiprocessing
import matplotlib.pyplot as plt
import os
import numpy as np
import sonnet as snt
import tensorflow as tf

from trfl import action_value_ops
from trfl import base_ops


def debug_loss(l1, l2, l3, l4, l5, l6):
    with open('./loss_debug.txt', 'a+') as file:
        print('reward:%f --policy_gradient_loss:%f --baseline_loss:%f --entropy_loss:%f --pixel_control_loss:%f '
              '\n--total_loss:%f ' % (l1, l2, l3, l4, l5, l6), file=file)
    return False

def get_session(sess):
    sess_copy = sess
    session = sess_copy._sess._sess._sess._sess
    return session


def task_evaluation(task_name, env_log,debug_dict):
    if env_log is not None:
        full_path = os.path.join(env_log.full_path, 'Evaluation')
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        plt.plot(debug_dict['data_collection'])
        plt.xlabel("Training episode")
        plt.ylabel("data collection ratio")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.savefig(full_path + '/%s--data_collection_ratio.png' % task_name)
        plt.close()

        plt.plot(debug_dict['efficiency'])
        plt.xlabel("Training episode")
        plt.ylabel("efficiency")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.savefig(full_path + '/%s--efficiency.png' % task_name)
        plt.close()

        plt.plot(debug_dict['fairness'])
        plt.xlabel("Training episode")
        plt.ylabel("fairness")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.savefig(full_path + '/%s--fairness.png' % task_name)
        plt.close()

        plt.plot(debug_dict['normal_fairness'])
        plt.xlabel("Training episode")
        plt.ylabel("normal fairness")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.savefig(full_path + '/%s--normal-fairness.png' % task_name)
        plt.close()

        plt.plot(debug_dict['wall_n'])
        plt.xlabel("Training episode")
        plt.ylabel("wall_n")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.savefig(full_path + '/%s--wall_n.png' % task_name)
        plt.close()

        plt.plot(debug_dict['use_energy'])
        plt.xlabel("Training episode")
        plt.ylabel("use_energy_n")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.savefig(full_path + '/%s--use_energy_n' % task_name)
        plt.close()
    else:
        print("Thread Error! This env is not equipped with logs.")


def get_trainable_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)


def normalize(tensor, stats):
    """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """
    if stats is None:
        return tensor
    return (tensor - stats.mean) / stats.std


def denormalize(tensor, stats):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return tensor
    return tensor * stats.std + stats.mean


def reduce_std(tensor, axis=None, keepdims=False):
    """
    get the standard deviation of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the std over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """
    get the variance of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the variance over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


# ================================================================
# Global session
# ================================================================

def make_session(num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


def single_threaded_session(make_default=False, graph=None):
    """
    Returns a session which will only use a single CPU

    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    return make_session(num_cpu=1, make_default=make_default, graph=graph)


def in_session(func):
    """
    wrappes a function so that it is in a TensorFlow Session

    :param func: (function) the function to wrap
    :return: (function)
    """

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        with tf.Session():
            func(*args, **kwargs)

    return newfunc


ALREADY_INITIALIZED = set()


def initialize(sess=None):
    """
    Initialize all the uninitialized variables in the global scope.

    :param sess: (TensorFlow Session)
    """
    if sess is None:
        sess = tf.get_default_session()
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


# ================================================================
# Theano-like Function
# ================================================================


def function(inputs, outputs, updates=None, givens=None):
    """
    Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs. Just like a Theano function.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
       >>> x = tf.placeholder(tf.int32, (), name="x")
       >>> y = tf.placeholder(tf.int32, (), name="y")
       >>> z = 3 * x + 2 * y
       >>> lin = function([x, y], z, givens={y: 0})
       >>> with single_threaded_session():
       >>>     initialize()
       >>>     assert lin(2) == 6
       >>>     assert lin(x=3) == 9
       >>>     assert lin(2, 2) == 10

    :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
    :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    :param updates: ([tf.Operation] or tf.Operation)
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
    :param givens: (dict) the values known for the output
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        func = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), func(*args, **kwargs)))
    else:
        func = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: func(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        """
        Theano like function

        :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
        :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
            value will also have the same shape.
        :param updates: ([tf.Operation] or tf.Operation)
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
        :param givens: (dict) the values known for the output
        """
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (isinstance(inpt, tf.Tensor) and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    @classmethod
    def _feed_input(cls, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args, sess=None, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = sess.run(self.outputs_update, feed_dict=feed_dict, **kwargs)[:-1]
        return results


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-2, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self._sum = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float64,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.cast(self._sum / self._count, tf.float32)
        self.std = tf.sqrt(tf.maximum(tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean),  # 标准差
                                      1e-2))

    def update(self, data):
        """
        update the running mean and std

        :param data: (np.ndarray) the data
        """
        data = tf.to_double(data)
        data_size = tf.cast(tf.reduce_prod(self.shape), dtype=tf.int32)
        totalvec = tf.zeros([data_size * 2 + 1], dtype=tf.float64)
        addvec = tf.concat(
            [tf.expand_dims(tf.reduce_sum(data), axis=0),
             tf.expand_dims(tf.reduce_sum(tf.square(data)), axis=0),
             tf.expand_dims(tf.to_double(tf.reduce_prod(tf.shape(data))), axis=0)], axis=0)
        totalvec = addvec + totalvec
        tf.assign_add(self._sum, tf.reshape(totalvec[0: data_size], self.shape))
        tf.assign_add(self._sumsq, tf.reshape(totalvec[data_size: 2 * data_size], self.shape))
        tf.assign_add(self._count, totalvec[2 * data_size])


def setup_popart(ret_rms, old_mean, old_std, agent_original_name):
    """
    setup pop-art normalization of the critic output

    See https://arxiv.org/pdf/1602.07714.pdf for details.
    Preserving Outputs Precisely, while Adaptively Rescaling Targets”.
    :rtype: object
    """
    new_std = ret_rms.std
    new_mean = ret_rms.mean

    # FC:fc_net_agent/batch_apply_1/baseline
    # RS:resnet_lstm_agent/batch_apply_1/baseline
    for out_vars in [
        [var for var in get_trainable_vars(agent_original_name + '/batch_apply_1/baseline')]]:  # TODO:注意这里没有超参化
        assert len(out_vars) == 2
        # wieght and bias of the last layer
        weight, bias = out_vars
        assert 'w' in weight.name
        assert 'b' in bias.name
        assert weight.get_shape()[-1] == 1
        assert bias.get_shape()[-1] == 1
        weight.assign(weight * old_std / new_std)
        bias.assign((bias * old_std + old_mean - new_mean) / new_std)


PixelControlExtra = collections.namedtuple(
    "pixel_control_extra", ["spatial_loss", "pseudo_rewards"])


def pixel_control_rewards(observations, cell_size):
    """Calculates pixel control task rewards from observation sequence.
    The observations are first split in a grid of KxK cells. For each cell a
    distinct pseudo reward is computed as the average absolute change in pixel
    intensity for all pixels in the cell. The change in intensity is averaged
    across both pixels and channels (e.g. RGB).
    The `observations` provided to this function should be cropped suitably, to
    ensure that the observations' height and width are a multiple of `cell_size`.
    The values of the `observations` tensor should be rescaled to [0, 1]. In the
    UNREAL agent observations are cropped to 80x80, and each cell is 4x4 in size.
    See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
    Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).
    Args:
      observations: A tensor of shape `[T+1,B,H,W,C...]`, where
        * `T` is the sequence length, `B` is the batch size.
        * `H` is height, `W` is width.
        * `C...` is at least one channel dimension (e.g., colour, stack).
        * `T` and `B` can be statically unknown.
      cell_size: The size of each cell.
    Returns:
      A tensor of pixel control rewards calculated from the observation. The
      shape is `[T,B,H',W']`, where `H'` and `W'` are determined by the
      `cell_size`. If evenly-divisible, `H' = H/cell_size`, and similar for `W`.
    """
    # Calculate the absolute differences across the sequence.
    abs_observation_diff = tf.abs(observations[1:] - observations[:-1])  # (sequence,batch,width,length,channel)
    # Average over cells. abs_observation_diff has shape [T,B,H,W,C...], e.g.,
    # [T,B,H,W,C] if we have a colour channel. We want to use the TF avg_pool
    # op, but it expects 4D inputs. We collapse T and B then collapse all channel
    # dimensions. After pooling, we can then undo the sequence/batch collapse.
    obs_shape = abs_observation_diff.get_shape().as_list()
    # Collapse sequence and batch into one: [TB,H,W,C...].
    abs_diff = tf.reshape(abs_observation_diff, [-1] + obs_shape[2:])  # (20,80,80,3)
    # Merge remaining dimensions after W: [TB,H,W,C'].
    abs_diff = snt.FlattenTrailingDimensions(dim_from=3)(abs_diff)  # 还是你自己
    # Apply the averaging using average pooling and reducing over channel.
    avg_abs_diff = tf.nn.avg_pool(  # (20,20,20,3)
        abs_diff,
        ksize=[1, cell_size, cell_size, 1],
        strides=[1, cell_size, cell_size, 1],
        padding="VALID")  # [TB, H', W', C'].
    avg_abs_diff = tf.reduce_mean(avg_abs_diff, axis=[3])  # [TB,H',W']. (20,20,20)
    # Restore sequence and batch dimensions, and static shape info where possible.
    pseudo_rewards = tf.reshape(
        avg_abs_diff, [
            tf.shape(abs_observation_diff)[0], tf.shape(abs_observation_diff)[1],  # sequence,batch
            tf.shape(avg_abs_diff)[1], tf.shape(avg_abs_diff)[2]  # H',W'
        ],
        name="pseudo_rewards")  # [T,B,H',W'].
    sequence_batch = abs_observation_diff.get_shape()[:2]  # sequence,batch
    new_height_width = avg_abs_diff.get_shape()[1:]  # H',W'
    pseudo_rewards.set_shape(sequence_batch.concatenate(new_height_width))
    return pseudo_rewards


def compute_pixel_control_loss(
        observations, actions, action_values, cell_size, discount_factor,
        scale, crop_height_dim=(None, None), crop_width_dim=(None, None)):
    """Calculate n-step Q-learning loss for pixel control auxiliary task.
    For each pixel-based pseudo reward signal, the corresponding action-value
    function is trained off-policy, using Q(lambda). A discount of 0.9 is
    commonly used for learning the value functions.
    Note that, since pseudo rewards have a spatial structure, with neighbouring
    cells exhibiting strong correlations, it is convenient to predict the action
    values for all the cells through a deconvolutional head.
    See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
    Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).
    Args:
      observations: A tensor of shape `[T+1,B, ...]`; `...` is the observation
        shape, `T` the sequence length, and `B` the batch size. `T` and `B` can
        be statically unknown for `observations`, `actions` and `action_values`.
      actions: A tensor, shape `[T,B]`, of the actions across each sequence.
      action_values: A tensor, shape `[T+1,B,H,W,N]` of pixel control action
        values, where `H`, `W` are the number of pixel control cells/tasks, and
        `N` is the number of actions.
      cell_size: size of the cells used to derive the pixel based pseudo-rewards.
      discount_factor: discount used for learning the value function associated
        to the pseudo rewards; must be a scalar or a Tensor of shape [T,B].
      scale: scale factor for pixels in `observations`.
      crop_height_dim: tuple (min_height, max_height) specifying how   移除不必要信息
        to crop the input observations before computing the pseudo-rewards.
      crop_width_dim: tuple (min_width, max_width) specifying how
        to crop the input observations before computing the pseudo-rewards.
    Returns:
      A namedtuple with fields:
      * `loss`: a tensor containing the batch of losses, shape [B].
      * `extra`: a namedtuple with fields:
          * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
          * `td_error`: batch of temporal difference errors, shape [B].
    Raises:
      ValueError: if the shape of `action_values` is not compatible with that of
        the pseudo-rewards derived from the observations.
    """
    # Useful shapes.
    sequence_length, batch_size = base_ops.best_effort_shape(actions)  # 弹出有用的sequence和batch
    num_actions = action_values.get_shape().as_list()[-1]  # num_action 25
    height_width_q = action_values.get_shape().as_list()[2:-1]  # 20,20
    # Calculate rewards using the observations. Crop observations if appropriate.
    if crop_height_dim[0] is not None:
        h_low, h_high = crop_height_dim
        observations = observations[:, :, h_low:h_high, :]
    if crop_width_dim[0] is not None:
        w_low, w_high = crop_width_dim
        observations = observations[:, :, :, w_low:w_high]
    # Rescale observations by a constant factor.
    observations *= tf.constant(scale)
    # Compute pseudo-rewards and get their shape.
    pseudo_rewards = pixel_control_rewards(observations, cell_size)  # 20,1,20,20
    height_width = pseudo_rewards.get_shape().as_list()[2:]  # 20,20
    # Check that pseudo-rewards and Q-values are compatible in shape.
    if height_width != height_width_q:
        raise ValueError(
            "Pixel Control values are not compatible with the shape of the"
            "pseudo-rewards derived from the observation. Pseudo-rewards have shape"
            "{}, while Pixel Control values have shape {}".format(
                height_width, height_width_q))
    # We now have Q(s,a) and rewards, so can calculate the n-step loss. The
    # QLambda loss op expects inputs of shape [T,B,N] and [T,B], but our tensors
    # are in a variety of incompatible shapes. The state-action values have
    # shape [T,B,H,W,N] and rewards have shape [T,B,H,W]. We can think of the
    # [H,W] dimensions as extra batch dimensions for the purposes of the loss
    # calculation, so we first collapse [B,H,W] into a single dimension.
    q_tm1 = tf.reshape(
        action_values[:-1],  # [T,B,H,W,N]. 前T
        [sequence_length, -1, num_actions],
        name="q_tm1")  # [T,BHW,N].
    r_t = tf.reshape(
        pseudo_rewards,  # [T,B,H,W].
        [sequence_length, -1],
        name="r_t")  # [T,BHW].
    q_t = tf.reshape(
        action_values[1:],  # [T,B,H,W,N]. 后T
        [sequence_length, -1, num_actions],
        name="q_t")  # [T,BHW,N].
    # The actions tensor is of shape [T,B], and is the same for each H and W.
    # We thus expand it to be same shape as the reward tensor, [T,BHW].
    expanded_actions = tf.expand_dims(tf.expand_dims(actions, -1), -1)  # (20,1)->(20,1,1,1)
    a_tm1 = tf.tile(
        expanded_actions, multiples=[1, 1] + height_width)  # [T,B,H,W].
    a_tm1 = tf.reshape(a_tm1, [sequence_length, -1])  # [T,BHW].
    # We similarly expand-and-tile the discount to [T,BHW].
    discount_factor = tf.convert_to_tensor(discount_factor)  # (sequence,batch)
    if discount_factor.shape.ndims == 0:
        pcont_t = tf.reshape(discount_factor, [1, 1])  # [1,1].
        pcont_t = tf.tile(pcont_t, tf.shape(a_tm1))  # [T,BHW].
    elif discount_factor.shape.ndims == 2:
        tiled_pcont = tf.tile(  # (20,1,20,20)
            tf.expand_dims(tf.expand_dims(discount_factor, -1), -1),
            [1, 1] + height_width)
        pcont_t = tf.reshape(tiled_pcont, [sequence_length, -1])  # (20,400)
    else:
        raise ValueError(
            "The discount_factor must be a scalar or a tensor of rank 2."
            "instead is a tensor of shape {}".format(
                discount_factor.shape.as_list()))
    # Compute a QLambda loss of shape [T,BHW]
    loss, _ = action_value_ops.qlambda(q_tm1, a_tm1, r_t, pcont_t, q_t, lambda_=1)  # [T,BHW]
    # Take sum over sequence, sum over cells.
    expanded_shape = [sequence_length, batch_size] + height_width
    spatial_loss = tf.reshape(loss, expanded_shape)  # [T,B,H,W].
    # Return.
    extra = PixelControlExtra(
        spatial_loss=spatial_loss, pseudo_rewards=pseudo_rewards)
    return base_ops.LossOutput(
        tf.reduce_mean(tf.reduce_sum(spatial_loss,axis=[2,3])), extra)  # [B]
