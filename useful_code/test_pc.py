# coding=utf-8
import tensorflow as tf
import sonnet as snt
import collections
import numpy as np
from trfl import action_value_ops
from trfl import base_ops

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
            tf.shape(avg_abs_diff)[1], tf.shape(avg_abs_diff)[2]  #H',W'
        ],
        name="pseudo_rewards")  # [T,B,H',W'].
    sequence_batch = abs_observation_diff.get_shape()[:2]   # sequence,batch
    new_height_width = avg_abs_diff.get_shape()[1:]   # H',W'
    pseudo_rewards.set_shape(sequence_batch.concatenate(new_height_width))
    return pseudo_rewards


def pixel_control_loss(
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
    expanded_actions = tf.expand_dims(tf.expand_dims(actions, -1), -1)  #(20,1)->(20,1,1,1)
    a_tm1 = tf.tile(
        expanded_actions, multiples=[1, 1] + height_width)  # [T,B,H,W].
    a_tm1 = tf.reshape(a_tm1, [sequence_length, -1])  # [T,BHW].
    # We similarly expand-and-tile the discount to [T,BHW].
    discount_factor = tf.convert_to_tensor(discount_factor)  #(sequence,batch)
    if discount_factor.shape.ndims == 0:
        pcont_t = tf.reshape(discount_factor, [1, 1])  # [1,1].
        pcont_t = tf.tile(pcont_t, tf.shape(a_tm1))  # [T,BHW].
    elif discount_factor.shape.ndims == 2:
        tiled_pcont = tf.tile(  #(20,1,20,20)
            tf.expand_dims(tf.expand_dims(discount_factor, -1), -1),
            [1, 1] + height_width)
        pcont_t = tf.reshape(tiled_pcont, [sequence_length, -1])  #(20,400)
    else:
        raise ValueError(
            "The discount_factor must be a scalar or a tensor of rank 2."
            "instead is a tensor of shape {}".format(
                discount_factor.shape.as_list()))
    # Compute a QLambda loss of shape [T,BHW]
    loss, _ = action_value_ops.qlambda(q_tm1, a_tm1, r_t, pcont_t, q_t, lambda_=1)  #[T,BHW]
    # Take sum over sequence, sum over cells.
    expanded_shape = [sequence_length, batch_size] + height_width
    spatial_loss = tf.reshape(loss, expanded_shape)  # [T,B,H,W].
    # Return.
    extra = PixelControlExtra(
        spatial_loss=spatial_loss, pseudo_rewards=pseudo_rewards)
    return base_ops.LossOutput(
        tf.reduce_mean(spatial_loss, axis=[0, 2, 3]), extra)  # [B]


# # Configure.
# _cell = 2
# obs_size = (5, 2, 4, 4, 3)
# y = obs_size[2] // _cell
# x = obs_size[3] // _cell
# channels = np.prod(obs_size[4:])  # product
# rew_size = (obs_size[0]-1, obs_size[1], x, y)
#
# # Input data.
# _obs_np = np.random.uniform(size=obs_size)
# _obs_tf = tf.placeholder(tf.float32, obs_size)
#
# # Expected pseudo-rewards.
# abs_diff = np.absolute(_obs_np[1:] - _obs_np[:-1])  # 后几个-前几个
# abs_diff = abs_diff.reshape((-1,) + obs_size[2:4] + (channels,))
# abs_diff = abs_diff.reshape((-1, y, _cell, x, _cell, channels))
# avg_abs_diff = abs_diff.mean(axis=(2, 4, 5))
# _expected_pseudo_rewards = avg_abs_diff.reshape(rew_size)
#
#
# pseudo_rewards_tf = pixel_control_rewards(_obs_tf, _cell)
#
# with tf.Session() as sess:
#     reward=sess.run(pseudo_rewards_tf, feed_dict={_obs_tf: _obs_np})
#
# print(_expected_pseudo_rewards.shape)
# print(reward.shape)

# TODO:
seq_length = 20
batch_size = 16
num_actions = 25
obs_shape = (80, 80, 3)
action_shape = (20, 20, num_actions)
discount = 0.9
cell_size = 4
scale = 1.0

# Create ops to feed actions and rewards.
observations_ph = tf.placeholder(  # (sequence+1,batch,height,width,channel)
    shape=(seq_length + 1, batch_size) + obs_shape, dtype=tf.float32)
action_values_ph = tf.placeholder(  # (sequence+1,batch,height,width,channel)
    shape=(seq_length + 1, batch_size) + action_shape, dtype=tf.float32)
actions_ph = tf.placeholder(  # (sequence,batch)
    shape=(seq_length, batch_size), dtype=tf.int32)

# Observations.
observations = np.random.uniform(size=(seq_length+1,batch_size) + obs_shape)
action_values= np.random.uniform(size=(seq_length+1,batch_size) + action_shape)
actions=np.random.randint(low=0,high=25,size=(seq_length,batch_size))



# # Compute loss for constant discount.
# qa_tm1 = obs3[:, :, action3]
# reward3 = np.mean(np.abs(obs4 - obs3), axis=2)
# qmax_t = np.amax(obs4, axis=2)
# target = reward3 + discount * qmax_t
# error3 = target - qa_tm1
#
# qa_tm1 = obs2[:, :, action2]
# reward2 = np.mean(np.abs(obs3 - obs2), axis=2)
# target = reward2 + discount * target
# error2 = target - qa_tm1
#
# qa_tm1 = obs1[:, :, action1]
# reward1 = np.mean(np.abs(obs2 - obs1), axis=2)
# target = reward1 + discount * target
# error1 = target - qa_tm1
#
# # Compute loss for episode termination with discount 0.
# qa_tm1 = obs1[:, :, action1]
# reward1 = np.mean(np.abs(obs2 - obs1), axis=2)
# target = reward1 + 0. * target
# error1_term = target - qa_tm1
#
# error = np.sum(
#     np.square(error1) + np.square(error2) + np.square(error3)) * 0.5
# error_term = np.sum(
#     np.square(error1_term) + np.square(error2) + np.square(error3)) * 0.5

"""Compute loss for given observations, actions, values, tensor discount."""

zero_discount = tf.zeros((1, batch_size))
non_zero_discount = tf.tile(
    tf.reshape(discount, [1, 1]),
    [seq_length - 1, batch_size])
tensor_discount = tf.concat([zero_discount, non_zero_discount], axis=0)  # (sequence,batch)
loss, _ = pixel_control_loss(
    observations_ph, actions_ph, action_values_ph,
    cell_size, tensor_discount, scale)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    feed_dict = {
        observations_ph: observations,
        action_values_ph: action_values,
        actions_ph: actions}
    loss_np = sess.run(loss, feed_dict=feed_dict)

print("testPixelControlLossTensorDiscount: ", loss_np)
