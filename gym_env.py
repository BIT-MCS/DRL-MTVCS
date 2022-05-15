import gym
import gym.spaces
import numpy as np
import random

from atari_wrappers import make_atari, wrap_deepmind
from gym_settings import AtariList


class TransposeWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.transpose(np.array(observation), axes=(2, 0, 1))


class NoRwdResetEnv(gym.Wrapper):
    def __init__(self, env, no_reward_thres):
        """Reset the environment if no reward received in N steps
    """
        gym.Wrapper.__init__(self, env)
        self.no_reward_thres = no_reward_thres
        self.no_reward_step = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward == 0.0:
            self.no_reward_step += 1
        else:
            self.no_reward_step = 0
        if self.no_reward_step > self.no_reward_thres:
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.no_reward_step = 0
        return obs


def make_final(env_name, env_id=-1, flags=None, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
    env_log = None

    if env_name in AtariList:
        env = wrap_deepmind(make_atari(env_name), episode_life, clip_rewards, frame_stack, scale)
        env = TransposeWrapper(env)
        env = NoRwdResetEnv(env, no_reward_thres=1000)
    elif env_name == 'CrazyMCS-0':  # debug
        from mcs_envs.crazy_env.crazy_data_collection import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-1':  # debug
        from mcs_envs.gym_base.data_collection_1 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-2':  # debug
        from mcs_envs.gym_base.data_collection_2 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-3':  # debug
        from mcs_envs.gym_base.data_collection_3 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-4':  # debug
        from mcs_envs.gym_base.data_collection_4 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-5':  # debug
        from mcs_envs.gym_base.data_collection_5 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-6':  # debug
        from mcs_envs.gym_base.data_collection_6 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-7':  # debug
        from mcs_envs.gym_base.data_collection_7 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-8':  # debug
        from mcs_envs.gym_base.data_collection_8 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-9':  # debug
        from mcs_envs.gym_base.data_collection_9 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-10':  # debug
        from mcs_envs.gym_base.data_collection_10 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-11':  # debug
        from mcs_envs.gym_base.data_collection_11 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-12':  # debug
        from mcs_envs.gym_base.data_collection_12 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-13':  # debug
        from mcs_envs.gym_base.data_collection_13 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-14':  # debug
        from mcs_envs.gym_base.data_collection_14 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-15':  # debug
        from mcs_envs.gym_base.data_collection_15 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-16':  # debug
        from mcs_envs.gym_base.data_collection_16 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-17':  # debug
        from mcs_envs.gym_base.data_collection_17 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-18':  # debug
        from mcs_envs.gym_base.data_collection_18 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-19':  # debug
        from mcs_envs.gym_base.data_collection_19 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()
    elif env_name == 'CrazyMCS-20':  # debug
        from mcs_envs.gym_base.data_collection_20 import Env
        import log_mcs as Log
        if env_id == 0:
            if flags.mode == 'train':
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + str(flags.population_index))
            else:
                env_log = Log.Log(level_name=env_name, dir=flags.logdir + "/" + "test_details")
            env = Env(env_log)
        else:
            env = Env()

    else:
        env = gym.make(env_name)
    return env, env_log


# def test_action_set(level_name):
#     dummy_env = make_final(level_name)
#     if env_type == 'D':
#         return dummy_env.action_space.n
#     else:
#         return dummy_env.action_space.shape[0]

if __name__ == '__main__':
    # env = make_final('BreakoutNoFrameskip-v4', True, True, False, False)
    # env = make_final('SeaquestNoFrameskip-v4', True, True, False, False)
    # env = make_final('BreakoutNoFrameskip-v4', True, True, True, False)
    # env = make_final('MontezumaRevengeNoFrameskip-v4', True, True, False, False)
    env = make_final('CartPole-v0')
    # env = make_final('LunarLanderContinuous-v2')

    print(env.observation_space)

    print(env.action_space)

    n_game = 5
    epi_max_len = 4096

    game_idx = 0
    for game_idx in range(n_game):
        # start
        obs = env.reset()
        for i in range(epi_max_len):
            action = random.randint(0, env.action_space.n - 1)
            # action = np.random.uniform(low=env.action_space.low[0], high=env.action_space.high[0],size=env.action_space.shape)
            obs, rwd, term, info = env.step(action)  # discrete
            if rwd != 0.0:
                print('reward:', rwd, type(rwd))
            if term:
                print('info:', info)
                print('obs:', obs.dtype)
                print(i)
                break
        # close
        print("game_idx: %d" % game_idx)
