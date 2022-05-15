import os
import log_mcs as Log
from crazy_env.crazy_data_collection import Env

if os.path.exists("./log") is False:
    os.makedirs("./log")
log=Log.Log(level_name="crazy_env",dir="./log")
env=Env(log)

# log.log(ARGUMENTS)

start_env=env.reset()

print(env.observation_space.shape)
print(env.action_space)
print('Starting a new TEST iterations...')
print('Log_dir:', env.log_dir)

iteration=0

episode_rewards = [0.0]   # sum of rewards for all agents
indicator = [0] * env.n  # TODO:状态指示器
episode_step = 0
meaningful_fill = [0] * env.n
meaningful_get = [0] * env.n

while iteration<1000:
    # action_n = np.random.uniform(low=-1, high=1, size=(len(env.action_space),env.action_space[0].shape[0]))
    # action_n=[action_space_i.sample()for action_space_i in env.action_space]
    action_n=[12 for action_space_i in env.action_space]

    new_obs_n, rew_n, done_n, info_n, indicator = env.step(actions=action_n, indicator=indicator)

    log.step_information(action_n, env, episode_step, iteration, meaningful_fill, meaningful_get,
                         indicator)

    indicator = [0] * env.n

    obs_n = new_obs_n
    done = done_n
    episode_step += 1
    terminal = (episode_step >= 300)

    episode_rewards[-1] += rew_n  # 每一个step的总reward


    if done or terminal:
        print('\n%d th episode:\n' % iteration)
        print('\tobstacle collisions:', env.walls)
        print('\tdata collection:', env.collection / env.totaldata)
        print('\treminding energy:', env.energy)
        # log.draw_path(env, iteration)
        log.draw_path(env, iteration, meaningful_fill, meaningful_get)
        iteration += 1

        meaningful_fill = [0] * env.n
        meaningful_get = [0] * env.n
        obs_n = env.reset()
        episode_step = 0
        episode_rewards.append(0)


