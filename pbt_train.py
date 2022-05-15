import os
from multiprocessing import Pool
from subprocess import Popen, PIPE


def create_worker(population_index, total_frames, population_size):
    p = Popen([
        'python', 'experiment.py', '--use_pbt=True', '--gpu_id=-1',
        '--population_index={}'.format(population_index),
        '--total_environment_frames={}'.format(int(total_frames)),
        '--population_size={}'.format(int(population_size))])


if __name__ == "__main__":
    population_size = 2
    total_frames = 5e3

    pool = Pool(population_size)

    for i in range(population_size):
        pool.apply_async(func=create_worker,
                         args=(i, total_frames, population_size))

    print("Let's play PBT of IMPALA-MCS now!!")
    pool.close()
    pool.join()
