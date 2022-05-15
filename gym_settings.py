# coding=utf-8
"""Utilities for Gym."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
from mcs_envs.crazy_env.crazy_data_collection import Env

LEVEL_MAPPING = collections.OrderedDict([
    ('CrazyMCS-1', 'CrazyMCS-1'),

    ('CrazyMCS-2', 'CrazyMCS-2'), ('CrazyMCS-3', 'CrazyMCS-3'), ('CrazyMCS-4', 'CrazyMCS-4'),
    ('CrazyMCS-5', 'CrazyMCS-5'),

    ('CrazyMCS-6', 'CrazyMCS-6'), ('CrazyMCS-7', 'CrazyMCS-7'), ('CrazyMCS-8', 'CrazyMCS-8'),
    ('CrazyMCS-9', 'CrazyMCS-9'), ('CrazyMCS-10', 'CrazyMCS-10'),

    ('CrazyMCS-11', 'CrazyMCS-11'), ('CrazyMCS-12', 'CrazyMCS-12'), ('CrazyMCS-13', 'CrazyMCS-13'),
    ('CrazyMCS-14', 'CrazyMCS-14'), ('CrazyMCS-15', 'CrazyMCS-15'),

    ('CrazyMCS-16', 'CrazyMCS-16'), ('CrazyMCS-17', 'CrazyMCS-17'), ('CrazyMCS-18', 'CrazyMCS-18'),
    ('CrazyMCS-19', 'CrazyMCS-19'), ('CrazyMCS-20', 'CrazyMCS-20'),
    # ('CrazyMCS-0', 'CrazyMCS-0'),
    # ('PongNoFrameskip-v4', 'PongNoFrameskip-v4'),
    # ('BreakoutNoFrameskip-v4', 'BreakoutNoFrameskip-v4'),
    # ('Pendulum-v0', 'Pendulum-v0'),
    # ('CartPole-v0', 'CartPole-v0'),
    # ('CarRacing-v0', 'CarRacing-v0'),
    # ('LunarLanderContinuous-v2','LunarLanderContinuous-v2'),
    # ('BipedalWalker-v2','BipedalWalker-v2'),taertudaxue
])

MCSList = ['CrazyMCS-0', 'CrazyMCS-1', 'CrazyMCS-2', 'CrazyMCS-3', 'CrazyMCS-4', 'CrazyMCS-5', 'CrazyMCS-6',
           'CrazyMCS-7', 'CrazyMCS-8', 'CrazyMCS-9', 'CrazyMCS-10', 'CrazyMCS-11', 'CrazyMCS-12', 'CrazyMCS-13',
           'CrazyMCS-14', 'CrazyMCS-15', 'CrazyMCS-16', 'CrazyMCS-17', 'CrazyMCS-18', 'CrazyMCS-19', 'CrazyMCS-20']

if list(LEVEL_MAPPING.keys())[0] in MCSList:
    env = Env()
    action_type = type(env.action_space[0]).__name__
else:
    env = gym.make(list(LEVEL_MAPPING.keys())[0])
    action_type = type(env.action_space).__name__

if action_type == "Discrete":
    env_type = "D"
    if list(LEVEL_MAPPING.keys())[0] in MCSList:
        action_space_list = [env.action_space[0].n for _ in list(LEVEL_MAPPING.keys())]
    else:
        action_space_list = [gym.make(level_name_i).action_space.n for level_name_i in list(LEVEL_MAPPING.keys())]
    print("action_set:", max(action_space_list))
elif action_type == "Box":
    env_type = "C"
    action_bound = env.action_space.high[0]
    print("action_set:", env.action_space.shape[0])
else:
    env_type = "Unknown"

print("type:", env_type)

AtariList = ["AlienNoFrameskip-v4",
             "AmidarNoFrameskip-v4",
             "AssaultNoFrameskip-v4",
             "AsterixNoFrameskip-v4",
             "AtlantisNoFrameskip-v4",
             "BankHeistNoFrameskip-v4",
             "BattleZoneNoFrameskip-v4",
             "BeamRiderNoFrameskip-v4",
             "BerzerkNoFrameskip-v4",
             "BowlingNoFrameskip-v4",
             "BoxingNoFrameskip-v4",
             "BreakoutNoFrameskip-v4",
             "CarnivalNoFrameskip-v4",
             "CentipedeNoFrameskip-v4",
             "ChopperCommandNoFrameskip-v4",
             "CrazyClimberNoFrameskip-v4",
             "DefenderNoFrameskip-v4",
             "DemonAttackNoFrameskip-v4",
             "DoubleDunkNoFrameskip-v4",
             "ElevatorActionNoFrameskip-v4",
             "EnduroNoFrameskip-v4",
             "FishingDerbyNoFrameskip-v4",
             "FreewayNoFrameskip-v4",
             "FrostbiteNoFrameskip-v4",
             "GopherNoFrameskip-v4",
             "GravitarNoFrameskip-v4",
             "HeroNoFrameskip-v4",
             "IceHockeyNoFrameskip-v4",
             "JamesbondNoFrameskip-v4",
             "JourneyEscapeNoFrameskip-v4",
             "KangarooNoFrameskip-v4",
             "KungFuMasterNoFrameskip-v4",
             "KrullNoFrameskip-v4",
             "MontezumaRevengeNoFrameskip-v4",
             "MsPacmanNoFrameskip-v4",
             "NameThisGameNoFrameskip-v4",
             "PhoenixNoFrameskip-v4",
             "PitfallNoFrameskip-v4",
             "PongNoFrameskip-v4",
             "PooyanNoFrameskip-v4",
             "PrivateEyeNoFrameskip-v4",
             "QbertNoFrameskip-v4",
             "RiverraidNoFrameskip-v4",
             "RoadRunnerNoFrameskip-v4",
             "RobotankNoFrameskip-v4",
             "SeaquestNoFrameskip-v4",
             "SkiingNoFrameskip-v4",
             "SolarisNoFrameskip-v4",
             "SpaceInvadersNoFrameskip-v4",
             "StarGunnerNoFrameskip-v4",
             "TennisNoFrameskip-v4",
             "TimePilotNoFrameskip-v4",
             "TutankhamNoFrameskip-v4",
             "UpNDownNoFrameskip-v4",
             "VentureNoFrameskip-v4",
             "VideoPinballNoFrameskip-v4",
             "WizardOfWorNoFrameskip-v4",
             "YarsRevengeNoFrameskip-v4",
             "ZaxxonNoFrameskip-v4",
             ]
