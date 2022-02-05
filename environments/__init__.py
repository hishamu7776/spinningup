from gym.envs.registration import register

register(
    id='InvPendulum-v0',
    entry_point='environments.envs:PendulumEnv',
    max_episode_steps=200
)

register(
    id='SafePendulum-v0',
    entry_point='environments.envs:SafePendulumEnv',
    max_episode_steps=200
)

