from gym.envs.registration import register

register(
    id='flatlands-v0',
    entry_point='gym_flatlands.envs:FlatlandsEnv',
)
