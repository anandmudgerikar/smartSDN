from gym.envs.registration import register

register(
    id='sdn-v0',
    entry_point='env.sdn_gym:SDN_Gym',
)