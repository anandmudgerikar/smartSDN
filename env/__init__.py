from gym.envs.registration import register

register(
    id='sdn-v0',
    entry_point='env.sdn_gym:SDN_Gym',
)

register(
    id='attack-sig-v0',
    entry_point='env.attack_sig_gym:Attack_Sig_Gym',
)

register(
    id='sdn-routing-v0',
    entry_point='env.sdn_routing_gym:SDN_Routing_Gym',
)