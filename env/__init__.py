from gym.envs.registration import register

register(
    id='sdn-v0',
    entry_point='env.SimulatedSDNRateControlGym:SimulatedSDNRateControlGym',
)

register(
    id='attack-sig-v0',
    entry_point='env.MininetSDNRoutingGym:MininetSDNRoutingGym',
)

register(
    id='sdn-routing-v0',
    entry_point='env.SimulatedSDNRoutingGym:SimulatedSDNRoutingGym',
)