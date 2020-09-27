import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from ddpg import DDPG

from tensorflow.python.keras.backend import set_session
from networks import get_session

def main():

    set_session(get_session())

    state_dim = (40,)
    action_dim = 10
    act_range = 1
    batch_size = 100

    algo = DDPG(action_dim, state_dim, act_range, batch_size)

    # Train
    stats = algo.train(batch_size)

    # algo.plot_graphs()

    # # Export results to CSV
    # #if(args.gather_stats):
    # df = pd.DataFrame(np.array(stats))
    # #df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')
    # print(df)
    # # Save weights and close environments
    # exp_dir = '{}/models/'.format(args.type)
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)
    #
    #export_path = ""
    #algo.save_weights(export_path)
    # env.env.close()

if __name__ == "__main__":
    main()