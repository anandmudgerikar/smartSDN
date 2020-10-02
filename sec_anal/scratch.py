
import random
import time
import pickle
import numpy as np
import pandas as pd

balance_data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/pcaps/mal_fixed_interval.csv" ,sep=',', header=0)

print(balance_data.mean(axis=0).values[0])