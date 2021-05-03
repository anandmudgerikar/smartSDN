import pickle
import pandas as pd
import ipaddress

#data = pd.read_pickle('/home/anand/ids_dataset/pcaps/mal_full_ip_dump.pickle')

file = open('/home/anand/ids_dataset/pcaps/benign_linux_full_ip_dump.pickle',"rb")
data = pickle.load(file)

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

items = loadall('/home/anand/ids_dataset/pcaps/benign_win7_full_ip_dump.pickle')

print(sum(1 for _ in items))

# for item in items:
#     print(ipaddress.ip_address(item[0].dst))



# data1 = pickle.load(file)
# data2 = pickle.load(file)
# print(data)
# print(data1)
# print(data2)

#print("Dataset Shape: ", data.shape)
#print("Column Names:", data.columns)

# Printing the dataset obseravtions
#print("Train Dataset: ", data.head())