import glob, os


os.chdir("/home/anand/ids_dataset/pcaps/")
for file in glob.glob("*.pcap"):
    os.system('tcpdump -r '+file+' -w new'+file)
