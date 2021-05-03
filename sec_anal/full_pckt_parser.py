import dpkt
import ipaddress
import datetime
import glob, os
import pickle

os.chdir("/home/anand/ids_dataset/pcaps/205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/newpcaps/")

with open("/home/anand/ids_dataset/pcaps/benign_win7_full_ip_dump.pickle", "wb") as fp:

    for file in glob.glob("*.pcap"):

        #file = "/home/anand/ids_dataset/pcaps/205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/newpcaps/newWednesday-WorkingHours.pcap"
        f = open(file,'rb')
        pcap = dpkt.pcap.Reader(f)



        #print(file)

        if("Thursday" in file):
            label = "web"
        elif("Wednesday" in file):
            label = "dos"
        elif("Tuesday" in file):
            label = "bruteforce"
        elif("Friday" in file):
            label = "ddos"
        else:
            label = "unknown"

        # print(label)

        # reset = 0
        # prev_ts = 0
        # interval = 0
        # episode_start = 0
        # episode_begin = False

        # # basic features
        # pckt_count_forward = 0
        # pckt_count_back = 0
        # bytes_forward = 0
        # bytes_back = 0

        #new features
        # max_pckt_size_forward = 0
        # max_pckt_size_back = 0
        # min_pckt_size_forward = 0
        # min_pckt_size_back = 0
        # avg_pckt_size_forward = 0
        # avg_pckt_size_back = 0


        #parameters
        # STATE_INTERVAL = 1
        # EPISODE_LEN = 60
        user_ips = {'192.168.10.9'} #,'192.168.10.9','192.168.10.25','192.168.10.25,192.168.10.19'
        attacker_ips = {'172.16.0.1'}

        for ts, buf in pcap:

            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data

            if not isinstance(eth.data, dpkt.ip.IP):
                #print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
                continue

            #print(ipaddress.ip_address(ip.src), ipaddress.ip_address(ip.dst))
            for user_ip in user_ips:
                if(ipaddress.ip_address(ip.dst) == ipaddress.ip_address('192.168.10.50') and ipaddress.ip_address(ip.src) == ipaddress.ip_address(user_ip)):
                    pickle.dump((eth.data,label), fp)

                # if(ipaddress.ip_address(ip.src) == ipaddress.ip_address('192.168.10.50') and ipaddress.ip_address(ip.dst) == ipaddress.ip_address(user_ip)):
                #     pckt_count_back +=1
                #     bytes_back += len(ip)

        f.close()

