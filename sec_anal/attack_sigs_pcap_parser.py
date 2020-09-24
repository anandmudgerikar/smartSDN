import dpkt
import ipaddress
import datetime
import glob, os


os.chdir("/home/anand/ids_dataset/pcaps/205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/newpcaps/")
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

    reset = 0
    prev_ts = 0
    interval = 0
    episode_start = 0

    # basic features
    pckt_count_forward = 0
    pckt_count_back = 0
    bytes_forward = 0
    bytes_back = 0

    #new features
    # max_pckt_size_forward = 0
    # max_pckt_size_back = 0
    # min_pckt_size_forward = 0
    # min_pckt_size_back = 0
    # avg_pckt_size_forward = 0
    # avg_pckt_size_back = 0


    #parameters
    STATE_INTERVAL = 1
    EPISODE_LEN = 60
    user_ips = {'192.168.10.25'} #,'192.168.10.14','192.168.10.9','192.168.10.25,192.168.10.19'
    attacker_ips = {'172.16.0.1'}

    for ts, buf in pcap:
        #maintaining window of STATE_INTERVAL secs
        if(ts > prev_ts + STATE_INTERVAL):
            prev_ts = ts
            interval +=1

            #writing to dataset for new/unknown attacks (building training/testing)
            if(pckt_count_forward > 0 or pckt_count_back > 0):
                # #print(ts, label)
                # if(label == "dos"):
                #     if(ts > 1499261700): #excluding slowloris and slowhttptest dos attacks
                #         print(str(datetime.datetime.utcfromtimestamp(ts)),pckt_count_forward,bytes_forward,pckt_count_back,bytes_back,'Malicious')
                # elif(label == "bruteforce"):
                #     if (ts > 1499174400):  # excluding ftp patator brute force attacks
                #         print(str(datetime.datetime.utcfromtimestamp(ts)), pckt_count_forward, bytes_forward,pckt_count_back, bytes_back,'Malicious')
                # else:
                print(str(datetime.datetime.utcfromtimestamp(ts)),interval, pckt_count_forward, bytes_forward,pckt_count_back, bytes_back,'Malicious')

            if(ts > episode_start + EPISODE_LEN):
                pckt_count_forward = 0
                pckt_count_back = 0
                bytes_forward = 0
                bytes_back = 0
                episode_start = ts
                interval = 0

        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data

        if not isinstance(eth.data, dpkt.ip.IP):
            #print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue

        #print(ipaddress.ip_address(ip.src), ipaddress.ip_address(ip.dst))
        for user_ip in user_ips:
            if(ipaddress.ip_address(ip.dst) == ipaddress.ip_address('192.168.10.50') and ipaddress.ip_address(ip.src) == ipaddress.ip_address(user_ip)):
                pckt_count_forward +=1
                bytes_forward += len(ip)

            if(ipaddress.ip_address(ip.src) == ipaddress.ip_address('192.168.10.50') and ipaddress.ip_address(ip.dst) == ipaddress.ip_address(user_ip)):
                pckt_count_back +=1
                bytes_back += len(ip)

    f.close()
