import dpkt
import ipaddress
import datetime
import glob, os


os.chdir("/home/anand/ids_dataset/pcaps/205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/newpcaps/")
for file in glob.glob("*.pcap"):


    f = open(file,'rb')
    pcap = dpkt.pcap.Reader(f)

    reset = 0
    prev_ts = 0
    pckt_count_forward = 0
    pckt_count_back = 0
    bytes_forward = 0
    bytes_back = 0

    #parameters
    STATE_INTERVAL = 2
    user_ips = {'192.168.10.19','192.168.10.14','192.168.10.9','192.168.10.25'}
    attacker_ips = {'172.16.0.1'}

    for ts, buf in pcap:
        #maintaining window of 10 secs
        if(ts > prev_ts + STATE_INTERVAL):
            prev_ts = ts
            if(pckt_count_forward > 0 or pckt_count_back > 0):
                print(str(datetime.datetime.utcfromtimestamp(ts)),pckt_count_forward,bytes_forward,pckt_count_back,bytes_back)
            pckt_count_forward = 0
            pckt_count_back = 0
            bytes_forward = 0
            bytes_back = 0

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
