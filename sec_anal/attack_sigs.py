import dpkt
import ipaddress

f = open('/home/anand/ids_dataset/new_wed.pcap','rb')
pcap = dpkt.pcap.Reader(f)

reset = 0
prev_ts = 0
pckt_count_forward = 0
pckt_count_back = 0
bytes_forward = 0
bytes_back = 0

for ts, buf in pcap:
 #maintaining window of 10 secs
 if(ts > prev_ts + 10):
  prev_ts = ts
  if(pckt_count_forward > 0 or pckt_count_back > 0):
   print(pckt_count_forward,bytes_forward,pckt_count_back,bytes_back)
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

 if(ipaddress.ip_address(ip.dst) == ipaddress.ip_address('192.168.10.50') and ipaddress.ip_address(ip.src) == ipaddress.ip_address('172.16.0.10')):
  pckt_count_forward +=1
  bytes_forward += len(ip)

 if(ipaddress.ip_address(ip.src) == ipaddress.ip_address('192.168.10.50') and ipaddress.ip_address(ip.dst) == ipaddress.ip_address('172.16.0.10')):
  pckt_count_back +=1
  bytes_back += len(ip)

f.close()
