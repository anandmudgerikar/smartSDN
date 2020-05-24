import sys
sys.path.append("/usr/lib/python2.7/dist-packages/")
#sys.path.append("home/anand/mininet/mininet/")

# from log import setLogLevel
from mininet.net import Mininet
from mininet.topo import Topo

# from mininet.topo import Topo
# from mininet.net import Mininet
# from mininet.log import setLogLevel


import time

class SingleSwitchTopo(Topo):
    "Single switch connected to n hosts."
    def build(self, n=6):
        switch = self.addSwitch('s1')
        # Python's range(N) generates 0..N-1
        for h in range(n):
            host = self.addHost('h%s' % (h + 1))
            self.addLink(host, switch)

class Mininet_Backend():

    n = 3 #Number of nodes in the network
    u1_curr_server_load = 0
    u2_curr_server_load = 0
    u3_curr_server_load = 0
    u4_curr_server_load = 0
    u5_curr_server_load = 0

    def startTest(self):
        "Create and test a simple network"
        topo = SingleSwitchTopo(self.n)
        net = Mininet(topo, ipBase='192.168.10.0/8')
        mn = net.start()
        # print("Dumping host connections")
        # dumpNodeConnections(net.hosts)
        # print("Testing network connectivity")
        #net.pingAll()

        #setting ips according to pcaps
        #users
        net.getNodeByName('h1').setIP('192.168.10.19') #user 1 - linux
        net.getNodeByName('h2').setIP('192.168.10.14') #user 2 - win10
        net.getNodeByName('h3').setIP('192.168.10.25') #user 3 - mac
        net.getNodeByName('h4').setIP('192.168.10.9') #user 4 - win7

        #attackers
        net.getNodeByName('h5').setIP('172.16.0.1') # attacker 1 - kali

        #servers
        net.getNodeByName('h6').setIP('192.168.10.50') #server 1 - ubuntu

        #adding flows for calculating server load and per flow server load
        #normal flow forwarding: for users 1-4 to server
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.14,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.25,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.9,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')
        #queue flow to load balance
        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=30000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        #forward malcicious looking flow to IDS
        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=10000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:4')

        #testing modifying flows for actions in RL
        #modifying flow to queue from normal forwarding
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        # net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.14,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        # net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.25,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        # net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.9,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        # net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=10000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')

        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.50,nw_dst=192.168.10.19,ip,tp_src=21,actions=output:1')

        #CLI(net)
        return net

    def replay_flows(self,net):
        # Replaying packets from pcaps
        print("replaying flow from user1")
        #print(net.getNodeByName('h1').cmd('tcpreplay -i h1-eth0 pcaps/ &'))
        net.getNodeByName('h1').cmd('tcpreplay -i h1-eth0 pcaps/user1_linux.pcap &')
        net.getNodeByName('h2').cmd('tcpreplay -i h1-eth0 pcaps/user2_win10.pcap &')
        net.getNodeByName('h3').cmd('tcpreplay -i h1-eth0 pcaps/user3_mac.pcap &')
        net.getNodeByName('h4').cmd('tcpreplay -i h1-eth0 pcaps/user4_win7.pcap &')

        net.getNodeByName('h5').cmd('tcpreplay -i h1-eth0 pcaps/attacker_kali.pcap &')

    def get_serverload(self,net):
        #get server load from each flow
        flows = net.getNodeByName('s1').cmd('ovs-ofctl dump-flows s1')


        for flow in flows.split('\n'):
            if("priority=60000,ip,nw_src=192.168.10.19,nw_dst=192.168.10.50" in flow):
                self.u1_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])
                #print(f1_curr_server_load)

            if ("priority=60000,ip,nw_src=192.168.10.14,nw_dst=192.168.10.50" in flow):
                self.u2_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

            if ("priority=60000,ip,nw_src=192.168.10.25,nw_dst=192.168.10.50" in flow):
                self.u3_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

            if ("priority=60000,ip,nw_src=192.168.10.9,nw_dst=192.168.10.50" in flow):
                self.u4_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

            if ("priority=60000,ip,nw_src=172.16.0.1,nw_dst=192.168.10.50" in flow):
                self.u5_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

        #print(u1_curr_server_load, u2_curr_server_load, u3_curr_server_load, u4_curr_server_load, u5_curr_server_load)

        return (self.u1_curr_server_load,self.u2_curr_server_load,self.u3_curr_server_load,self.u4_curr_server_load,self.u5_curr_server_load)

    def stop_test(self,net):
        net.stop()

if __name__ == '__main__':
    # Tell mininet to print useful information
    mn_backend = Mininet_Backend()
    curr_net = mn_backend.startTest()
    mn_backend.replay_flows(curr_net)

    #sleeping for flows to populate
    print("sleeping")
    time.sleep(50)

    #printing all flows for testing
    flows = curr_net.getNodeByName('s1').cmd('ovs-ofctl dump-flows s1')
    print(flows)

    #printing server loads from each user
    print("Current server load from users:"+str(mn_backend.get_serverload(curr_net)))
    mn_backend.stop_test(curr_net)