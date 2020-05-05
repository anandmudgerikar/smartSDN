from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.net import Mininet, CLI
from mininet.node import OVSKernelSwitch, Host
from mininet.link import TCLink, Link
from mininet.log import setLogLevel, info

import time

class SingleSwitchTopo(Topo):
    "Single switch connected to n hosts."
    def build(self, n=2):
        switch = self.addSwitch('s1')
        # Python's range(N) generates 0..N-1
        for h in range(n):
            host = self.addHost('h%s' % (h + 1))
            self.addLink(host, switch)

class Mininet_Backend():

    n = 3 #Number of nodes in the network

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
        net.getNodeByName('h1').setIP('192.168.10.19')
        net.getNodeByName('h2').setIP('192.168.10.50')
        net.getNodeByName('h3').setIP('172.16.0.1')
        #net.getNodeByName('s1').setIP('192.168.10.19')

        #adding flows for calculating server load and per flow server load
        #normal flow
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')
        #queue flow to load balance
        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=30000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        #forward malcicious looking flow to IDS
        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=10000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:4')

        #testing modifying flows for actions in RL
        #modifying flow to queue from normal forwarding
        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        #net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=10000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')

        net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 add-flow s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.50,nw_dst=192.168.10.19,ip,tp_src=21,actions=output:1')

        #CLI(net)
        return net

    def replay_flows(self,net):
        # Replaying packets from pcaps
        print("replaying flow from user1")
        #print(net.getNodeByName('h1').cmd('tcpreplay -i h1-eth0 pcaps/ &'))
        net.getNodeByName('h1').cmd('tcpreplay -i h1-eth0 pcaps/user1_linux.pcap &')

    def get_serverload(self,net):
        #get server load from each flow
        flows = net.getNodeByName('s1').cmd('ovs-ofctl dump-flows s1')


        for flow in flows.split('\n'):
            if("priority=60000,ip,nw_src=192.168.10.19,nw_dst=192.168.10.50" in flow):
                u1_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])
                #print(f1_curr_server_load)

            if ("priority=60000,ip,nw_src=192.168.10.19,nw_dst=192.168.10.50" in flow):
                u2_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

            if ("priority=60000,ip,nw_src=192.168.10.19,nw_dst=192.168.10.50" in flow):
                u3_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

            if ("priority=60000,ip,nw_src=192.168.10.19,nw_dst=192.168.10.50" in flow):
                u4_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

            if ("priority=60000,ip,nw_src=192.168.10.19,nw_dst=192.168.10.50" in flow):
                u5_curr_server_load = int(flow.split("n_bytes=")[1].split(",")[0])

        print(u1_curr_server_load, u2_curr_server_load, u3_curr_server_load, u4_curr_server_load, u5_curr_server_load)

        return (u1_curr_server_load,u2_curr_server_load,u3_curr_server_load,u4_curr_server_load,u5_curr_server_load)

    def stop_test(self,net):
        net.stop()

if __name__ == '__main__':
    # Tell mininet to print useful information
    setLogLevel('info')
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