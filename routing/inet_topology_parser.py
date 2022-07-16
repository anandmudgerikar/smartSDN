import collections
import random
from functools import lru_cache
import pandas as pd

class inet_topology_parser:

    def __init__(self):
        self.data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/routing/net_4000", sep='\t', names = ['Node1', 'Node2', 'Weight'])
        self.edge_weights = collections.defaultdict(int)
        self.adj_list = collections.defaultdict(list)
        self.no_of_paths = 10
        self.no_of_nodes = 100
        self.no_security_services = {"Brute":5} #"DDoS":5,"Web":5
        self.security_services_per_node = collections.defaultdict(set)
        self.security_services_per_pair_paths = collections.defaultdict(list)

        #randomly assigning security services to nodes
        for security_service,count in self.no_security_services.items():
            for _ in range(count):
                node_index = random.randint(0,self.no_of_nodes)
                self.security_services_per_node[node_index].add(security_service)

        for index in self.data.index:
            node1 = self.data['Node1'][index]
            node2 = self.data['Node2'][index]
            weight = self.data['Weight'][index]

            if node1 <= self.no_of_nodes and node2 <= self.no_of_nodes:
                self.edge_weights[(node1,node2)] = weight
                self.edge_weights[(node2, node1)] = weight
                self.adj_list[node1].append(node2)
                self.adj_list[node2].append(node1)

    #print(edge_weights)

    @lru_cache(None)
    def find_paths(self,node1,node2):

        #array of 3 candidate paths from node1 to node2
        paths = []
        path_weights = []
        path_not_found = False
        visited = set()
        self.ddos_not_set = True
        sec_services_per_path = []

        # #direct path
        # if (node1,node2) in edge_weights:
        #     paths.append(edge_weights[(node1,node2)])

        @lru_cache(None)
        def dfs(node,weight_so_far,target,curr_path):

            if node in visited or len(path_weights) >= self.no_of_paths or len(curr_path) >= 10:
                return

            # if self.ddos_not_set:
            #     for security_service,_ in self.no_security_services.items():
            #         if security_service in self.security_services_per_node[node]:
            #             sec_services.add(security_service)
            #             self.ddos_not_set = False

            for next_node in self.adj_list[node]:
               if next_node == target and len(path_weights) < self.no_of_paths:
                   weight_so_far += self.edge_weights[(node,next_node)]
                   path_weights.append(weight_so_far)
                   paths.append(list(curr_path) + [next_node])
                   return

               visited.add(node)
               dfs(next_node,weight_so_far+self.edge_weights[(node,next_node)],target,tuple(list(curr_path)+[node]))
               visited.remove(node)

        #indirect paths
        dfs(node1,0,node2,tuple([node1]))
        #print(paths)

        #if less than 10 paths possible
        if len(path_weights) < 10:
            path_weights += [float("inf")]*(10 - len(path_weights))

        while len(paths) < 10:
            paths.append([float("inf")])

        for index,path in enumerate(paths):
            security_services = set()

            if path_weights[index] == float("inf"):
                sec_services_per_path.append(security_services)
            else:
                for node in path:
                    for security_service,_ in self.no_security_services.items():
                        if security_service in self.security_services_per_node[node]:
                            security_services.add(security_service)
                sec_services_per_path.append(security_services)

        #print(paths)
        #print(path_weights)
        return path_weights,sec_services_per_path

# test_parser = inet_topology_parser()
# test_parser.find_paths(4,5)