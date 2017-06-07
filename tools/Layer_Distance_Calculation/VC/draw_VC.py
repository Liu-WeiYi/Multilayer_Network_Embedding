#coding: utf-8

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import math

layers = ['1.txt','2.txt','3.txt']
graphs = []
nodes_infor = [[1,2,3,4,5,6,7,8,9,10,11,12],[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
nodes_infor1 = [str(i) for i in nodes_infor[0]]
nodes_infor2 = [str(i) for i in nodes_infor[1]]
nodes_infor = []
nodes_infor.append(nodes_infor1)
nodes_infor.append(nodes_infor2)

for l in layers:
    with open(l,'r+') as f:
        graph = nx.Graph(name=l)
        for line in f.readlines():
            src,dst = line.strip().split()
            graph.add_node(src)
            graph.add_node(dst)
            graph.add_edge(src,dst)
        graphs.append(graph)

merged_graph = nx.Graph(name='merged')
for g in graphs:
    merged_graph.add_nodes_from(g.nodes())
    merged_graph.add_edges_from(g.edges())


# defin pos
pos = {}
node_info_flat = nodes_infor[0] + nodes_infor[1]
delta_theta = 36  0/len(node_info_flat)
init_theta = 90
r = 1
for i in node_info_flat:
    if int(i) < 13:
        theta = init_theta + int(i)*delta_theta
    else:
        theta = init_theta + int(i)*delta_theta + 30
    x = r*math.sin(theta)
    y = r*math.cos(theta)
    pos[i] = [x,y]



graphs.append(merged_graph)

for g in graphs:
    plt.figure(g.name)
    nx.draw_networkx_nodes(g,pos,node_size=150,nodelist=list(set(nodes_infor[0])&set(g.nodes())),node_color='r',node_shape='o',alpha=0.8)
    nx.draw_networkx_nodes(g,pos,node_size=150,nodelist=list(set(nodes_infor[1])&set(g.nodes())),node_color='b',node_shape='D',alpha=0.8)
    nx.draw_networkx_edges(g,pos)
    nx.draw_networkx_labels(g,pos,font_size=8)
    plt.axis('off')
    plt.savefig(g.name+'.pdf')
plt.show()
