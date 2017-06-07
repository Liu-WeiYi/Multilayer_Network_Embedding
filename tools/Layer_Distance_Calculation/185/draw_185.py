#coding: utf-8

import pickle
import networkx as nx
import matplotlib.pyplot as plt


layers = ['0_Computer.txt','1_Partners.txt','2_Time.txt']
graphs = []

# nodes_infor = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 43, 44, 45, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 61, 62, 63, 64, 65, 68, 69, 73, 74, 75, 82, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 139, 140, 141, 142, 146, 147, 148, 149, 150, 151, 152, 158, 159, 160, 163], [13, 14, 26, 27, 38, 39, 41, 42, 46, 47, 48, 57, 58, 66, 67, 70, 71, 72, 76, 77, 78, 79, 80, 81, 83, 84, 85, 94, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 143, 144, 145, 153, 154, 155, 156, 157, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190]]
# nodes_infor1 = [str(i) for i in nodes_infor[0]]
# nodes_infor2 = [str(i) for i in nodes_infor[1]]
# nodes_infor = []
# nodes_infor.append(nodes_infor1)
# nodes_infor.append(nodes_infor2)

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

pos = nx.spring_layout(merged_graph)

graphs.append(merged_graph)

for g in graphs:
    plt.figure(g.name)
    # nx.draw_networkx_nodes(g,pos,node_size=150,nodelist=list(set(nodes_infor[0])&set(g.nodes())),node_color='r',node_shape='o',alpha=0.8)
    nx.draw_networkx_nodes(g,pos,node_size=150,node_color='r',node_shape='o',alpha=0.8)
    nx.draw_networkx_edges(g,pos)
    nx.draw_networkx_labels(g,pos,font_size=8)
    plt.axis('off')
    plt.savefig(g.name+'.pdf')
plt.show()




