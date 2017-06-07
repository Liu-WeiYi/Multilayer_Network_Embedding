import networkx as nx
import pickle, sys, os



def read_data(filename, path):
    """Extract the file and convert it into a neighbor list"""

    g = nx.Graph(name=filename)
    for line in open(path + filename):
        #(s, d, _) = line.split(' ')
        (s, d) = line.split(' ')
        #(s,d) = line.split('\t')
        src = s
        dst = d.split('\n')[0]
        g.add_edge(src, dst)

    pickle.dump(g, open(path+filename+'nx_graph.pickle','+wb'))


if __name__ == "__main__":

    path = sys.argv[1]
    files = os.listdir(path)
    for name in files:
        if name.endswith(".txt"):
            read_data(name, path)
