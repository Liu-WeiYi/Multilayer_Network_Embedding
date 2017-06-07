#!/usr/bin/python
# -*- coding: utf-8 -*-

import Reader
import random, math
def single_sampling(path, p):
    g = Reader.single_readG(path)
    Removelist = []
    shuf_edges = g.edges()
    random.shuffle(shuf_edges)

    for edge in shuf_edges:
        r = random.random()
        if r < p:
            if len(g.neighbors(edge[0])) == 1 or len(g.neighbors(edge[1]))  == 1:
                continue
            Removelist.append(edge)
            g.remove_edge(*edge)


    return Removelist, g

def multi_sampling(path, p):
    graphs, n_edges = Reader.multi_readG(path)
    Removelist = []
    portion = math.ceil((n_edges * p)/len(graphs))

    for g in graphs:
        shuf_edges = g.edges()
        random.shuffle(shuf_edges)
        for edge in shuf_edges:
            r = random.random()
            if r < p:
                if len(g.neighbors(edge[0])) == 1 or len(g.neighbors(edge[1]))  == 1:
                    continue
                Removelist.append(edge)
                g.remove_edge(*edge)
            if len(Removelist) >= portion:
                break


    return Removelist, graphs

def multi_sampling_with_Merg (path, p):
    m_graph, nx_graphs, _ = Reader.multi_readG_with_Merg(path)
    Removelist = []
    shuf_edges = m_graph.edges()
    random.shuffle(shuf_edges)

    for edge in shuf_edges:
        r = random.random()
        if r < p:
            if len(m_graph.neighbors(edge[0])) == 1 or len(m_graph.neighbors(edge[1]))  == 1:
                continue
            Removelist.append(edge)
            m_graph.remove_edge(*edge)

    #print(len(nx_graphs[0].edges()))
    for edge in Removelist:
        for g in nx_graphs:
            if g.has_edge(*edge):
                g.remove_edge(*edge)
    #print(len(nx_graphs[0].edges()))
    return Removelist, m_graph, nx_graphs
