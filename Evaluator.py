#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import random, collections, itertools, math, link_pred, sys

def edge_in_graphs(edge, Original_graphs):
    if isinstance(Original_graphs, list):
        for g in Original_graphs:
            if edge in g.edges():
                return True
        return False
    else:
        if edge in Original_graphs.edges():
            return True
        else:
            return False

class Precision_Eval:
    def __init__(self, matrix, mapping, Original_graphs, Permutation_list, Examping_prob):
        self.mx = matrix
        self.mp = mapping
        self.ori_g = Original_graphs
        self.perm_ls = Permutation_list
        self.ep = Examping_prob

    def check(self, src, dst):
        mapping = self.mp
        matrix = self.mx
        r = np.linalg.norm(matrix[mapping[src]] - matrix[mapping[dst]])
        return r

    def eval(self):
        ori_g = self.ori_g
        perm_ls = self.perm_ls
        ep = self.ep
        result = {}
        s = 0
        for edge in itertools.combinations(perm_ls, 2):
            if edge[0] not in self.mp.keys():
                #print('found a pair not in mapping ', edge[0])
                continue
            if edge[1] not in self.mp.keys():
                #print('found a pair not in mapping ', edge[1])
                continue
            dist = self.check(edge[0], edge[1])
            s += dist
            result[edge] = dist

        normalized_result = {}
        for e in result:
            normalized_result[e] = result[e] / s

        sorted_result = collections.OrderedDict(sorted(normalized_result.items(), key=lambda t: t[1]))

        n = len(normalized_result) * ep
        TP = 0
        count = 1
        print ('getting the first ', n)

        for edge in sorted_result:
            #print (edge)
            if edge_in_graphs(edge, ori_g):
                TP += 1

            count += 1
            if count > n:
                break
        print ('TP', TP)
        FN = count - TP
        FP = 0
        TN = 0
        print('FN', FN)

        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        F = 2*TP/(2*TP + FP + FN)
        return precision, recall, F


    def edge_list_eval(self, airport_dst, airport_mapping):
        TP = 0
        count = 0
        true_distance = {}
        true_cons = 0
        est_distance = {}
        est_cons = 0
        for g in self.ori_g:
            for edge in g.edges():
                if edge[0] not in self.mp.keys() or edge[1] not in self.mp.keys():
                    continue
                true_src = airport_dst[airport_mapping[int(edge[0])]]
                true_dst = airport_dst[airport_mapping[int(edge[1])]]
                true_distance[edge] = math.sqrt(math.pow(float(true_src[0]) - float(true_dst[0]), 2) + math.pow(float(true_src[0]) - float(true_dst[0]), 2))
                true_cons += true_distance[edge]
                est_distance[edge] = self.check(edge[0], edge[1])
                est_cons += est_distance[edge]

        normalized_true = {}
        normalized_est = {}
        for e in true_distance:
            normalized_true[e] = true_distance[e] / true_cons
            normalized_est[e] = est_distance[e] / est_cons

        sorted_true = collections.OrderedDict(sorted(normalized_true.items(), key=lambda t: t[1]))
        sorted_est = collections.OrderedDict(sorted(normalized_est.items(), key=lambda t: t[1]))

        n = math.ceil(len(normalized_true) * self.ep)
        count = 1
        TP = 0
        for edge in sorted_true:
            if edge in list(sorted_est.keys())[:n]:
                TP+=1
            count+=1
            if count > n:
                break

        print ('TP', TP)
        FN = count - TP
        FP = 0
        TN = 0
        print('FN', FN)

        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        F = 2*TP/(2*TP + FP + FN)
        return precision, recall, F

class combining_Precision_Eval:

    def __init__(self, matrix_dic, mapping_dic, original_graphs, perm_list, examing_prob):
        self.mxd = matrix_dic
        self.mpd = mapping_dic
        self.ori_gs = original_graphs
        self.perm_ls = perm_list
        self.ep = examing_prob

    def check(self, src, dst):
        mapping = self.mpd
        matrix = self.mxd
        src_score = np.array([])
        dst_score = np.array([])
        for g in mapping:
            if src in mapping[g].keys():
                src_score = np.append(src_score, matrix[g][mapping[g][src]])
            if dst in mapping[g].keys():
                dst_score = np.append(dst_score, matrix[g][mapping[g][dst]])

        if src_score.shape != dst_score.shape:
            if src_score.shape > dst_score.shape:
                diff = src_score.shape[0] - dst_score.shape[0]
                dst_score = np.append(dst_score, np.zeros(diff))
            else:
                diff = dst_score.shape[0] - src_score.shape[0]
                src_score = np.append(src_score, np.zeros(diff))

        return np.linalg.norm(src_score - dst_score)

    def eval(self):
        ori_g = self.ori_gs
        perm_ls = self.perm_ls
        ep = self.ep
        result = {}
        s = 0
        for edge in itertools.combinations(perm_ls, 2):
            dist = self.check(edge[0], edge[1])
            s += dist
            result[edge] = dist

        normalized_result = {}
        for e in result:
            normalized_result[e] = result[e] / s

        sorted_result = collections.OrderedDict(sorted(normalized_result.items(), key=lambda t: t[1]))

        n = len(normalized_result) * ep
        TP = 0
        count = 1
        #print ('getting the first ', n)

        for edge in sorted_result:
            if edge_in_graphs(edge, ori_g):
                TP += 1

            count += 1
            if count > n:
                break
        #print ('correct', correct)
        print ('TP', TP)
        FN = count - TP
        FP = 0
        TN = 0
        print('FN', FN)

        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        F = 2*TP/(2*TP + FP + FN)
        return precision, recall, F




class AUC_Eval:
    def __init__(self, matrix, mapping, Original_graphs, Sampled_graphs):
        self.mx = matrix
        self.mp = mapping
        self.ori_g = Original_graphs
        self.samp_g = Sampled_graphs

    def check(self, src, dst):
        mapping = self.mp
        matrix = self.mx
        r = np.linalg.norm(matrix[mapping[src]] - matrix[mapping[dst]])
        ### reciprocal
        return 1.0/r


    def eval_auc(self, f):
        ori_g = self.ori_g
        samp_g = self.samp_g

        #### Step 1: classification
        unlinked_pair = []
        linked_pair = []
        if isinstance(ori_g, list):
            if f == 0:
                for g in ori_g:
                    for s_g in samp_g:
                        for edge in itertools.combinations(s_g.nodes(), 2):
                            if edge[0] not in self.mp.keys():
                                #print('found a pair not in mapping ', edge[0])
                                continue
                            if edge[1] not in self.mp.keys():
                                #print('found a pair not in mapping ', edge[1])
                                continue
                            if edge not in s_g.edges():
                                if edge in g.edges():
                                    linked_pair.append(self.check(edge[0], edge[1]))
                                else:
                                    unlinked_pair.append(self.check(edge[0], edge[1]))

            else:
                for g in ori_g:
                    indx = ori_g.index(g)
                    s_g = samp_g[indx]
                    for edge in itertools.combinations(s_g.nodes(), 2):
                        if edge[0] not in self.mp.keys():
                            #print('found a pair not in mapping ', edge[0])
                            continue
                        if edge[1] not in self.mp.keys():
                            #print('found a pair not in mapping ', edge[1])
                            continue
                        if edge not in s_g.edges():
                            if edge in g.edges():
                                linked_pair.append(self.check(edge[0], edge[1]))
                            else:
                                unlinked_pair.append(self.check(edge[0], edge[1]))

        else:
            for edge in itertools.combinations(samp_g.nodes(), 2):
                if edge[0] not in self.mp.keys():
                    #print('found a pair not in mapping ', edge[0])
                    continue
                if edge[1] not in self.mp.keys():
                    #print('found a pair not in mapping ', edge[1])
                    continue
                if edge not in samp_g.edges():
                    if edge in ori_g.edges():
                        linked_pair.append(self.check(edge[0], edge[1]))
                    else:
                        unlinked_pair.append(self.check(edge[0], edge[1]))

        print ('link pair', len(linked_pair))
        print ('unlinked pair', len(unlinked_pair))
        #print(unlinked_pair)
        auc = 0.0
        freq = min(len(unlinked_pair), len(linked_pair))
        ##### Step 2: calculation

        for fre in range(0, freq):
            unlinked_score = float(unlinked_pair[random.randint(0, freq - 1)])
            linked_score = float(linked_pair[random.randint(0, freq - 1)])
            if linked_score > unlinked_score:
                auc += 1.0
            elif linked_score == unlinked_score:
                auc += 0.5
        print(auc)
        return auc/ freq

class combining_AUC_Eval:

    def __init__(self, matrix_dic, mapping_dic, original_graphs, sampled_graphs):
        self.mxd = matrix_dic
        self.mpd = mapping_dic
        self.ori_g = original_graphs
        self.samp_g = sampled_graphs

    def check(self, src, dst):
        mapping = self.mpd
        matrix = self.mxd
        src_score = np.array([])
        dst_score = np.array([])
        for g in mapping:
            if src in mapping[g].keys():
                src_score = np.append(src_score, matrix[g][mapping[g][src]])
            if dst in mapping[g].keys():
                dst_score = np.append(dst_score, matrix[g][mapping[g][dst]])

        if src_score.shape != dst_score.shape:
            if src_score.shape > dst_score.shape:
                diff = src_score.shape[0] - dst_score.shape[0]
                dst_score = np.append(dst_score, np.zeros(diff))
            else:
                diff = dst_score.shape[0] - src_score.shape[0]
                src_score = np.append(src_score, np.zeros(diff))

        return 1.0/np.linalg.norm(src_score - dst_score)

    def eval_auc(self, f):
        ori_g = self.ori_g
        samp_g = self.samp_g

        #### Step 1: classification
        unlinked_pair = []
        linked_pair = []
        if f == 0:
            for g in ori_g:
                for s_g in samp_g:
                    for edge in itertools.combinations(s_g.nodes(), 2):
                        if edge not in s_g.edges():
                            if edge in g.edges():
                                linked_pair.append(self.check(edge[0], edge[1]))
                            else:
                                unlinked_pair.append(self.check(edge[0], edge[1]))

        else:
            for g in ori_g:
                indx = ori_g.index(g)
                s_g = samp_g[indx]
                for edge in itertools.combinations(s_g.nodes(), 2):
                    if edge not in s_g.edges():
                        if edge in g.edges():
                            linked_pair.append(self.check(edge[0], edge[1]))
                        else:
                            unlinked_pair.append(self.check(edge[0], edge[1]))

        print ('link pair', len(linked_pair))
        print ('unlinked pair', len(unlinked_pair))
        auc = 0.0
        freq = min(len(unlinked_pair), len(linked_pair))
        for fre in range(0, freq):
            unlinked_score = float(unlinked_pair[random.randint(0, freq - 1)])
            linked_score = float(linked_pair[random.randint(0, freq - 1)])
            if linked_score > unlinked_score:
                auc += 1.0
            elif linked_score == unlinked_score:
                auc += 0.5

        return auc/ freq
