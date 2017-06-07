#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random, collections
import pandas as pd
import numpy.matlib


class Prediction:

    def create_vertex(self, nodepair_set):
        vertex_set = {}
        num = 0
        for i in nodepair_set:
            if i[0] not in vertex_set:
                vertex_set[i[0]] = num
                num += 1
            if i[1] not in vertex_set:
                vertex_set[i[1]] = num
                num += 1
        return vertex_set

    def create_adjmatrix(self, nodepair_set, vertex_set):
        init_matrix = np.zeros([len(vertex_set), len(vertex_set)])
        for pair in nodepair_set:
            if pair[0] in vertex_set and pair[1] in vertex_set:
                init_matrix[vertex_set[pair[0]]][vertex_set[pair[1]]] = 1
                init_matrix[vertex_set[pair[1]]][vertex_set[pair[0]]] = 1

        return init_matrix

    def auc_score(
        self,
        matrix_score,
        matrix_test,
        matrix_train,
        n_compare=10,
        ):

        if type(n_compare) == int:
            if len(matrix_test[0]) < 2:
                raise Exception('Invalid ndim!', train.ndim)
            elif len(matrix_test[0]) < 10:
                n_compare = len(matrix_test[0])
        else:
            if n_compare != 'cc':
                raise Exception('Invalid n_compare!', n_compare)

        unlinked_pair = []
        linked_pair = []

        l = 1
        for i in range(0, len(matrix_test)):
            for j in range(0, l):
                if i != j and matrix_train[i][j] != 1:
                    if matrix_test[i][j] == 1:
                        linked_pair.append(matrix_score[i, j])
                    elif matrix_test[i][j] == 0:
                        unlinked_pair.append(matrix_score[i, j])
                    else:
                        raise Exception('Invalid connection!',
                                matrix_test[i][j])
            l += 1
        print ('link pair', len(linked_pair))
        print ('unlinked pair', len(unlinked_pair))
        auc = 0.0
        if n_compare == 'cc':
            frequency = min(len(unlinked_pair), len(linked_pair))
        else:
            frequency = n_compare
        for fre in range(0, frequency):
            unlinked_score = float(unlinked_pair[random.randint(0,frequency - 1)])
            linked_score = float(linked_pair[random.randint(0, frequency - 1)])
            if linked_score > unlinked_score:
                auc += 1.0
            elif linked_score == unlinked_score:
                auc += 0.5
        print ('auc before frequency', auc)
        auc = auc / frequency

        return auc

    def acc(self, matrix_score, matrix_ori, perm_list, ep):
        result = {}
        s = 0
        l = 1
        for i in range(0, len(perm_list)):
            for j in range(l, len(perm_list)):
                if matrix_score[i, j] != 0:
                    dist = matrix_score[i, j]
                    s += dist
                    result[(i, j)] = dist
            l += 1

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
            if matrix_ori[edge[0]][edge[1]] > 0:
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


class similarity(object):

    """docstring for  similarity"""

    def fit(self, train_adj):

        train = np.matrix(train_adj)
        if train.ndim < 2:
            raise Exception('Invalid ndim!', train.ndim)
        if train.size < 2:
            raise Exception('Invalid size!', train.size)
        if train.shape[0] != train.shape[1]:
            raise Exception('Invalid shape!', train.shape)


class CommonNeighbors(similarity):

    """
            CommonNeighbors
    """

    def fit(self, train_adj):
        similarity.fit(self, train_adj)
        train_adj = np.matrix(train_adj)

        return train_adj * train_adj


class Jaccard(similarity):

    def fit(self, train_adj):
        similarity.fit(self, train_adj)
        train_adj = np.matrix(train_adj)
        numerator = train_adj * train_adj
        deg0 = np.matlib.repmat(train_adj.sum(0), len(train_adj), 1)
        deg1 = np.matlib.repmat(train_adj.sum(1), 1, len(train_adj))
        denominator = (deg0 + deg1) - numerator
        sim = numerator / denominator
        sim[np.isnan(sim)] = 0
        sim[np.isinf(sim)] = 0
        return sim


